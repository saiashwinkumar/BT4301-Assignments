
from fastapi import FastAPI, HTTPException, Request
from mlflow.tracking import MlflowClient

import itertools
import json
import math
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd


app = FastAPI(
    title="Sales Demand Prediction API",
    description=(
        "Deploys the best Random Forest and Gradient Boosting models from MLflow "
        "and supports live A/B testing."
    ),
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:9080")
MLARTIFACTS_DIR = os.getenv("MLARTIFACTS_DIR", "/root/mlartifacts")
MODEL_SELECTION_METRIC = os.getenv("MODEL_SELECTION_METRIC", "test_rmse")
REQUIRE_REDUCED_SHAP_MODELS = os.getenv("REQUIRE_REDUCED_SHAP_MODELS", "true").lower() == "true"
REDUCED_FEATURE_SELECTION_TAG = "shap_cumulative_95_raw_features"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODELS: Dict[str, Dict[str, Any]] = {}
AB_ORDER = ["gradient_boosting", "random_forest"]
AB_COUNTER = itertools.count()


# -----------------------------
# Model discovery and loading
# -----------------------------
def run_belongs_to_model(run, model_key: str) -> bool:
    tags = {k.lower(): str(v).lower() for k, v in run.data.tags.items()}
    params = {k.lower(): str(v).lower() for k, v in run.data.params.items()}

    run_name = tags.get("mlflow.runname", "")
    model_key_tag = tags.get("model_key", "")
    model_family = tags.get("model_family", "")
    training_info = tags.get("training info", "")
    artifact_path = tags.get("artifact_path", "")

    searchable_text = " ".join([
        run_name,
        model_key_tag,
        model_family,
        training_info,
        artifact_path,
        " ".join(params.values()),
    ])

    if model_key == "random_forest":
        return (
            model_key_tag == "random_forest"
            or "random_forest" in searchable_text
            or "random forest" in searchable_text
            or model_family == "tree_ensemble"
        )

    if model_key == "gradient_boosting":
        return (
            model_key_tag == "gradient_boosting"
            or "gradient_boosting" in searchable_text
            or "gradient boosting" in searchable_text
            or model_family == "boosting"
        )

    return False


def get_artifact_path_for_model(model_key: str, run) -> str:
    tag_artifact_path = run.data.tags.get("artifact_path")
    if tag_artifact_path:
        return tag_artifact_path

    if model_key == "random_forest":
        return "random_forest_model"

    if model_key == "gradient_boosting":
        return "gradient_boosting_model"

    raise ValueError(f"Unknown model_key: {model_key}")


def metric_value(run, metric_name: str) -> float:
    value = run.data.metrics.get(metric_name)
    if value is None:
        return math.inf
    return float(value)

def is_reduced_shap_run(run) -> bool:
    """
    Ensures the API selects only the SHAP 95% reduced-feature runs,
    instead of accidentally loading older full-feature models.
    """
    feature_selection_tag = str(run.data.tags.get("feature_selection", "")).lower()
    feature_selection_param = str(run.data.params.get("feature_selection_method", "")).lower()
    reduced_count = run.data.params.get("reduced_feature_count")

    return (
        feature_selection_tag == REDUCED_FEATURE_SELECTION_TAG
        or "shap" in feature_selection_param
        or reduced_count is not None
    )


def get_selected_features_from_run(run):
    selected_features_param = run.data.params.get("selected_features", "")
    if selected_features_param:
        return [
            feature.strip()
            for feature in selected_features_param.split(",")
            if feature.strip()
        ]
    return []

def find_best_run_for_model(model_key: str):
    client = MlflowClient()
    experiments = client.search_experiments()
    experiment_ids = [exp.experiment_id for exp in experiments]

    runs = client.search_runs(
        experiment_ids=experiment_ids,
        order_by=[f"metrics.{MODEL_SELECTION_METRIC} ASC"],
        max_results=500,
    )

    candidate_runs = [
        run
        for run in runs
        if run_belongs_to_model(run, model_key)
        and MODEL_SELECTION_METRIC in run.data.metrics
    ]
    if REQUIRE_REDUCED_SHAP_MODELS:
        candidate_runs = [
            run for run in candidate_runs
            if is_reduced_shap_run(run)
        ]
    if not candidate_runs:
        raise RuntimeError(
            f"No reduced SHAP MLflow runs found for {model_key} with metric "
            f"{MODEL_SELECTION_METRIC}. Rerun the updated RF/GB training scripts, "
            "or set REQUIRE_REDUCED_SHAP_MODELS=false only if you intentionally want "
            "to deploy full-feature models."
        )

    return sorted(candidate_runs, key=lambda r: metric_value(r, MODEL_SELECTION_METRIC))[0]


def mlflow_artifact_uri_to_local_path(uri: str) -> Optional[Path]:
    if uri.startswith("file://"):
        return Path(uri.replace("file://", "", 1))

    if uri.startswith("mlflow-artifacts:"):
        # Example:
        # mlflow-artifacts:/3/<run-id>/artifacts
        # becomes /root/mlartifacts/3/<run-id>/artifacts
        suffix = uri.replace("mlflow-artifacts:", "", 1).lstrip("/")
        return Path(MLARTIFACTS_DIR) / suffix

    return None


def find_local_model_dir(run, artifact_path: str) -> Optional[Path]:
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id

    candidate_roots = []

    local_artifact_root = mlflow_artifact_uri_to_local_path(run.info.artifact_uri)
    if local_artifact_root:
        candidate_roots.append(local_artifact_root)

    candidate_roots.append(Path(MLARTIFACTS_DIR) / str(experiment_id))
    candidate_roots.append(Path(MLARTIFACTS_DIR))

    for root in candidate_roots:
        if not root.exists():
            continue

        # Prefer MLmodel folders that explicitly reference the selected run id.
        for mlmodel_path in root.rglob("MLmodel"):
            try:
                text = mlmodel_path.read_text(errors="ignore")
            except Exception:
                text = ""

            model_dir = mlmodel_path.parent
            has_model_pickle = (model_dir / "model.pkl").exists()
            path_matches_artifact = artifact_path in str(model_dir)

            if has_model_pickle and (run_id in text or path_matches_artifact):
                return model_dir

        # Fallback: any model folder under the artifact path.
        for model_pickle_path in root.rglob("model.pkl"):
            if artifact_path in str(model_pickle_path):
                return model_pickle_path.parent

    return None


def load_model_from_best_run(model_key: str):
    best_run = find_best_run_for_model(model_key)
    run_id = best_run.info.run_id
    artifact_path = get_artifact_path_for_model(model_key, best_run)

    model = None
    model_uri_used = None

    # First try normal MLflow loading.
    model_uri = f"runs:/{run_id}/{artifact_path}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
        model_uri_used = model_uri
    except Exception as e:
        print(f"Could not load {model_key} through {model_uri}: {e}")

    # Fallback for local Docker setup where artifacts are mounted under /root/mlartifacts.
    if model is None:
        local_model_dir = find_local_model_dir(best_run, artifact_path)
        if local_model_dir is None:
            raise RuntimeError(
                f"Could not find local model artifacts for {model_key}, run_id={run_id}."
            )

        try:
            model = mlflow.sklearn.load_model(str(local_model_dir))
            model_uri_used = str(local_model_dir)
        except Exception:
            with open(local_model_dir / "model.pkl", "rb") as f:
                model = pickle.load(f)
            model_uri_used = str(local_model_dir / "model.pkl")

    selected_features_from_run = get_selected_features_from_run(best_run)

    if hasattr(model, "feature_names_in_"):
        required_cols = list(model.feature_names_in_)
    elif selected_features_from_run:
        required_cols = selected_features_from_run
    else:
        required_cols = []

    return {
        "model_key": model_key,
        "display_name": "Gradient Boosting" if model_key == "gradient_boosting" else "Random Forest",
        "model": model,
        "required_cols": required_cols,
        "selected_features_from_run": selected_features_from_run,
        "run_id": run_id,
        "experiment_id": best_run.info.experiment_id,
        "metric_name": MODEL_SELECTION_METRIC,
        "metric_value": metric_value(best_run, MODEL_SELECTION_METRIC),
        "model_uri": model_uri_used,
        "artifact_path": artifact_path,
    }


@app.on_event("startup")
def load_models():
    global MODELS

    loaded_models = {}
    errors = {}

    for model_key in AB_ORDER:
        try:
            loaded_models[model_key] = load_model_from_best_run(model_key)
            print(
                f"Loaded {model_key}: run_id={loaded_models[model_key]['run_id']}, "
                f"{MODEL_SELECTION_METRIC}={loaded_models[model_key]['metric_value']}, "
                f"features={loaded_models[model_key]['required_cols']}"
            )
        except Exception as e:
            errors[model_key] = str(e)
            print(f"Error loading {model_key}: {e}")

    if not loaded_models:
        print(f"No models loaded. Errors: {errors}")

    MODELS = loaded_models


# -----------------------------
# Request parsing and feature engineering
# -----------------------------
def parse_request_to_dataframe(req_json: Any) -> pd.DataFrame:
    if (
        isinstance(req_json, dict)
        and "columns" in req_json
        and isinstance(req_json.get("data", []), list)
        and len(req_json["data"]) > 0
        and isinstance(req_json["data"][0], list)
    ):
        return pd.DataFrame(req_json["data"], columns=req_json["columns"])

    if (
        isinstance(req_json, dict)
        and "data" in req_json
        and isinstance(req_json.get("data", []), list)
        and len(req_json["data"]) > 0
        and isinstance(req_json["data"][0], dict)
    ):
        return pd.DataFrame(req_json["data"])

    if isinstance(req_json, list):
        return pd.DataFrame(req_json)

    if isinstance(req_json, dict):
        return pd.DataFrame([req_json])

    raise ValueError("Unsupported request JSON format.")


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "month_num" in df.columns:
        month_num = pd.to_numeric(df["month_num"], errors="coerce")

        df["quarter_num"] = np.ceil(month_num / 3).astype("Int64")
        df["is_q4"] = np.where(df["quarter_num"] == 4, 1, 0)
        df["is_year_end_season"] = np.where(month_num.isin([11, 12]), 1, 0)
        df["is_mid_year"] = np.where(month_num.isin([6, 7]), 1, 0)
        df["month_sin"] = np.sin(2 * np.pi * month_num / 12)
        df["month_cos"] = np.cos(2 * np.pi * month_num / 12)

    if {"current_month_discount_amount", "current_month_gross_sales"}.issubset(df.columns):
        gross = pd.to_numeric(df["current_month_gross_sales"], errors="coerce")
        discount = pd.to_numeric(df["current_month_discount_amount"], errors="coerce")
        df["discount_percentage"] = np.where(gross != 0, discount / gross, 0)

    if {"current_month_margin_amount", "current_month_net_sales"}.issubset(df.columns):
        margin = pd.to_numeric(df["current_month_margin_amount"], errors="coerce")
        net_sales = pd.to_numeric(df["current_month_net_sales"], errors="coerce")
        df["product_profitability_ratio"] = np.where(net_sales != 0, margin / net_sales, 0)

    if {"current_month_net_sales", "previous_month_sales"}.issubset(df.columns):
        current = pd.to_numeric(df["current_month_net_sales"], errors="coerce")
        previous = pd.to_numeric(df["previous_month_sales"], errors="coerce")
        df["sales_growth_previous_period"] = np.where(
            previous != 0,
            (current - previous) / previous,
            0,
        )

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def choose_ab_model_key(explicit_model_key: Optional[str] = None) -> str:
    if explicit_model_key:
        if explicit_model_key not in MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Requested model_key '{explicit_model_key}' is not loaded. Available: {list(MODELS.keys())}",
            )
        return explicit_model_key

    available_order = [model_key for model_key in AB_ORDER if model_key in MODELS]
    if not available_order:
        raise HTTPException(status_code=500, detail="No models are loaded.")

    next_index = next(AB_COUNTER) % len(available_order)
    return available_order[next_index]


# -----------------------------
# API endpoints
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "healthy" if bool(MODELS) else "no_models_loaded",
        "loaded_models": list(MODELS.keys()),
        "model_selection_metric": MODEL_SELECTION_METRIC,
    }


@app.get("/models")
def list_models():
    return {
        "ab_order": [model_key for model_key in AB_ORDER if model_key in MODELS],
        "models": {
            model_key: {
                key: value
                for key, value in bundle.items()
                if key != "model"
            }
            for model_key, bundle in MODELS.items()
        },
    }


@app.get("/next-model")
def next_model_preview():
    available_order = [model_key for model_key in AB_ORDER if model_key in MODELS]
    if not available_order:
        raise HTTPException(status_code=500, detail="No models are loaded.")

    # Preview only. Does not increment the API-side A/B counter.
    preview_key = available_order[0]
    return {
        "next_model_key": preview_key,
        "model_info": {
            key: value
            for key, value in MODELS[preview_key].items()
            if key != "model"
        },
    }


@app.post("/predict")
async def predict(request: Request):
    if not MODELS:
        raise HTTPException(status_code=500, detail="No models loaded from MLflow.")

    try:
        req_json = await request.json()

        explicit_model_key = None
        if isinstance(req_json, dict):
            explicit_model_key = req_json.get("model_key")

        model_key = choose_ab_model_key(explicit_model_key)
        model_bundle = MODELS[model_key]

        df = parse_request_to_dataframe(req_json)
        df = df.drop(columns=["model_key"], errors="ignore")

        if df.empty:
            raise ValueError("Unable to parse input data into tabular format.")

        df = derive_features(df)

        required_cols = model_bundle["required_cols"]

        missing_before_fill = [col for col in required_cols if col not in df.columns]
        for col in missing_before_fill:
            df[col] = None

        df = df[required_cols]

        predictions = model_bundle["model"].predict(df)

        return {
            "predictions": [float(x) for x in predictions],
            "model_key": model_key,
            "model_display_name": model_bundle["display_name"],
            "model_run_id": model_bundle["run_id"],
            "model_metric_name": model_bundle["metric_name"],
            "model_metric_value": model_bundle["metric_value"],
            "required_features": required_cols,
            "missing_features_filled_with_null": missing_before_fill,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
