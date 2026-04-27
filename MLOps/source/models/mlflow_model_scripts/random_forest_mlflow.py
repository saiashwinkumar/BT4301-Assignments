"""
MLflow experiment script for sales demand forecasting.

This script is based on the modelling workflow in assignment2_ml.ipynb:
- Target: next_period_net_sales
- Train/validation/test split: 70% / 10% / 20%
- Preprocessing:
    - Numeric: SimpleImputer(strategy="median") + StandardScaler()
    - Categorical: SimpleImputer(strategy="most_frequent") + OneHotEncoder(handle_unknown="ignore")
- MLflow tracking URI default: http://localhost:9080

Run example:
    python random_forest_mlflow.py

Optional CSV usage:
    python random_forest_mlflow.py --csv-path ../data/ml_dataset.csv

If --csv-path is not provided, the script loads the dataset from the local MySQL
datawarehouse database used in the notebook.
"""

import argparse
import os
import warnings

import json
import tempfile
from pathlib import Path

import shap
from scipy import sparse

import mysql.connector
import numpy as np
import pandas as pd

import mlflow
from mlflow.models import infer_signature

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore", category=FutureWarning)


SQL_QUERY = """
WITH monthly_sales AS (
    SELECT
        p.product_key,
        p.product_id,
        p.product_name,
        p.product_category_name,
        p.product_subcategory_name,
        p.product_model_name,
        p.color,
        p.size,
        p.weight,
        p.standard_cost,
        p.list_price,

        c.customer_key,
        c.customer_id,
        c.store_id,
        c.store_name,
        c.territory_id,

        sp.salesperson_key,
        sp.salesperson_id,
        sp.sales_quota,
        sp.bonus,
        sp.commission_pct,
        sp.sales_ytd,
        sp.sales_last_year,

        t.year_num,
        t.month_num,
        t.quarter_num,

        SUM(f.order_qty) AS current_month_order_qty,
        SUM(f.gross_amount) AS current_month_gross_sales,
        SUM(f.discount_amount) AS current_month_discount_amount,
        SUM(f.net_amount) AS current_month_net_sales,
        SUM(f.margin_amount) AS current_month_margin_amount,

        AVG(f.unit_price) AS avg_unit_price,
        AVG(f.unit_price_discount) AS avg_unit_price_discount,

        COUNT(DISTINCT f.sales_order_id) AS number_of_orders,
        COUNT(*) AS number_of_order_lines

    FROM fact_store_sales f
    JOIN dim_product p
        ON f.product_key = p.product_key
    JOIN dim_store_customer c
        ON f.customer_key = c.customer_key
    JOIN dim_time t
        ON f.order_date_key = t.time_key
    JOIN dim_salesperson sp
        ON f.salesperson_key = sp.salesperson_key

    GROUP BY
        p.product_key,
        p.product_id,
        p.product_name,
        p.product_category_name,
        p.product_subcategory_name,
        p.product_model_name,
        p.color,
        p.size,
        p.weight,
        p.standard_cost,
        p.list_price,

        c.customer_key,
        c.customer_id,
        c.store_id,
        c.store_name,
        c.territory_id,

        sp.salesperson_key,
        sp.salesperson_id,
        sp.sales_quota,
        sp.bonus,
        sp.commission_pct,
        sp.sales_ytd,
        sp.sales_last_year,

        t.year_num,
        t.month_num,
        t.quarter_num
),

ml_dataset AS (
    SELECT
        monthly_sales.*,

        LEAD(current_month_net_sales) OVER (
            PARTITION BY product_key, store_id
            ORDER BY year_num, month_num
        ) AS next_period_net_sales

    FROM monthly_sales
)

SELECT *
FROM ml_dataset
WHERE next_period_net_sales IS NOT NULL
ORDER BY product_id, store_id, year_num, month_num;
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Run tuned Random Forest experiment and register best model in MLflow.")
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Optional path to a CSV extract of the ML dataset. If omitted, MySQL is used.",
    )
    parser.add_argument(
        "--tracking-uri",
        default="http://localhost:9080",
        help="MLflow tracking server URI.",
    )
    parser.add_argument(
        "--experiment-name",
        default="Sales Demand Forecasting - Random Forest",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--registered-model-name",
        default="sales_random_forest",
        help="Name to use in the MLflow Model Registry.",
    )


    parser.add_argument(
        "--shap-sample-size",
        type=int,
        default=1000,
        help="Maximum number of training rows to use for SHAP importance calculation.",
    )
    parser.add_argument(
        "--shap-cumulative-coverage",
        type=float,
        default=0.95,
        help="Select features until this cumulative share of absolute SHAP importance is captured.",
    )
    parser.add_argument(
        "--reference-output-dir",
        default=os.getenv("REFERENCE_OUTPUT_DIR", "/root/webapp"),
        help="Folder where reduced reference input data will be exported for the Streamlit drift dashboard.",
    )

    # MySQL defaults match the notebook.
    parser.add_argument("--db-host", default=os.getenv("DB_HOST", "localhost"))
    parser.add_argument("--db-user", default=os.getenv("DB_USER", "bt4301"))
    parser.add_argument("--db-password", default=os.getenv("DB_PASSWORD", "password"))
    parser.add_argument("--db-name", default=os.getenv("DB_NAME", "datawarehouse"))

    return parser.parse_args()


def load_data(args):
    if args.csv_path:
        print(f"Loading dataset from CSV: {args.csv_path}")
        return pd.read_csv(args.csv_path)

    print(f"Loading dataset from MySQL database: {args.db_name}@{args.db_host}")
    connection = mysql.connector.connect(
        host=args.db_host,
        user=args.db_user,
        passwd=args.db_password,
        database=args.db_name,
    )

    try:
        return pd.read_sql(sql=SQL_QUERY, con=connection)
    finally:
        connection.close()


def engineer_features(df):
    df = df.copy()

    df["year_month_date"] = pd.to_datetime(
        df["year_num"].astype(str) + "-" + df["month_num"].astype(str).str.zfill(2) + "-01"
    )

    df = df.sort_values(["product_key", "store_id", "year_month_date"])
    group_cols = ["product_key", "store_id"]

    if "previous_month_sales" not in df.columns:
        df["previous_month_sales"] = (
            df.groupby(group_cols)["current_month_net_sales"].shift(1)
        )

    if "rolling_avg_sales_3m" not in df.columns:
        df["rolling_avg_sales_3m"] = (
            df.groupby(group_cols)["current_month_net_sales"]
            .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
        )

    if "previous_month_margin" not in df.columns:
        df["previous_month_margin"] = (
            df.groupby(group_cols)["current_month_margin_amount"].shift(1)
        )

    if "discount_percentage" not in df.columns:
        df["discount_percentage"] = np.where(
            df["current_month_gross_sales"] != 0,
            df["current_month_discount_amount"] / df["current_month_gross_sales"],
            0,
        )

    if "product_profitability_ratio" not in df.columns:
        df["product_profitability_ratio"] = np.where(
            df["current_month_net_sales"] != 0,
            df["current_month_margin_amount"] / df["current_month_net_sales"],
            0,
        )

    if "is_q4" not in df.columns:
        df["is_q4"] = np.where(df["quarter_num"] == 4, 1, 0)

    if "is_year_end_season" not in df.columns:
        df["is_year_end_season"] = np.where(df["month_num"].isin([11, 12]), 1, 0)

    if "is_mid_year" not in df.columns:
        df["is_mid_year"] = np.where(df["month_num"].isin([6, 7]), 1, 0)

    if "month_sin" not in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)

    if "month_cos" not in df.columns:
        df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

    if "sales_growth_previous_period" not in df.columns:
        df["sales_growth_previous_period"] = np.where(
            df["previous_month_sales"] != 0,
            (df["current_month_net_sales"] - df["previous_month_sales"])
            / df["previous_month_sales"],
            0,
        )

    df = df.replace([np.inf, -np.inf], np.nan)

    target = "next_period_net_sales"
    df_model = df.dropna(subset=[target]).copy()

    df_model["previous_month_sales"] = df_model["previous_month_sales"].fillna(0)
    df_model["previous_month_margin"] = df_model["previous_month_margin"].fillna(0)
    df_model["rolling_avg_sales_3m"] = df_model["rolling_avg_sales_3m"].fillna(
        df_model["current_month_net_sales"]
    )

    return df_model


def split_dataset(df_model):
    target = "next_period_net_sales"

    drop_cols = [
        target,
        "year_month_date",
        "product_id",
        "product_name",
        "store_name",
        "customer_id",
        "salesperson_id",
    ]
    drop_cols = [col for col in drop_cols if col in df_model.columns]

    X = df_model.drop(columns=drop_cols)
    y = df_model[target]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=2 / 3,
        random_state=42,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(X_train):
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X_train.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    numeric_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_preprocessor, numeric_cols),
            ("cat", categorical_preprocessor, categorical_cols),
        ]
    )


def evaluate_model(pipeline, X, y):
    y_pred = pipeline.predict(X)

    return {
        "mae": float(mean_absolute_error(y, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "r2": float(r2_score(y, y_pred)),
    }


def log_metrics_with_prefix(prefix, metrics_dict):
    for metric_name, metric_value in metrics_dict.items():
        mlflow.log_metric(f"{prefix}_{metric_name}", metric_value)


def log_dataset_shape(X_train, X_val, X_test):
    mlflow.log_metric("train_rows", X_train.shape[0])
    mlflow.log_metric("validation_rows", X_val.shape[0])
    mlflow.log_metric("test_rows", X_test.shape[0])
    mlflow.log_metric("feature_count", X_train.shape[1])





def get_raw_feature_groups(X_train):
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X_train.select_dtypes(
        include=["int64", "float64", "int32", "float32", "bool"]
    ).columns.tolist()
    return numeric_cols, categorical_cols


def map_transformed_feature_to_raw(transformed_feature_name, categorical_cols):
    """Map ColumnTransformer/OneHotEncoder output names back to original raw columns."""
    if transformed_feature_name.startswith("num__"):
        return transformed_feature_name.replace("num__", "", 1)

    if transformed_feature_name.startswith("cat__"):
        stripped = transformed_feature_name.replace("cat__", "", 1)
        matches = [
            col for col in categorical_cols
            if stripped == col or stripped.startswith(f"{col}_")
        ]
        if matches:
            return max(matches, key=len)
        return stripped

    return transformed_feature_name


def compute_grouped_shap_importance(best_pipeline, X_train, sample_size=1000):
    """
    Compute mean absolute SHAP values on transformed features, then aggregate them
    back to original raw input columns. This keeps the reduced model deployable because
    the frontend can ask for raw business features instead of one-hot encoded columns.
    """
    numeric_cols, categorical_cols = get_raw_feature_groups(X_train)

    if len(X_train) > sample_size:
        X_sample = X_train.sample(n=sample_size, random_state=42)
    else:
        X_sample = X_train.copy()

    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
    fitted_model = best_pipeline.named_steps["model"]

    X_transformed = fitted_preprocessor.transform(X_sample)
    if sparse.issparse(X_transformed):
        X_transformed = X_transformed.toarray()

    transformed_feature_names = fitted_preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(fitted_model)
    shap_values = explainer.shap_values(X_transformed)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    transformed_importance = pd.DataFrame({
        "transformed_feature": transformed_feature_names,
        "raw_feature": [
            map_transformed_feature_to_raw(name, categorical_cols)
            for name in transformed_feature_names
        ],
        "mean_abs_shap": mean_abs_shap,
    })

    grouped_importance = (
        transformed_importance
        .groupby("raw_feature", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    total_importance = grouped_importance["mean_abs_shap"].sum()
    if total_importance > 0:
        grouped_importance["importance_share"] = grouped_importance["mean_abs_shap"] / total_importance
        grouped_importance["cumulative_importance_share"] = grouped_importance["importance_share"].cumsum()
    else:
        grouped_importance["importance_share"] = 0.0
        grouped_importance["cumulative_importance_share"] = 0.0

    return grouped_importance, transformed_importance


def select_features_by_cumulative_importance(grouped_importance, coverage=0.95):
    """Select features in descending SHAP importance until cumulative importance >= coverage."""
    if grouped_importance.empty:
        raise ValueError("SHAP importance table is empty; cannot select reduced features.")

    if grouped_importance["mean_abs_shap"].sum() <= 0:
        return grouped_importance["raw_feature"].head(10).tolist()

    crossing_rows = grouped_importance.index[
        grouped_importance["cumulative_importance_share"] >= coverage
    ].tolist()

    if crossing_rows:
        last_idx = crossing_rows[0]
        selected = grouped_importance.loc[:last_idx, "raw_feature"].tolist()
    else:
        selected = grouped_importance["raw_feature"].tolist()

    return selected


def write_reference_data(X_train_reduced, model_key, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_specific_path = output_path / f"reference_input_data_{model_key}.csv"
    X_train_reduced.to_csv(model_specific_path, index=False)

    # Also write a generic file for backward compatibility with your existing Streamlit app.
    generic_path = output_path / "reference_input_data.csv"
    X_train_reduced.to_csv(generic_path, index=False)

    return str(model_specific_path), str(generic_path)


def log_artifact_dataframe(df, artifact_file_name):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / artifact_file_name
        df.to_csv(path, index=False)
        mlflow.log_artifact(str(path))


def log_artifact_json(obj, artifact_file_name):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / artifact_file_name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        mlflow.log_artifact(str(path))

def main():
    args = parse_args()

    df = load_data(args)
    df_model = engineer_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df_model)

    # 1) Train the full-feature model first.
    preprocessor = build_preprocessor(X_train)

    base_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
        ]
    )

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [8, 12, 16],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
    }

    full_grid_search = GridSearchCV(
        base_pipeline,
        param_grid,
        scoring="neg_root_mean_squared_error",
        cv=3,
        n_jobs=-1,
    )

    full_grid_search.fit(X_train, y_train)
    full_best_pipeline = full_grid_search.best_estimator_

    full_val_metrics = evaluate_model(full_best_pipeline, X_val, y_val)
    full_test_metrics = evaluate_model(full_best_pipeline, X_test, y_test)

    # 2) Use SHAP on the full-feature model to rank raw input features.
    grouped_importance, transformed_importance = compute_grouped_shap_importance(
        full_best_pipeline,
        X_train,
        sample_size=args.shap_sample_size,
    )

    selected_features = select_features_by_cumulative_importance(
        grouped_importance,
        coverage=args.shap_cumulative_coverage,
    )

    # 3) Retrain the same model family using only selected raw features.
    X_train_reduced = X_train[selected_features].copy()
    X_val_reduced = X_val[selected_features].copy()
    X_test_reduced = X_test[selected_features].copy()

    reduced_preprocessor = build_preprocessor(X_train_reduced)

    base_pipeline = Pipeline(
        steps=[
            ("reduced_preprocessor", reduced_preprocessor),
            ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
        ]
    )

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [8, 12, 16],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
    }

    reduced_grid_search = GridSearchCV(
        base_pipeline,
        param_grid,
        scoring="neg_root_mean_squared_error",
        cv=3,
        n_jobs=-1,
    )

    reduced_grid_search.fit(X_train_reduced, y_train)
    reduced_best_pipeline = reduced_grid_search.best_estimator_

    val_metrics = evaluate_model(reduced_best_pipeline, X_val_reduced, y_val)
    test_metrics = evaluate_model(reduced_best_pipeline, X_test_reduced, y_test)

    reference_model_path, reference_generic_path = write_reference_data(
        X_train_reduced,
        model_key="random_forest",
        output_dir=args.reference_output_dir,
    )

    mlflow.set_tracking_uri(uri=args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name="random_forest_shap_reduced_95"):
        mlflow.log_param("cv", 3)
        mlflow.log_param("scoring", "neg_root_mean_squared_error")
        mlflow.log_param("feature_selection_method", "mean_absolute_shap_cumulative_importance")
        mlflow.log_param("shap_cumulative_coverage", args.shap_cumulative_coverage)
        mlflow.log_param("shap_sample_size", min(args.shap_sample_size, len(X_train)))
        mlflow.log_param("full_feature_count", X_train.shape[1])
        mlflow.log_param("reduced_feature_count", X_train_reduced.shape[1])
        mlflow.log_param("selected_features", ",".join(selected_features))

        mlflow.log_param("param_grid_model__n_estimators", param_grid["model__n_estimators"])
        mlflow.log_param("param_grid_model__max_depth", param_grid["model__max_depth"])
        mlflow.log_param("param_grid_model__min_samples_split", param_grid["model__min_samples_split"])
        mlflow.log_param("param_grid_model__min_samples_leaf", param_grid["model__min_samples_leaf"])
        mlflow.log_params(reduced_grid_search.best_params_)

        mlflow.log_metric("full_best_cv_rmse", float(-full_grid_search.best_score_))
        mlflow.log_metric("best_cv_rmse", float(-reduced_grid_search.best_score_))

        log_metrics_with_prefix("full_validation", full_val_metrics)
        log_metrics_with_prefix("full_test", full_test_metrics)

        log_dataset_shape(X_train_reduced, X_val_reduced, X_test_reduced)
        log_metrics_with_prefix("validation", val_metrics)
        log_metrics_with_prefix("test", test_metrics)

        mlflow.set_tag(
            "Training Info",
            "SHAP-reduced Random Forest for next-period net sales forecasting",
        )
        mlflow.set_tag("model_key", "random_forest")
        mlflow.set_tag("model_family", "tree_ensemble")
        mlflow.set_tag("target", "next_period_net_sales")
        mlflow.set_tag("split_strategy", "70_train_10_validation_20_test")
        mlflow.set_tag("feature_selection", "shap_cumulative_95_raw_features")
        mlflow.set_tag("artifact_path", "random_forest_model")

        log_artifact_dataframe(grouped_importance, "shap_grouped_raw_feature_importance.csv")
        log_artifact_dataframe(transformed_importance, "shap_transformed_feature_importance.csv")
        log_artifact_json(selected_features, "selected_features.json")
        log_artifact_json(
            {
                "model_key": "random_forest",
                "selected_features": selected_features,
                "reference_model_path": reference_model_path,
                "reference_generic_path": reference_generic_path,
            },
            "model_input_schema.json",
        )

        input_example = X_train_reduced.head(5)
        signature = infer_signature(input_example, reduced_best_pipeline.predict(input_example))

        model_info = mlflow.sklearn.log_model(
            sk_model=reduced_best_pipeline,
            artifact_path="random_forest_model",
            signature=signature,
            input_example=input_example,
            registered_model_name=args.registered_model_name,
        )

        print("Full-feature best params:", full_grid_search.best_params_)
        print("Reduced-feature best params:", reduced_grid_search.best_params_)
        print("Selected features:", selected_features)
        print("Full validation metrics:", full_val_metrics)
        print("Full test metrics:", full_test_metrics)
        print("Reduced validation metrics:", val_metrics)
        print("Reduced test metrics:", test_metrics)
        print("Reference data written to:", reference_model_path)
        print("Model URI:", model_info.model_uri)


if __name__ == "__main__":
    main()
