import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import ks_2samp


# -----------------------------
# Config
# -----------------------------
DEFAULT_API_URL = "http://localhost:8000"
LOG_FILE = "prediction_log.csv"

DRIFT_P_VALUE_THRESHOLD = 0.05
CATEGORICAL_FREQ_SHIFT_THRESHOLD = 0.25

MODEL_ORDER = ["gradient_boosting", "random_forest"]

MODEL_DISPLAY_NAMES = {
    "gradient_boosting": "Gradient Boosting",
    "random_forest": "Random Forest",
}

REFERENCE_FILE_CANDIDATES = {
    "gradient_boosting": [
        "reference_input_data_gradient_boosting.csv",
        "reference_input_data_gb.csv",
        "reference_input_data.csv",
    ],
    "random_forest": [
        "reference_input_data_random_forest.csv",
        "reference_input_data_rf.csv",
        "reference_input_data.csv",
    ],
}

EXPLORER_REFERENCE_FILES = [
    "product_store_reference.csv",
    "data_explorer_reference.csv",
    "reference_input_data_gradient_boosting.csv",
    "reference_input_data_random_forest.csv",
    "reference_input_data.csv",
]

FEATURE_IMPORTANCE_FILE_CANDIDATES = {
    "gradient_boosting": [
        "shap_grouped_raw_feature_importance_gradient_boosting.csv",
        "shap_raw_feature_importance_gradient_boosting.csv",
        "feature_importance_gradient_boosting.csv",
        "gradient_boosting_feature_importance.csv",
        "shap_grouped_raw_feature_importance.csv",
    ],
    "random_forest": [
        "shap_grouped_raw_feature_importance_random_forest.csv",
        "shap_raw_feature_importance_random_forest.csv",
        "feature_importance_random_forest.csv",
        "random_forest_feature_importance.csv",
        "shap_grouped_raw_feature_importance.csv",
    ],
}

DERIVED_FEATURES = {
    "quarter_num",
    "is_q4",
    "is_year_end_season",
    "is_mid_year",
    "month_sin",
    "month_cos",
    "discount_percentage",
    "product_profitability_ratio",
    "sales_growth_previous_period",
}

DERIVED_DEPENDENCIES = {
    "quarter_num": ["month_num"],
    "is_q4": ["month_num"],
    "is_year_end_season": ["month_num"],
    "is_mid_year": ["month_num"],
    "month_sin": ["month_num"],
    "month_cos": ["month_num"],
    "discount_percentage": ["current_month_discount_amount", "current_month_gross_sales"],
    "product_profitability_ratio": ["current_month_margin_amount", "current_month_net_sales"],
    "sales_growth_previous_period": ["current_month_net_sales", "previous_month_sales"],
}

# These are always useful in a business-facing demand forecasting demo.
# They are shown even if a model does not explicitly require all of them.
SCENARIO_ADJUSTMENT_FEATURES = [
    "year_num",
    "month_num",
    "current_month_order_qty",
    "current_month_gross_sales",
    "current_month_discount_amount",
    "current_month_net_sales",
    "current_month_margin_amount",
    "previous_month_sales",
    "rolling_avg_sales_3m",
    "previous_month_margin",
]

DEFAULT_CATEGORICAL_OPTIONS = {
    "product_category_name": ["Clothing", "Bikes", "Accessories", "Components"],
    "product_subcategory_name": [
        "Socks", "Mountain Bikes", "Road Bikes", "Touring Bikes",
        "Helmets", "Jerseys", "Tires and Tubes", "Handlebars"
    ],
    "product_model_name": [
        "Mountain Socks", "Mountain-100", "Road-150", "Touring-1000",
        "Sport-100", "Classic Vest", "HL Mountain Tire"
    ],
    "color": ["Red", "Black", "Blue", "Silver", "Yellow", "Multi", "NA"],
    "size": ["", "S", "M", "L", "XL", "42", "44", "48"],
    "weight": ["", "0.0", "50.0", "100.0", "500.0", "1000.0"],
}

DEFAULT_VALUES = {
    "product_key": 100,
    "product_category_name": "Clothing",
    "product_subcategory_name": "Socks",
    "product_model_name": "Mountain Socks",
    "size": "",
    "weight": "",
    "color": "Red",
    "standard_cost": 2.50,
    "list_price": 5.00,
    "customer_key": 200,
    "customer_id": 200,
    "customer_name": "Customer 200",
    "store_id": 300,
    "store_name": "Store 300",
    "territory_id": 1,
    "salesperson_key": 400,
    "sales_quota": 5000.0,
    "bonus": 100.0,
    "commission_pct": 0.05,
    "sales_ytd": 10000.0,
    "sales_last_year": 15000.0,
    "year_num": 2008,
    "month_num": 7,
    "current_month_order_qty": 50,
    "current_month_gross_sales": 250.0,
    "current_month_discount_amount": 10.0,
    "current_month_net_sales": 240.0,
    "current_month_margin_amount": 115.0,
    "avg_unit_price": 5.0,
    "avg_unit_price_discount": 0.2,
    "number_of_orders": 5,
    "number_of_order_lines": 5,
    "previous_month_sales": 200.0,
    "rolling_avg_sales_3m": 210.0,
    "previous_month_margin": 90.0,
}

FEATURE_LABELS = {
    "product_key": "Product",
    "product_category_name": "Product Category",
    "product_subcategory_name": "Product Subcategory",
    "product_model_name": "Product Model",
    "size": "Product Size",
    "weight": "Product Weight",
    "color": "Product Color",
    "standard_cost": "Standard Cost",
    "list_price": "List Price",
    "customer_key": "Customer",
    "customer_id": "Customer",
    "customer_name": "Customer",
    "store_id": "Store",
    "store_name": "Store",
    "territory_id": "Sales Territory",
    "salesperson_key": "Salesperson",
    "sales_quota": "Sales Quota",
    "bonus": "Salesperson Bonus",
    "commission_pct": "Commission Percentage",
    "sales_ytd": "Sales Year-to-Date",
    "sales_last_year": "Sales Last Year",
    "year_num": "Year",
    "month_num": "Month",
    "current_month_order_qty": "Current Month Order Quantity",
    "current_month_gross_sales": "Current Month Gross Sales",
    "current_month_discount_amount": "Current Month Discount Amount",
    "current_month_net_sales": "Current Month Net Sales",
    "current_month_margin_amount": "Current Month Margin Amount",
    "avg_unit_price": "Average Unit Price",
    "avg_unit_price_discount": "Average Unit Price Discount",
    "number_of_orders": "Number of Orders",
    "number_of_order_lines": "Number of Order Lines",
    "previous_month_sales": "Previous Month Sales",
    "rolling_avg_sales_3m": "Rolling Average Sales, Last 3 Months",
    "previous_month_margin": "Previous Month Margin",
}

FEATURE_HELP = {
    "previous_month_sales": "Historical demand signal. Higher values usually indicate stronger expected demand.",
    "rolling_avg_sales_3m": "Smoothed historical sales over the last 3 months.",
    "current_month_net_sales": "Current month sales after discounts.",
    "current_month_gross_sales": "Current month sales before discounts.",
    "current_month_discount_amount": "Total discount given in the current month.",
    "month_num": "Used to automatically derive quarter and seasonality features.",
    "store_id": "Shown as a store name in the UI, but sent as store_id in the API payload.",
    "customer_key": "Shown as a customer/store name in the UI, but sent as customer_key in the API payload.",
}

ID_FEATURES = {
    "product_key",
    "customer_key",
    "customer_id",
    "store_id",
    "territory_id",
    "salesperson_key",
    "year_num",
    "month_num",
    "current_month_order_qty",
    "number_of_orders",
    "number_of_order_lines",
}


# -----------------------------
# Generic helpers
# -----------------------------
def clean_missing(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    return value


def empty_to_none(value: Any) -> Any:
    value = clean_missing(value)
    if value is None:
        return None
    text = str(value).strip()
    return None if text == "" else value


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def as_records(df: pd.DataFrame, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    if max_rows is not None:
        df = df.head(max_rows)
    return df.replace({np.nan: None}).to_dict(orient="records")


# -----------------------------
# API helpers
# -----------------------------
def api_get(api_base_url: str, endpoint: str) -> Dict[str, Any]:
    response = requests.get(f"{api_base_url.rstrip('/')}{endpoint}", timeout=30)
    response.raise_for_status()
    return response.json()


def api_post(api_base_url: str, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(f"{api_base_url.rstrip('/')}{endpoint}", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False)
def load_model_metadata(api_base_url: str) -> Dict[str, Any]:
    return api_get(api_base_url, "/models")


def get_loaded_model_keys(metadata: Dict[str, Any]) -> List[str]:
    model_keys = metadata.get("ab_order") or list(metadata.get("models", {}).keys())
    ordered = [key for key in MODEL_ORDER if key in model_keys]
    extras = [key for key in model_keys if key not in ordered]
    return ordered + extras


def get_model_display_name(model_key: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    if metadata and model_key in metadata.get("models", {}):
        return metadata["models"][model_key].get("display_name", MODEL_DISPLAY_NAMES.get(model_key, model_key))
    return MODEL_DISPLAY_NAMES.get(model_key, model_key)


def extract_prediction(response_json: Dict[str, Any]) -> float:
    preds = response_json.get("predictions")
    if isinstance(preds, list) and preds:
        return float(preds[0])
    if isinstance(preds, (int, float)):
        return float(preds)
    raise ValueError(f"Could not parse prediction from response: {response_json}")


# -----------------------------
# A/B testing helpers
# -----------------------------
def load_prediction_log() -> pd.DataFrame:
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()

    df = pd.read_csv(LOG_FILE)

    # Compatibility with older logs created before A/B testing was added.
    if "model_key" not in df.columns:
        df["model_key"] = "legacy_unknown"
    if "model_display_name" not in df.columns:
        df["model_display_name"] = df["model_key"].map(MODEL_DISPLAY_NAMES).fillna("Legacy / Unknown")
    if "predicted_next_period_net_sales" not in df.columns:
        df["predicted_next_period_net_sales"] = np.nan

    return df


def get_successful_ab_prediction_count(log_df: pd.DataFrame, loaded_model_keys: List[str]) -> int:
    if log_df.empty or "model_key" not in log_df.columns:
        return 0

    valid_keys = [key for key in MODEL_ORDER if key in loaded_model_keys]
    if not valid_keys:
        valid_keys = loaded_model_keys

    return int(log_df[log_df["model_key"].isin(valid_keys)].shape[0])


def get_next_ab_model_key(loaded_model_keys: List[str], log_df: pd.DataFrame) -> str:
    ordered_loaded = [key for key in MODEL_ORDER if key in loaded_model_keys]
    if not ordered_loaded:
        ordered_loaded = loaded_model_keys

    if not ordered_loaded:
        raise ValueError("No loaded models available for A/B testing.")

    prediction_count = get_successful_ab_prediction_count(log_df, ordered_loaded)
    return ordered_loaded[prediction_count % len(ordered_loaded)]


def save_prediction_log(input_record: Dict[str, Any], response_json: Dict[str, Any], api_base_url: str) -> None:
    prediction = extract_prediction(response_json)

    log_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "api_url": api_base_url,
        "model_key": response_json.get("model_key"),
        "model_display_name": response_json.get("model_display_name"),
        "model_run_id": response_json.get("model_run_id"),
        "model_metric_name": response_json.get("model_metric_name"),
        "model_metric_value": response_json.get("model_metric_value"),
        "predicted_next_period_net_sales": prediction,
    }

    log_record.update(input_record)
    new_row = pd.DataFrame([log_record])

    if os.path.exists(LOG_FILE):
        old_log = pd.read_csv(LOG_FILE)
        updated_log = pd.concat([old_log, new_row], ignore_index=True)
    else:
        updated_log = new_row

    updated_log.to_csv(LOG_FILE, index=False)


# -----------------------------
# Reference data and lookup helpers
# -----------------------------
def read_csv_if_exists(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def get_reference_file_for_model(model_key: str) -> Optional[str]:
    for candidate in REFERENCE_FILE_CANDIDATES.get(model_key, ["reference_input_data.csv"]):
        if os.path.exists(candidate):
            return candidate
    return None


@st.cache_data(show_spinner=False)
def load_reference_data_for_model(model_key: str) -> pd.DataFrame:
    reference_file = get_reference_file_for_model(model_key)
    if reference_file is None:
        return pd.DataFrame()

    df = pd.read_csv(reference_file)
    df["source_model_key"] = model_key
    return df


@st.cache_data(show_spinner=False)
def load_explorer_reference_data() -> pd.DataFrame:
    frames = []

    for file_name in EXPLORER_REFERENCE_FILES:
        df = read_csv_if_exists(file_name)
        if df is not None and not df.empty:
            df = df.copy()
            df["source_file"] = file_name
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.drop_duplicates()
    return combined


def create_display_column(
    df: pd.DataFrame,
    id_col: str,
    name_candidates: List[str],
    fallback_prefix: str,
) -> Tuple[pd.DataFrame, str]:
    df = df.copy()
    display_col = f"{id_col}_display"

    name_col = None
    for candidate in name_candidates:
        if candidate in df.columns:
            name_col = candidate
            break

    if id_col not in df.columns:
        df[display_col] = fallback_prefix
        return df, display_col

    if name_col:
        df[display_col] = df.apply(
            lambda row: (
                f"{row[name_col]} ({fallback_prefix} {row[id_col]})"
                if clean_missing(row.get(name_col)) is not None
                else f"{fallback_prefix} {row[id_col]}"
            ),
            axis=1,
        )
    else:
        df[display_col] = df[id_col].apply(lambda value: f"{fallback_prefix} {value}")

    return df, display_col


def enrich_reference_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    if "product_key" in df.columns:
        product_name_cols = ["product_name", "product_model_name", "product_subcategory_name", "product_category_name"]
        existing = [col for col in product_name_cols if col in df.columns]
        if existing:
            def product_display(row):
                parts = [str(row[col]) for col in existing if clean_missing(row.get(col)) is not None]
                label = " / ".join(parts[:3]) if parts else f"Product {row['product_key']}"
                return f"{label} (Product {row['product_key']})"
            df["product_key_display"] = df.apply(product_display, axis=1)
        else:
            df["product_key_display"] = df["product_key"].apply(lambda value: f"Product {value}")

    df, _ = create_display_column(df, "store_id", ["store_name", "store", "customer_store_name"], "Store")
    df, _ = create_display_column(df, "customer_key", ["customer_name", "store_name", "customer", "customer_id"], "Customer")
    df, _ = create_display_column(df, "salesperson_key", ["salesperson_name", "salesperson_id"], "Salesperson")
    df, _ = create_display_column(df, "territory_id", ["territory_name", "sales_territory_name"], "Territory")

    return df


def build_lookup_options(reference_df: pd.DataFrame, feature: str) -> Dict[str, Any]:
    if reference_df.empty or feature not in reference_df.columns:
        return {}

    display_col = f"{feature}_display"
    if display_col not in reference_df.columns:
        display_col = feature

    cols = [feature, display_col]
    lookup = reference_df[cols].dropna(subset=[feature]).drop_duplicates().copy()

    if lookup.empty:
        return {}

    lookup[display_col] = lookup[display_col].astype(str)
    lookup = lookup.sort_values(display_col)

    mapping = {}
    for _, row in lookup.iterrows():
        display_value = row[display_col]
        raw_id = row[feature]
        mapping[display_value] = raw_id

    return mapping


def get_feature_options_from_reference(reference_df: pd.DataFrame, feature: str) -> List[Any]:
    if not reference_df.empty and feature in reference_df.columns:
        values = reference_df[feature].dropna().drop_duplicates().astype(str).sort_values().tolist()
        if values:
            if feature in ["size", "weight"] and "" not in values:
                values = [""] + values
            return values

    return DEFAULT_CATEGORICAL_OPTIONS.get(feature, [])


# -----------------------------
# Feature importance helpers
# -----------------------------
def normalise_feature_importance_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["feature", "importance", "importance_pct", "cumulative_importance_pct"])

    df = df.copy()

    feature_col_candidates = ["feature", "raw_feature", "input_feature", "feature_name", "column"]
    importance_col_candidates = [
        "importance",
        "mean_abs_shap",
        "mean_abs_shap_value",
        "abs_importance",
        "shap_importance",
        "value",
    ]
    pct_col_candidates = ["importance_pct", "importance_percent", "pct_importance", "percentage"]

    feature_col = next((col for col in feature_col_candidates if col in df.columns), None)
    importance_col = next((col for col in importance_col_candidates if col in df.columns), None)
    pct_col = next((col for col in pct_col_candidates if col in df.columns), None)

    if feature_col is None:
        return pd.DataFrame(columns=["feature", "importance", "importance_pct", "cumulative_importance_pct"])

    out = pd.DataFrame()
    out["feature"] = df[feature_col].astype(str)

    if importance_col is not None:
        out["importance"] = pd.to_numeric(df[importance_col], errors="coerce").fillna(0.0)
    elif pct_col is not None:
        out["importance_pct"] = pd.to_numeric(df[pct_col], errors="coerce").fillna(0.0)
        if out["importance_pct"].max() > 1:
            out["importance_pct"] = out["importance_pct"] / 100.0
        out["importance"] = out["importance_pct"]
    else:
        out["importance"] = 0.0

    if pct_col is not None:
        out["importance_pct"] = pd.to_numeric(df[pct_col], errors="coerce").fillna(0.0)
        if out["importance_pct"].max() > 1:
            out["importance_pct"] = out["importance_pct"] / 100.0
    else:
        total = out["importance"].sum()
        out["importance_pct"] = out["importance"] / total if total != 0 else 0.0

    out = out.groupby("feature", as_index=False)[["importance", "importance_pct"]].sum()
    out = out.sort_values("importance", ascending=False)
    out["cumulative_importance_pct"] = out["importance_pct"].cumsum()

    return out


def feature_importance_from_metadata(model_info: Dict[str, Any]) -> pd.DataFrame:
    raw = (
        model_info.get("feature_importance")
        or model_info.get("feature_importances")
        or model_info.get("shap_feature_importance")
        or model_info.get("feature_importance_table")
    )

    if raw is None:
        return pd.DataFrame()

    if isinstance(raw, dict):
        rows = []
        for feature, value in raw.items():
            if isinstance(value, dict):
                rows.append({"feature": feature, **value})
            else:
                rows.append({"feature": feature, "importance": value})
        return normalise_feature_importance_df(pd.DataFrame(rows))

    if isinstance(raw, list):
        return normalise_feature_importance_df(pd.DataFrame(raw))

    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_feature_importance(model_key: str, model_info: Dict[str, Any]) -> pd.DataFrame:
    metadata_df = feature_importance_from_metadata(model_info)
    if not metadata_df.empty:
        return metadata_df

    for file_name in FEATURE_IMPORTANCE_FILE_CANDIDATES.get(model_key, []):
        df = read_csv_if_exists(file_name)
        if df is not None and not df.empty:
            return normalise_feature_importance_df(df)

    return pd.DataFrame(columns=["feature", "importance", "importance_pct", "cumulative_importance_pct"])


def build_importance_lookup(importance_df: pd.DataFrame) -> Dict[str, float]:
    if importance_df.empty or "feature" not in importance_df.columns:
        return {}
    return dict(zip(importance_df["feature"].astype(str), importance_df["importance_pct"].astype(float)))


def get_visible_feature_importance_pct(
    visible_feature: str,
    required_features: List[str],
    importance_lookup: Dict[str, float],
) -> float:
    pct = importance_lookup.get(visible_feature, 0.0)

    # If a user-facing input is the dependency for hidden derived features,
    # display the combined importance controlled by that input.
    for derived_feature in required_features:
        if derived_feature in DERIVED_DEPENDENCIES and visible_feature in DERIVED_DEPENDENCIES[derived_feature]:
            pct += importance_lookup.get(derived_feature, 0.0)

    return float(pct)


def create_visible_feature_table(
    required_features: List[str],
    importance_df: pd.DataFrame,
) -> pd.DataFrame:
    importance_lookup = build_importance_lookup(importance_df)

    visible_features = []
    for feature in required_features:
        if feature in DERIVED_FEATURES:
            for dep in DERIVED_DEPENDENCIES.get(feature, []):
                if dep not in visible_features:
                    visible_features.append(dep)
        else:
            if feature not in visible_features:
                visible_features.append(feature)

    # Always allow the user to adjust key business drivers.
    for feature in SCENARIO_ADJUSTMENT_FEATURES:
        if feature not in visible_features:
            visible_features.append(feature)

    rows = []
    for feature in visible_features:
        rows.append({
            "feature": feature,
            "label": FEATURE_LABELS.get(feature, feature),
            "importance_pct": get_visible_feature_importance_pct(feature, required_features, importance_lookup),
            "is_model_required_or_dependency": feature in required_features or any(
                feature in DERIVED_DEPENDENCIES.get(req_feature, []) for req_feature in required_features
            ),
        })

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    table = table.sort_values(
        by=["is_model_required_or_dependency", "importance_pct", "feature"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return table


# -----------------------------
# Feature engineering in frontend
# API also derives these as a safety net.
# -----------------------------
def derive_features(record: Dict[str, Any]) -> Dict[str, Any]:
    record = dict(record)

    month_num = record.get("month_num")
    if month_num is not None:
        month_num = safe_int(month_num, 1)
        quarter_num = int(np.ceil(month_num / 3))

        record["quarter_num"] = quarter_num
        record["is_q4"] = 1 if quarter_num == 4 else 0
        record["is_year_end_season"] = 1 if month_num in [11, 12] else 0
        record["is_mid_year"] = 1 if month_num in [6, 7] else 0
        record["month_sin"] = float(np.sin(2 * np.pi * month_num / 12))
        record["month_cos"] = float(np.cos(2 * np.pi * month_num / 12))

    gross = record.get("current_month_gross_sales")
    discount = record.get("current_month_discount_amount")
    if gross is not None and discount is not None:
        gross = safe_float(gross)
        discount = safe_float(discount)
        record["discount_percentage"] = float(discount / gross) if gross != 0 else 0.0

    net_sales = record.get("current_month_net_sales")
    margin = record.get("current_month_margin_amount")
    if net_sales is not None and margin is not None:
        net_sales = safe_float(net_sales)
        margin = safe_float(margin)
        record["product_profitability_ratio"] = float(margin / net_sales) if net_sales != 0 else 0.0

    previous_sales = record.get("previous_month_sales")
    if net_sales is not None and previous_sales is not None:
        net_sales = safe_float(net_sales)
        previous_sales = safe_float(previous_sales)
        record["sales_growth_previous_period"] = (
            float((net_sales - previous_sales) / previous_sales)
            if previous_sales != 0
            else 0.0
        )

    return record


def get_feature_group(feature: str) -> str:
    if feature.startswith("product_") or feature in ["color", "size", "weight", "standard_cost", "list_price"]:
        return "Product inputs"
    if feature.startswith("customer_") or feature in ["store_id", "territory_id", "customer_id", "customer_name", "store_name"]:
        return "Customer and store inputs"
    if feature.startswith("salesperson") or feature in ["sales_quota", "bonus", "commission_pct", "sales_ytd", "sales_last_year"]:
        return "Salesperson inputs"
    if feature in ["year_num", "month_num"]:
        return "Time inputs"
    return "Sales and demand inputs"


def get_prefill_value(feature: str, prefill_record: Dict[str, Any]) -> Any:
    if feature in prefill_record and clean_missing(prefill_record.get(feature)) is not None:
        return prefill_record[feature]
    return DEFAULT_VALUES.get(feature, 0.0)


def render_lookup_selectbox(
    feature: str,
    label: str,
    mapping: Dict[str, Any],
    default_raw_value: Any,
    help_text: Optional[str],
    widget_key: str,
) -> Any:
    options = list(mapping.keys())
    if not options:
        return None

    selected_index = 0
    for idx, display_value in enumerate(options):
        if str(mapping[display_value]) == str(default_raw_value):
            selected_index = idx
            break

    selected_display = st.selectbox(
        label,
        options,
        index=selected_index,
        help=help_text,
        key=widget_key,
    )
    return mapping[selected_display]


def render_feature_input(
    feature: str,
    reference_df: pd.DataFrame,
    prefill_record: Dict[str, Any],
    feature_importance_pct: float,
) -> Any:
    label = FEATURE_LABELS.get(feature, feature)
    label_with_importance = f"{label}  ·  {feature_importance_pct * 100:.2f}% importance"
    help_text = FEATURE_HELP.get(feature, None)
    default = get_prefill_value(feature, prefill_record)

    # Key fix: Streamlit preserves widget values for fixed keys.
    # prefill_version changes whenever a historical combination is selected,
    # forcing the prediction widgets to recreate with the new defaults.
    widget_key = f"input_{feature}_v{st.session_state.get('prefill_version', 0)}"

    lookup_features = {"product_key", "customer_key", "customer_id", "store_id", "territory_id", "salesperson_key"}
    if feature in lookup_features and not reference_df.empty:
        mapping = build_lookup_options(reference_df, feature)
        if mapping:
            return render_lookup_selectbox(
                feature=feature,
                label=label_with_importance,
                mapping=mapping,
                default_raw_value=default,
                help_text=help_text,
                widget_key=widget_key,
            )

    if feature in DEFAULT_CATEGORICAL_OPTIONS or (
        not reference_df.empty and feature in reference_df.columns and not pd.api.types.is_numeric_dtype(reference_df[feature])
    ):
        options = get_feature_options_from_reference(reference_df, feature)
        default_value = str(default) if default is not None else ""
        if default_value not in options:
            options = [default_value] + options if default_value != "" else [""] + options
        # Preserve order while removing duplicates.
        options = list(dict.fromkeys(options))
        index = options.index(default_value) if default_value in options else 0
        return st.selectbox(
            label_with_importance,
            options,
            index=index,
            help=help_text,
            key=widget_key,
        )

    if feature in ID_FEATURES:
        min_value = 1 if feature == "month_num" else 0
        max_value = 12 if feature == "month_num" else None
        return st.number_input(
            label_with_importance,
            value=safe_int(default, safe_int(DEFAULT_VALUES.get(feature, 0))),
            min_value=min_value,
            max_value=max_value,
            step=1,
            help=help_text,
            key=widget_key,
        )

    return st.number_input(
        label_with_importance,
        value=safe_float(default, safe_float(DEFAULT_VALUES.get(feature, 0.0))),
        step=10.0 if "sales" in feature or "amount" in feature or feature in ["sales_quota", "bonus"] else 0.01,
        format="%.4f",
        help=help_text,
        key=widget_key,
    )


def build_input_form(
    required_features: List[str],
    importance_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    prefill_record: Dict[str, Any],
) -> Dict[str, Any]:
    feature_table = create_visible_feature_table(required_features, importance_df)

    if feature_table.empty:
        st.warning("No required feature list found for the selected model.")
        return {}

    grouped_features: Dict[str, pd.DataFrame] = {}
    for group_name, group_df in feature_table.groupby(feature_table["feature"].apply(get_feature_group), sort=False):
        grouped_features[group_name] = group_df

    input_record: Dict[str, Any] = {}

    for group_name, group_df in grouped_features.items():
        st.subheader(group_name)
        cols = st.columns(3)
        for i, row in group_df.iterrows():
            feature = row["feature"]
            importance_pct = float(row.get("importance_pct", 0.0))
            with cols[i % 3]:
                value = render_feature_input(feature, reference_df, prefill_record, importance_pct)
                input_record[feature] = empty_to_none(value)

    input_record = derive_features(input_record)
    return input_record


# -----------------------------
# Data explorer helpers
# -----------------------------
def choose_metric_column(df: pd.DataFrame) -> Optional[str]:
    for col in [
        "current_month_net_sales",
        "rolling_avg_sales_3m",
        "previous_month_sales",
        "current_month_gross_sales",
        "predicted_next_period_net_sales",
    ]:
        if col in df.columns:
            return col
    return None


def build_product_store_summary(reference_df: pd.DataFrame) -> pd.DataFrame:
    if reference_df.empty:
        return pd.DataFrame()

    df = enrich_reference_display_columns(reference_df)
    metric_col = choose_metric_column(df)
    if metric_col is None:
        return pd.DataFrame()

    group_cols = []
    for col in [
        "product_key_display",
        "product_category_name",
        "product_subcategory_name",
        "product_model_name",
        "store_id_display",
        "customer_key_display",
        "territory_id_display",
    ]:
        if col in df.columns:
            group_cols.append(col)

    if not group_cols:
        return pd.DataFrame()

    agg_spec = {
        "avg_monthly_sales": (metric_col, "mean"),
        "historical_records": (metric_col, "count"),
    }

    for optional_col in [
        "current_month_order_qty",
        "current_month_net_sales",
        "previous_month_sales",
        "rolling_avg_sales_3m",
        "current_month_discount_amount",
        "current_month_gross_sales",
        "previous_month_margin",
    ]:
        if optional_col in df.columns:
            agg_spec[f"avg_{optional_col}"] = (optional_col, "mean")

    summary = df.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()
    summary = summary.sort_values("avg_monthly_sales", ascending=False).reset_index(drop=True)
    summary["combination_label"] = summary.apply(
        lambda row: " | ".join([
            str(row[col])
            for col in ["product_model_name", "product_key_display", "store_id_display", "customer_key_display"]
            if col in summary.columns and clean_missing(row.get(col)) is not None
        ]),
        axis=1,
    )

    return summary


def find_matching_reference_row(reference_df: pd.DataFrame, selected_summary_row: pd.Series) -> Dict[str, Any]:
    if reference_df.empty:
        return {}

    df = enrich_reference_display_columns(reference_df)
    mask = pd.Series(True, index=df.index)

    for col in [
        "product_key_display",
        "product_category_name",
        "product_subcategory_name",
        "product_model_name",
        "store_id_display",
        "customer_key_display",
        "territory_id_display",
    ]:
        if col in df.columns and col in selected_summary_row.index:
            mask = mask & (df[col].astype(str) == str(selected_summary_row[col]))

    matched = df[mask].copy()
    if matched.empty:
        return {}

    metric_col = choose_metric_column(matched)
    if metric_col:
        matched = matched.sort_values(metric_col, ascending=False)

    row = matched.iloc[0].to_dict()

    # Replace key business values with group averages where available,
    # so the prefill is representative rather than a random individual row.
    for avg_col in [col for col in selected_summary_row.index if col.startswith("avg_")]:
        original_col = avg_col.replace("avg_", "", 1)
        row[original_col] = selected_summary_row[avg_col]

    return {key: clean_missing(value) for key, value in row.items()}


def update_prediction_defaults_from_row(row: Dict[str, Any]) -> None:
    """
    Apply a selected historical product-store/customer row to the prediction form.

    The important part is incrementing prefill_version. All prediction widgets use
    prefill_version in their Streamlit keys, so the form recreates with the new
    historical values instead of keeping the previous widget state.
    """
    cleaned_prefill_record = {}

    for key, value in dict(row).items():
        cleaned_value = clean_missing(value)
        if cleaned_value is not None:
            cleaned_prefill_record[key] = cleaned_value

    st.session_state.prefill_record = {
        **DEFAULT_VALUES,
        **cleaned_prefill_record,
    }
    st.session_state.prefill_version = st.session_state.get("prefill_version", 0) + 1
    st.session_state.prefill_success_message = (
        "Historical product-store combination loaded into the prediction form. "
        "Open the Live Prediction tab to review and adjust the values."
    )


# -----------------------------
# Drift helpers
# -----------------------------
def calculate_numeric_drift(reference_df: pd.DataFrame, production_df: pd.DataFrame, numeric_features: List[str]) -> pd.DataFrame:
    rows = []

    for col in numeric_features:
        if col not in reference_df.columns or col not in production_df.columns:
            continue

        ref_values = pd.to_numeric(reference_df[col], errors="coerce").dropna()
        prod_values = pd.to_numeric(production_df[col], errors="coerce").dropna()

        if len(ref_values) < 2 or len(prod_values) < 2:
            rows.append({
                "feature": col,
                "type": "numeric",
                "test": "KS test",
                "reference_count": len(ref_values),
                "production_count": len(prod_values),
                "reference_mean": np.nan,
                "production_mean": np.nan,
                "p_value": np.nan,
                "frequency_shift": np.nan,
                "drift_status": "Not enough data",
            })
            continue

        _, p_value = ks_2samp(ref_values, prod_values)

        rows.append({
            "feature": col,
            "type": "numeric",
            "test": "KS test",
            "reference_count": len(ref_values),
            "production_count": len(prod_values),
            "reference_mean": round(ref_values.mean(), 4),
            "production_mean": round(prod_values.mean(), 4),
            "p_value": round(float(p_value), 6),
            "frequency_shift": np.nan,
            "drift_status": "Possible Drift" if p_value < DRIFT_P_VALUE_THRESHOLD else "No Drift",
        })

    return pd.DataFrame(rows)


def calculate_categorical_drift(reference_df: pd.DataFrame, production_df: pd.DataFrame, categorical_features: List[str]) -> pd.DataFrame:
    rows = []

    for col in categorical_features:
        if col not in reference_df.columns or col not in production_df.columns:
            continue

        ref_values = reference_df[col].fillna("MISSING").astype(str)
        prod_values = production_df[col].fillna("MISSING").astype(str)

        if len(ref_values) == 0 or len(prod_values) == 0:
            rows.append({
                "feature": col,
                "type": "categorical",
                "test": "Frequency shift",
                "reference_count": len(ref_values),
                "production_count": len(prod_values),
                "reference_mean": np.nan,
                "production_mean": np.nan,
                "p_value": np.nan,
                "frequency_shift": np.nan,
                "drift_status": "Not enough data",
            })
            continue

        ref_freq = ref_values.value_counts(normalize=True)
        prod_freq = prod_values.value_counts(normalize=True)

        all_categories = sorted(set(ref_freq.index).union(set(prod_freq.index)))

        max_shift = 0.0
        max_shift_category = None
        for category in all_categories:
            shift = abs(prod_freq.get(category, 0.0) - ref_freq.get(category, 0.0))
            if shift > max_shift:
                max_shift = float(shift)
                max_shift_category = category

        rows.append({
            "feature": col,
            "type": "categorical",
            "test": f"Max frequency shift: {max_shift_category}",
            "reference_count": len(ref_values),
            "production_count": len(prod_values),
            "reference_mean": np.nan,
            "production_mean": np.nan,
            "p_value": np.nan,
            "frequency_shift": round(max_shift, 4),
            "drift_status": "Possible Drift" if max_shift >= CATEGORICAL_FREQ_SHIFT_THRESHOLD else "No Drift",
        })

    return pd.DataFrame(rows)


def calculate_drift(reference_df: pd.DataFrame, production_df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    features = [feature for feature in features if feature in reference_df.columns or feature in production_df.columns]

    numeric_features = []
    categorical_features = []

    for col in features:
        if col not in reference_df.columns:
            continue
        if pd.api.types.is_numeric_dtype(reference_df[col]):
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    numeric_drift = calculate_numeric_drift(reference_df, production_df, numeric_features)
    categorical_drift = calculate_categorical_drift(reference_df, production_df, categorical_features)

    return pd.concat([numeric_drift, categorical_drift], ignore_index=True)


# -----------------------------
# Dashboard helpers
# -----------------------------
def model_summary_metrics(log_df: pd.DataFrame, loaded_model_keys: List[str], metadata: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for model_key in loaded_model_keys:
        model_df = log_df[log_df.get("model_key", pd.Series(dtype=str)) == model_key] if not log_df.empty else pd.DataFrame()
        rows.append({
            "model_key": model_key,
            "model": get_model_display_name(model_key, metadata),
            "requests": int(len(model_df)),
            "avg_predicted_sales": float(model_df["predicted_next_period_net_sales"].mean()) if not model_df.empty else np.nan,
            "latest_prediction": float(model_df["predicted_next_period_net_sales"].iloc[-1]) if not model_df.empty else np.nan,
            "mlflow_metric": metadata.get("models", {}).get(model_key, {}).get("metric_name", "test_rmse"),
            "mlflow_metric_value": metadata.get("models", {}).get(model_key, {}).get("metric_value", np.nan),
        })
    return pd.DataFrame(rows)


def render_model_comparison_cards(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        st.warning("No model comparison data available yet.")
        return

    cols = st.columns(len(summary_df))
    for i, (_, row) in enumerate(summary_df.iterrows()):
        with cols[i]:
            st.markdown(f"### {row['model']}")
            st.metric("Prediction Requests", int(row["requests"]))
            st.metric(
                "Average Predicted Sales",
                "NA" if pd.isna(row["avg_predicted_sales"]) else f"{row['avg_predicted_sales']:,.2f}",
            )
            st.metric(
                "Latest Prediction",
                "NA" if pd.isna(row["latest_prediction"]) else f"{row['latest_prediction']:,.2f}",
            )
            st.caption(f"MLflow {row['mlflow_metric']}: {row['mlflow_metric_value']}")


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="AdventureWorks Demand Forecasting",
    layout="wide",
)

st.title("AdventureWorks Demand Forecasting Platform")

st.write(
    "Use historical product-store patterns to create realistic inputs, run live A/B predictions "
    "between Gradient Boosting and Random Forest, and monitor input drift side by side."
)

with st.sidebar:
    st.header("Connection")
    api_base_url = st.text_input("FastAPI base URL", value=DEFAULT_API_URL)

    st.markdown("---")
    st.header("A/B Testing")
    st.caption("The next prediction model is chosen from successful requests in the log: GB → RF → GB → RF.")

    st.markdown("---")
    st.header("Monitoring Files")
    st.write(f"Prediction log: `{LOG_FILE}`")
    st.caption("Reference files and SHAP importance files should be placed in this webapp folder.")

try:
    metadata = load_model_metadata(api_base_url)
    loaded_model_keys = get_loaded_model_keys(metadata)
except Exception as e:
    metadata = {"models": {}}
    loaded_model_keys = []
    st.error(f"Could not connect to FastAPI model metadata endpoint: {e}")

explorer_reference_df = enrich_reference_display_columns(load_explorer_reference_data())
log_df = load_prediction_log()

if "prefill_record" not in st.session_state:
    st.session_state.prefill_record = dict(DEFAULT_VALUES)

if "prefill_version" not in st.session_state:
    st.session_state.prefill_version = 0

if "last_prediction_message" in st.session_state:
    st.success(st.session_state["last_prediction_message"])
    del st.session_state["last_prediction_message"]


tab_explorer, tab_predict, tab_dashboard, tab_drift = st.tabs(
    [
        "Data Explorer",
        "Live Prediction and A/B Test",
        "Prediction Dashboard",
        "Input Drift Monitoring",
    ]
)


with tab_explorer:
    st.header("Data Explorer: Historical Product-Store Demand")

    if explorer_reference_df.empty:
        st.warning(
            "No reference data found. Place `product_store_reference.csv`, `data_explorer_reference.csv`, "
            "or `reference_input_data_<model>.csv` in the webapp folder."
        )
    else:
        metric_col = choose_metric_column(explorer_reference_df)
        st.caption(
            f"Loaded {len(explorer_reference_df):,} historical/reference rows. "
            f"Demand summary metric used: `{metric_col}`."
        )

        summary_df = build_product_store_summary(explorer_reference_df)

        if summary_df.empty:
            st.warning("Could not create product-store summary because required product/store/sales columns were not found.")
            with st.expander("Available reference columns"):
                st.write(list(explorer_reference_df.columns))
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Reference Rows", f"{len(explorer_reference_df):,}")
            col2.metric("Product-Store Combinations", f"{len(summary_df):,}")
            col3.metric("Average Monthly Sales", f"{summary_df['avg_monthly_sales'].mean():,.2f}")

            st.subheader("Top Product-Store Combinations by Average Monthly Sales")
            display_cols = [
                col for col in [
                    "combination_label",
                    "product_category_name",
                    "product_subcategory_name",
                    "avg_monthly_sales",
                    "historical_records",
                    "avg_current_month_net_sales",
                    "avg_previous_month_sales",
                    "avg_rolling_avg_sales_3m",
                ] if col in summary_df.columns
            ]
            st.dataframe(summary_df[display_cols].head(20), use_container_width=True)

            st.subheader("Select a Historical Combination to Use as Prediction Input")
            options_df = summary_df.head(100).copy()
            labels = options_df["combination_label"].fillna("Historical combination").astype(str).tolist()
            selected_label = st.selectbox("Choose product-store/customer pattern", labels)
            selected_idx = labels.index(selected_label)
            selected_summary_row = options_df.iloc[selected_idx]

            st.write("Selected combination profile:")
            st.dataframe(pd.DataFrame([selected_summary_row])[display_cols], use_container_width=True)

            if st.button("Use this combination in prediction form", type="primary"):
                prefill_record = find_matching_reference_row(explorer_reference_df, selected_summary_row)
                if not prefill_record:
                    st.error("Could not find a matching reference row for the selected combination.")
                else:
                    update_prediction_defaults_from_row(prefill_record)
                    st.rerun()

        with st.expander("Raw reference data sample"):
            st.dataframe(explorer_reference_df.head(50), use_container_width=True)


with tab_predict:
    st.header("Live Prediction")

    if "prefill_success_message" in st.session_state:
        st.success(st.session_state["prefill_success_message"])
        del st.session_state["prefill_success_message"]

    if not loaded_model_keys:
        st.warning("No models loaded from the API yet. Start FastAPI first and refresh this page.")
    else:
        selected_model_key = get_next_ab_model_key(loaded_model_keys, log_df)
        selected_model_info = metadata["models"][selected_model_key]
        required_features = selected_model_info.get("required_cols", [])
        selected_display_name = selected_model_info.get("display_name", get_model_display_name(selected_model_key))
        importance_df = load_feature_importance(selected_model_key, selected_model_info)
        feature_table = create_visible_feature_table(required_features, importance_df)

        st.info(
            f"Current A/B test model: **{selected_display_name}** "
            f"| MLflow {selected_model_info.get('metric_name', 'test_rmse')}: "
            f"**{selected_model_info.get('metric_value', 'NA')}** "
            f"| Model-required features: **{len(required_features)}**"
        )

        auto_features = [feature for feature in required_features if feature in DERIVED_FEATURES]
        if auto_features:
            st.caption("Auto-derived model inputs hidden from the user: " + ", ".join(auto_features))

        st.subheader("Selected Model Features by SHAP Importance")
        if feature_table.empty:
            st.warning("No feature metadata available for the selected model.")
        else:
            feature_importance_display = feature_table.copy()
            feature_importance_display["importance_%"] = (feature_importance_display["importance_pct"] * 100).round(2)
            st.dataframe(
                feature_importance_display[["label", "feature", "importance_%", "is_model_required_or_dependency"]],
                use_container_width=True,
            )

        st.subheader("Prediction Inputs")
        st.caption(
            "Product, customer, and store identifiers are shown as readable names where reference data is available. "
            "The numeric IDs are mapped back into the API payload automatically."
        )

        model_reference_df = enrich_reference_display_columns(load_reference_data_for_model(selected_model_key))
        if model_reference_df.empty:
            model_reference_df = explorer_reference_df

        with st.form("prediction_form"):
            input_record = build_input_form(
                required_features=required_features,
                importance_df=importance_df,
                reference_df=model_reference_df,
                prefill_record=st.session_state.prefill_record,
            )

            submitted = st.form_submit_button("Predict Next-Period Demand")

        if submitted:
            payload = {
                "model_key": selected_model_key,
                "data": [input_record],
            }

            try:
                response_json = api_post(api_base_url, "/predict", payload)
                prediction = extract_prediction(response_json)

                save_prediction_log(input_record, response_json, api_base_url)

                st.session_state["last_prediction_message"] = (
                    f"Predicted next-period net sales: {prediction:,.2f} using "
                    f"{response_json.get('model_display_name', selected_display_name)}. "
                    "The next prediction will automatically use the other A/B test model."
                )

                with st.expander("API response"):
                    st.json(response_json)

                with st.expander("Payload sent to API"):
                    st.json(payload)

                st.rerun()

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI. Make sure Docker/FastAPI is running on port 8000.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")


with tab_dashboard:
    st.header("Prediction Request Dashboard")

    if log_df.empty:
        st.warning("No predictions logged yet.")
    else:
        total_requests = len(log_df)
        avg_prediction = pd.to_numeric(log_df["predicted_next_period_net_sales"], errors="coerce").mean()
        latest_prediction = pd.to_numeric(log_df["predicted_next_period_net_sales"], errors="coerce").iloc[-1]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Prediction Requests", total_requests)
        col2.metric("Average Predicted Sales", f"{avg_prediction:,.2f}")
        col3.metric("Latest Prediction", f"{latest_prediction:,.2f}")

        st.subheader("Gradient Boosting vs Random Forest")
        comparison_df = model_summary_metrics(log_df, loaded_model_keys, metadata)
        render_model_comparison_cards(comparison_df)
        st.dataframe(comparison_df, use_container_width=True)

        st.subheader("A/B Test Usage")
        if "model_display_name" in log_df.columns:
            model_counts = log_df["model_display_name"].value_counts().reset_index()
            model_counts.columns = ["Model", "Requests"]
            st.bar_chart(model_counts.set_index("Model"))

        st.subheader("Latest Predictions")
        preferred_cols = [
            "timestamp",
            "model_display_name",
            "predicted_next_period_net_sales",
            "product_category_name",
            "product_subcategory_name",
            "product_model_name",
            "store_id",
            "customer_key",
            "current_month_net_sales",
            "previous_month_sales",
            "rolling_avg_sales_3m",
            "month_num",
        ]
        available_cols = [col for col in preferred_cols if col in log_df.columns]
        st.dataframe(
            log_df[available_cols].tail(20).sort_values("timestamp", ascending=False),
            use_container_width=True,
        )

        st.subheader("Prediction Trend by Model")
        trend_df = log_df.copy()
        trend_df["timestamp"] = pd.to_datetime(trend_df["timestamp"], errors="coerce")
        trend_df["predicted_next_period_net_sales"] = pd.to_numeric(
            trend_df["predicted_next_period_net_sales"], errors="coerce"
        )
        trend_df = trend_df.dropna(subset=["timestamp", "predicted_next_period_net_sales"])

        if not trend_df.empty:
            trend_df = trend_df.sort_values("timestamp")
            trend_pivot = trend_df.pivot_table(
                index="timestamp",
                columns="model_display_name",
                values="predicted_next_period_net_sales",
                aggfunc="mean",
            )
            st.line_chart(trend_pivot)


with tab_drift:
    st.header("Input Drift Monitoring")

    if log_df.empty:
        st.warning("No production input data available yet. Make predictions first.")
    elif not loaded_model_keys:
        st.warning("Model metadata is unavailable. Start FastAPI and refresh.")
    else:
        st.subheader("Side-by-Side Drift Comparison")
        drift_cols = st.columns(len(loaded_model_keys))

        for i, model_key in enumerate(loaded_model_keys):
            with drift_cols[i]:
                model_info = metadata["models"].get(model_key, {})
                display_name = model_info.get("display_name", get_model_display_name(model_key))
                required_features = model_info.get("required_cols", [])

                st.markdown(f"### {display_name}")

                reference_df = load_reference_data_for_model(model_key)
                production_df = log_df[log_df["model_key"] == model_key].copy()

                if reference_df.empty:
                    st.error(
                        f"Reference file missing for {display_name}. Expected one of: "
                        + ", ".join(REFERENCE_FILE_CANDIDATES.get(model_key, []))
                    )
                    continue

                if production_df.empty:
                    st.warning(f"No production predictions logged yet for {display_name}.")
                    continue

                drift_features = list(dict.fromkeys(required_features + SCENARIO_ADJUSTMENT_FEATURES))
                drift_df = calculate_drift(reference_df, production_df, drift_features)

                possible_drift_count = int(drift_df["drift_status"].eq("Possible Drift").sum()) if not drift_df.empty else 0

                st.metric("Reference Records", f"{len(reference_df):,}")
                st.metric("Production Records", f"{len(production_df):,}")
                st.metric("Features with Possible Drift", possible_drift_count)

                if possible_drift_count > 0:
                    st.warning("Possible drift detected.")
                else:
                    st.success("No major drift detected.")

                if drift_df.empty:
                    st.info("No overlapping features available for drift calculation.")
                else:
                    st.dataframe(
                        drift_df.sort_values(["drift_status", "feature"], ascending=[False, True]),
                        use_container_width=True,
                    )

        st.subheader("Combined Drift Detail")
        combined_rows = []
        for model_key in loaded_model_keys:
            model_info = metadata["models"].get(model_key, {})
            display_name = model_info.get("display_name", get_model_display_name(model_key))
            reference_df = load_reference_data_for_model(model_key)
            production_df = log_df[log_df["model_key"] == model_key].copy()
            if reference_df.empty or production_df.empty:
                continue
            required_features = model_info.get("required_cols", [])
            drift_features = list(dict.fromkeys(required_features + SCENARIO_ADJUSTMENT_FEATURES))
            drift_df = calculate_drift(reference_df, production_df, drift_features)
            if drift_df.empty:
                continue
            drift_df.insert(0, "model", display_name)
            combined_rows.append(drift_df)

        if combined_rows:
            combined_drift_df = pd.concat(combined_rows, ignore_index=True)
            st.dataframe(combined_drift_df, use_container_width=True)
        else:
            st.info("Combined drift table will appear after both models receive production predictions.")

        with st.expander("Production log used for monitoring"):
            st.dataframe(log_df, use_container_width=True)
