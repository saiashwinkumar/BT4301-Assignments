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
    python baseline_linear_regression_mlflow.py

Optional CSV usage:
    python baseline_linear_regression_mlflow.py --csv-path ../data/ml_dataset.csv

If --csv-path is not provided, the script loads the dataset from the local MySQL
datawarehouse database used in the notebook.
"""

import argparse
import os
import warnings

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

from sklearn.linear_model import LinearRegression

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
    parser = argparse.ArgumentParser(description="Run baseline Linear Regression experiment and register model in MLflow.")
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
        default="Sales Demand Forecasting - Baseline Linear Regression",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--registered-model-name",
        default="sales_baseline_linear_regression",
        help="Name to use in the MLflow Model Registry.",
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




def main():
    args = parse_args()

    df = load_data(args)
    df_model = engineer_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df_model)
    preprocessor = build_preprocessor(X_train)

    params = {
        "fit_intercept": True,
    }

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression(**params)),
        ]
    )

    pipeline.fit(X_train, y_train)

    val_metrics = evaluate_model(pipeline, X_val, y_val)
    test_metrics = evaluate_model(pipeline, X_test, y_test)

    mlflow.set_tracking_uri(uri=args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name="baseline_linear_regression"):
        mlflow.log_params(params)
        log_dataset_shape(X_train, X_val, X_test)
        log_metrics_with_prefix("validation", val_metrics)
        log_metrics_with_prefix("test", test_metrics)

        mlflow.set_tag(
            "Training Info",
            "Baseline Linear Regression for next-period net sales forecasting",
        )
        mlflow.set_tag("model_family", "linear_model")
        mlflow.set_tag("target", "next_period_net_sales")
        mlflow.set_tag("split_strategy", "70_train_10_validation_20_test")

        input_example = X_train.head(5)
        signature = infer_signature(input_example, pipeline.predict(input_example))

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="baseline_linear_regression_model",
            signature=signature,
            input_example=input_example,
            registered_model_name=args.registered_model_name,
        )

        print("Validation metrics:", val_metrics)
        print("Test metrics:", test_metrics)
        print("Model URI:", model_info.model_uri)



if __name__ == "__main__":
    main()
