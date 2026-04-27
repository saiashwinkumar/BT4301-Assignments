from __future__ import annotations

import hashlib
import os
from contextlib import closing
from datetime import date, datetime
from decimal import Decimal

import mysql.connector
from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator

# -------------------------------------------------------------------
# Same warehouse credentials as the notebook / ETL DAG
# -------------------------------------------------------------------
MYSQL_HOST = "localhost"
MYSQL_USER = "bt4301"
MYSQL_PASSWORD = "password"
DW_DB = "datawarehouse"

# Daily verification. For testing, you can change this to "* * * * *"
DAG_SCHEDULE = "0 0 * * *"
LOG_DIR = "/tmp/fact_store_sales_watermark_logs"

FINGERPRINT_COLUMNS = [
    "fact_store_sales_key",
    "sales_order_id",
    "sales_order_detail_id",
    "sales_order_number",
    "customer_key",
    "product_key",
    "order_date_key",
    "salesperson_key",
    "order_qty",
    "unit_price",
    "unit_price_discount",
    "gross_amount",
    "discount_amount",
    "margin_amount",
    "net_amount",
    "header_subtotal",
    "header_tax_amt",
    "header_freight",
    "header_total_due",
]


def get_mysql_connection():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        passwd=MYSQL_PASSWORD,
        database=DW_DB,
    )


def normalize_value(value) -> str:
    if value is None:
        return "<NULL>"
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def compute_sha256_fingerprint(row_dict: dict) -> str:
    payload = "||".join(normalize_value(row_dict.get(col)) for col in FINGERPRINT_COLUMNS)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


from airflow.sdk import get_current_context

def get_log_filename() -> str:
    context = get_current_context()
    run_start = context["dag_run"].start_date or context["logical_date"]
    return run_start.strftime("%Y%m%d%H%M") + ".log"


@dag(
    dag_id="fact_store_sales_verify_watermark_dag",
    schedule=DAG_SCHEDULE,
    start_date=datetime(2026, 3, 22),
    catchup=False,
    max_active_runs=1,
    tags=["adventureworks", "watermark", "verification", "sha256"],
)
def fact_store_sales_verify_watermark_dag():
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    @task
    def ensure_fingerprint_column_exists() -> str:
        with closing(get_mysql_connection()) as conn:
            with closing(conn.cursor()) as cursor:
                # Check if fingerprint column already exists
                cursor.execute(
                    """
                    SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_NAME = 'fact_store_sales' 
                    AND COLUMN_NAME = 'fingerprint'
                    AND TABLE_SCHEMA = %s
                    """,
                    (DW_DB,)
                )
                if not cursor.fetchone():
                    # Column doesn't exist, create it
                    cursor.execute(
                        """
                        ALTER TABLE fact_store_sales
                        ADD COLUMN fingerprint CHAR(64) NULL
                        """
                    )
            conn.commit()
        return "fingerprint column ready"

    @task
    def verify_fact_fingerprints(_: str) -> dict:
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, get_log_filename())

        select_sql = f"""
            SELECT {', '.join(FINGERPRINT_COLUMNS)}, fingerprint
            FROM fact_store_sales
            ORDER BY fact_store_sales_key
        """

        total_rows = 0
        invalid_rows = 0

        with closing(get_mysql_connection()) as conn:
            with closing(conn.cursor(dictionary=True)) as cursor:
                cursor.execute(select_sql)
                rows = cursor.fetchall()

        total_rows = len(rows)

        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write("Fact Store Sales Watermark Verification Log\n")
            log_file.write(f"Rows checked: {total_rows}\n")
            log_file.write(f"Generated at: {datetime.utcnow().isoformat()}Z\n\n")

            for row in rows:
                stored_fingerprint = row.pop("fingerprint", None)
                expected_fingerprint = compute_sha256_fingerprint(row)

                if not stored_fingerprint:
                    invalid_rows += 1
                    log_file.write(
                        "MISSING_FINGERPRINT | "
                        f"fact_store_sales_key={row['fact_store_sales_key']} | "
                        f"sales_order_id={row['sales_order_id']} | "
                        f"sales_order_detail_id={row['sales_order_detail_id']}\n"
                    )
                elif stored_fingerprint != expected_fingerprint:
                    invalid_rows += 1
                    log_file.write(
                        "FINGERPRINT_MISMATCH | "
                        f"fact_store_sales_key={row['fact_store_sales_key']} | "
                        f"sales_order_id={row['sales_order_id']} | "
                        f"sales_order_detail_id={row['sales_order_detail_id']} | "
                        f"stored={stored_fingerprint} | expected={expected_fingerprint}\n"
                    )

            if invalid_rows == 0:
                log_file.write("No invalid rows found.\n")

        return {
            "rows_checked": total_rows,
            "invalid_rows": invalid_rows,
            "log_file": log_path,
        }

    verified = verify_fact_fingerprints(ensure_fingerprint_column_exists())
    start >> verified >> end


dag = fact_store_sales_verify_watermark_dag()
