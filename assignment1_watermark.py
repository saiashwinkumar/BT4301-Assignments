from __future__ import annotations

import hashlib
from contextlib import closing
from datetime import date, datetime
from decimal import Decimal

import mysql.connector
from airflow.decorators import dag, task

# -------------------------------------------------------------------
# Same warehouse credentials as the notebook / ETL DAG
# -------------------------------------------------------------------
MYSQL_HOST = "localhost"
MYSQL_USER = "bt4301"
MYSQL_PASSWORD = "password"
DW_DB = "datawarehouse"

# Watermark new fact rows frequently so the ETL pipeline remains unchanged.
# For testing, you can change this to "* * * * *"
DAG_SCHEDULE = "*/5 * * * *"

# Columns covered by the fingerprint, in fixed order.
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


@dag(
    dag_id="fact_store_sales_watermark_dag",
    schedule=DAG_SCHEDULE,
    start_date=datetime(2026, 3, 17),
    catchup=False,
    max_active_runs=1,
    tags=["adventureworks", "watermark", "sha256", "integrity"],
)
def fact_store_sales_watermark_dag():
    @task
    def ensure_fingerprint_column() -> str:
        column_exists_sql = """
            SELECT COUNT(*)
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = %s
              AND TABLE_NAME = 'fact_store_sales'
              AND COLUMN_NAME = 'fingerprint'
        """

        add_column_sql = """
            ALTER TABLE fact_store_sales
            ADD COLUMN fingerprint CHAR(64) NULL
        """

        with closing(get_mysql_connection()) as conn:
            with closing(conn.cursor()) as cursor:
                cursor.execute(column_exists_sql, (DW_DB,))
                column_exists = cursor.fetchone()[0] > 0

                if not column_exists:
                    cursor.execute(add_column_sql)
            conn.commit()
        return "fingerprint column ready"

    @task
    def watermark_new_or_missing_rows(_: str) -> dict:
        updated_rows = 0
        scanned_rows = 0

        select_sql = f"""
            SELECT {', '.join(FINGERPRINT_COLUMNS)}
            FROM fact_store_sales
            WHERE fingerprint IS NULL OR fingerprint = ''
            ORDER BY fact_store_sales_key
        """

        update_sql = """
            UPDATE fact_store_sales
            SET fingerprint = %s
            WHERE fact_store_sales_key = %s
        """

        with closing(get_mysql_connection()) as conn:
            with closing(conn.cursor(dictionary=True)) as cursor:
                cursor.execute(select_sql)
                rows = cursor.fetchall()

            scanned_rows = len(rows)

            if rows:
                with closing(conn.cursor()) as update_cursor:
                    for row in rows:
                        fingerprint = compute_sha256_fingerprint(row)
                        update_cursor.execute(
                            update_sql,
                            (fingerprint, row["fact_store_sales_key"]),
                        )
                        updated_rows += 1
                conn.commit()

        return {
            "rows_scanned_without_fingerprint": scanned_rows,
            "rows_watermarked": updated_rows,
        }

    watermark_new_or_missing_rows(ensure_fingerprint_column())


dag = fact_store_sales_watermark_dag()
