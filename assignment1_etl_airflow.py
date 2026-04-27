from __future__ import annotations

from contextlib import closing
from datetime import datetime

import mysql.connector
import pandas as pd
from sqlalchemy import create_engine

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

# -------------------------------------------------------------------
# Same source, warehouse and credentials as the notebook
# -------------------------------------------------------------------
MYSQL_HOST = "localhost"
MYSQL_USER = "bt4301"
MYSQL_PASSWORD = "password"
SOURCE_DB = "adventureworks2012"
DW_DB = "datawarehouse"

# -------------------------------------------------------------------
# Incremental period configuration
# -------------------------------------------------------------------
PERIOD_COUNTER_VAR = "aw_store_sales_current_period"
BASE_PERIOD_START = pd.Timestamp("2005-07-01")
MAX_PERIOD = 37

# Use every 5 minutes for submission. For testing, you can change to "* * * * *"
DAG_SCHEDULE = "*/5 * * * *"


def get_dw_engine():
    return create_engine(
        f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:3306/{DW_DB}",
        echo=False,
    )


def get_mysql_connection(database: str):
    return mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        passwd=MYSQL_PASSWORD,
        database=database,
    )


def get_period_window(period: int) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if 1 <= period <= MAX_PERIOD:
        period_start = BASE_PERIOD_START + pd.DateOffset(months=period - 1)
        period_end = period_start + pd.DateOffset(months=1)
        return period_start, period_end
    return None, None


def ensure_warehouse_objects() -> None:
    with closing(get_mysql_connection(DW_DB)) as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS dim_store_customer (
                    customer_key INT AUTO_INCREMENT PRIMARY KEY,
                    customer_id INT NOT NULL,
                    store_id INT NOT NULL,
                    store_name VARCHAR(100),
                    account_number VARCHAR(30),
                    territory_id INT,
                    store_salesperson_id INT NULL,
                    UNIQUE KEY uk_dim_store_customer_customer_id (customer_id)
                );
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS dim_product (
                    product_key INT AUTO_INCREMENT PRIMARY KEY,
                    product_id INT NOT NULL,
                    product_name VARCHAR(100),
                    product_number VARCHAR(50),
                    color VARCHAR(30),
                    size VARCHAR(10),
                    size_unit_measure_code CHAR(6),
                    weight_unit_measure_code CHAR(6),
                    weight DECIMAL(8,2),
                    standard_cost DECIMAL(19,4),
                    list_price DECIMAL(19,4),
                    product_model_name VARCHAR(100),
                    product_subcategory_name VARCHAR(100),
                    product_category_name VARCHAR(100),
                    sell_start_date DATETIME NULL,
                    sell_end_date DATETIME NULL,
                    discontinued_date DATETIME NULL,
                    UNIQUE KEY uk_dim_product_product_id (product_id)
                );
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS dim_time (
                    time_key INT PRIMARY KEY,
                    full_date DATE NOT NULL,
                    day_of_month TINYINT,
                    month_num TINYINT,
                    month_name VARCHAR(20),
                    quarter_num TINYINT,
                    year_num SMALLINT,
                    week_of_year TINYINT,
                    day_name VARCHAR(20),
                    is_weekend TINYINT,
                    UNIQUE KEY uk_dim_time_full_date (full_date)
                );
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS dim_salesperson (
                    salesperson_key INT AUTO_INCREMENT PRIMARY KEY,
                    salesperson_id INT NOT NULL,
                    full_name VARCHAR(250),
                    title VARCHAR(16),
                    sales_quota DECIMAL(19,4) NULL,
                    bonus DECIMAL(19,4) NULL,
                    commission_pct DECIMAL(10,4) NULL,
                    sales_ytd DECIMAL(19,4) NULL,
                    sales_last_year DECIMAL(19,4) NULL,
                    UNIQUE KEY uk_dim_salesperson_salesperson_id (salesperson_id)
                );
                """
            )

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS raw_store_sales (
                    raw_store_sales_id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    sales_order_id INT NOT NULL,
                    sales_order_detail_id INT NOT NULL,
                    order_date DATETIME NOT NULL,
                    due_date DATETIME NULL,
                    ship_date DATETIME NULL,
                    status TINYINT,
                    sales_order_number VARCHAR(50),
                    purchase_order_number VARCHAR(50),
                    account_number VARCHAR(30),
                    customer_id INT NOT NULL,
                    store_id INT NOT NULL,
                    store_name VARCHAR(100),
                    salesperson_id INT NULL,
                    territory_id INT NULL,
                    product_id INT NOT NULL,
                    order_qty SMALLINT,
                    unit_price DECIMAL(19,4),
                    unit_price_discount DECIMAL(19,4),
                    line_total DECIMAL(38,6),
                    subtotal DECIMAL(19,4),
                    tax_amt DECIMAL(19,4),
                    freight DECIMAL(19,4),
                    total_due DECIMAL(19,4),
                    currency_rate_id INT NULL,
                    ship_method_id INT NULL,
                    comment VARCHAR(256),
                    UNIQUE KEY uk_raw_store_sales (sales_order_id, sales_order_detail_id)
                );
                ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fact_store_sales (
                    fact_store_sales_key BIGINT AUTO_INCREMENT PRIMARY KEY,
                    sales_order_id INT NOT NULL,
                    sales_order_detail_id INT NOT NULL,
                    sales_order_number VARCHAR(50),
                    customer_key INT NOT NULL,
                    product_key INT NOT NULL,
                    order_date_key INT NOT NULL,
                    salesperson_key INT NOT NULL,
                    order_qty SMALLINT NOT NULL,
                    unit_price DECIMAL(19,4) NOT NULL,
                    unit_price_discount DECIMAL(19,4) NOT NULL,
                    gross_amount DECIMAL(19,4) NOT NULL,
                    discount_amount DECIMAL(19,4) NOT NULL,
                    margin_amount DECIMAL(38,6),
                    net_amount DECIMAL(38,6) NOT NULL,
                    header_subtotal DECIMAL(19,4),
                    header_tax_amt DECIMAL(19,4),
                    header_freight DECIMAL(19,4),
                    header_total_due DECIMAL(19,4),
                    UNIQUE KEY uk_fact_store_sales (sales_order_id, sales_order_detail_id),
                    CONSTRAINT fk_fact_customer FOREIGN KEY (customer_key) REFERENCES dim_store_customer(customer_key),
                    CONSTRAINT fk_fact_product FOREIGN KEY (product_key) REFERENCES dim_product(product_key),
                    CONSTRAINT fk_fact_time FOREIGN KEY (order_date_key) REFERENCES dim_time(time_key),
                    CONSTRAINT fk_fact_salesperson FOREIGN KEY (salesperson_key) REFERENCES dim_salesperson(salesperson_key)
                );
                ''')

            cursor.execute(
                """
                INSERT IGNORE INTO dim_salesperson
                    (salesperson_id, full_name, title, sales_quota, bonus, commission_pct, sales_ytd, sales_last_year)
                VALUES
                    (-1, 'Unknown', NULL, NULL, NULL, NULL, NULL, NULL);
                """
            )

        conn.commit()


@dag(
    dag_id="adventureworks_store_sales_incremental_dag_v2",
    schedule=DAG_SCHEDULE,
    start_date=datetime(2026, 3, 22),
    catchup=False,
    max_active_runs=1,
    tags=["adventureworks", "airflow", "etl", "incremental"],
)
def adventureworks_store_sales_incremental_dag():
    start = EmptyOperator(task_id="start")
    do_nothing = EmptyOperator(task_id="do_nothing")
    end = EmptyOperator(
        task_id="end",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    @task
    def determine_current_period() -> dict:
        current_period = int(Variable.get(PERIOD_COUNTER_VAR, default_var="1"))

        if current_period < 1:
            current_period = 1
            Variable.set(PERIOD_COUNTER_VAR, "1")

        period_start, period_end = get_period_window(current_period)

        payload = {
            "current_period": current_period,
            "period_start": period_start.strftime("%Y-%m-%d") if period_start is not None else None,
            "period_end": period_end.strftime("%Y-%m-%d") if period_end is not None else None,
        }
        print(payload)
        return payload

    @task.branch
    def check_period(period_ctx: dict) -> str:
        current_period = int(period_ctx["current_period"])
        if 1 <= current_period <= MAX_PERIOD:
            return "etl_new_unique_dimension_data_for_current_period"
        return "do_nothing"

    @task(task_id="etl_new_unique_dimension_data_for_current_period")
    def etl_new_unique_dimension_data_for_current_period(period_ctx: dict) -> dict:
        ensure_warehouse_objects()

        period_start = period_ctx["period_start"]
        period_end = period_ctx["period_end"]
        dw_conn = get_mysql_connection(DW_DB)

        # 1. Store customer dimension
        str_sql = f'''
        SELECT DISTINCT
            c.CustomerID AS customer_id,
            c.StoreID AS store_id,
            st.Name AS store_name,
            c.AccountNumber AS account_number,
            c.TerritoryID AS territory_id,
            st.SalesPersonID AS store_salesperson_id
        FROM {SOURCE_DB}.salesorderheader soh
        JOIN {SOURCE_DB}.customer c
            ON soh.CustomerID = c.CustomerID
        JOIN {SOURCE_DB}.store st
            ON c.StoreID = st.BusinessEntityID
        LEFT JOIN {DW_DB}.dim_store_customer d
            ON c.CustomerID = d.customer_id
        WHERE c.StoreID IS NOT NULL
          AND soh.OrderDate >= '{period_start}'
          AND soh.OrderDate < '{period_end}'
          AND d.customer_id IS NULL
        ORDER BY c.CustomerID ASC;
        '''
        df_store_customer = pd.read_sql(sql=str_sql, con=dw_conn)
        print("New store customers to load:", len(df_store_customer))
        if len(df_store_customer) > 0:
            cursor = dw_conn.cursor()
            for _, row in df_store_customer.iterrows():
                # Convert NaN to None for MySQL NULL compatibility
                row_values = tuple(None if pd.isna(val) else val for val in row)
                cursor.execute('''
                    INSERT INTO dim_store_customer 
                    (customer_id, store_id, store_name, account_number, territory_id, store_salesperson_id)
                    VALUES (%s, %s, %s, %s, %s, %s)
                ''', row_values)
            dw_conn.commit()
            cursor.close()

        # 2. Product dimension
        str_sql = f'''
        SELECT DISTINCT
            p.ProductID AS product_id,
            p.Name AS product_name,
            p.ProductNumber AS product_number,
            p.Color AS color,
            p.Size AS size,
            p.SizeUnitMeasureCode AS size_unit_measure_code,
            p.WeightUnitMeasureCode AS weight_unit_measure_code,
            p.Weight AS weight,
            p.StandardCost AS standard_cost,
            p.ListPrice AS list_price,
            pm.Name AS product_model_name,
            psc.Name AS product_subcategory_name,
            pc.Name AS product_category_name,
            p.SellStartDate AS sell_start_date,
            p.SellEndDate AS sell_end_date,
            p.DiscontinuedDate AS discontinued_date
        FROM {SOURCE_DB}.salesorderheader soh
        JOIN {SOURCE_DB}.salesorderdetail sod
            ON soh.SalesOrderID = sod.SalesOrderID
        JOIN {SOURCE_DB}.customer c
            ON soh.CustomerID = c.CustomerID
        JOIN {SOURCE_DB}.product p
            ON sod.ProductID = p.ProductID
        LEFT JOIN {SOURCE_DB}.productmodel pm
            ON p.ProductModelID = pm.ProductModelID
        LEFT JOIN {SOURCE_DB}.productsubcategory psc
            ON p.ProductSubcategoryID = psc.ProductSubcategoryID
        LEFT JOIN {SOURCE_DB}.productcategory pc
            ON psc.ProductCategoryID = pc.ProductCategoryID
        LEFT JOIN {DW_DB}.dim_product d
            ON p.ProductID = d.product_id
        WHERE c.StoreID IS NOT NULL
          AND soh.OrderDate >= '{period_start}'
          AND soh.OrderDate < '{period_end}'
          AND d.product_id IS NULL
        ORDER BY p.ProductID ASC;
        '''
        df_product = pd.read_sql(sql=str_sql, con=dw_conn)
        print("New products to load:", len(df_product))
        if len(df_product) > 0:
            cursor = dw_conn.cursor()
            for _, row in df_product.iterrows():
                # Convert NaN to None for MySQL NULL compatibility
                row_values = tuple(None if pd.isna(val) else val for val in row)
                cursor.execute('''
                    INSERT INTO dim_product 
                    (product_id, product_name, product_number, color, size, size_unit_measure_code, 
                    weight_unit_measure_code, weight, standard_cost, list_price, product_model_name, 
                    product_subcategory_name, product_category_name, sell_start_date, sell_end_date, discontinued_date)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', row_values)
            dw_conn.commit()
            cursor.close()

        # 3. Time dimension
        str_sql = f'''
        SELECT DISTINCT
            CAST(DATE_FORMAT(DATE(soh.OrderDate), '%Y%m%d') AS UNSIGNED) AS time_key,
            DATE(soh.OrderDate) AS full_date,
            DAY(DATE(soh.OrderDate)) AS day_of_month,
            MONTH(DATE(soh.OrderDate)) AS month_num,
            MONTHNAME(DATE(soh.OrderDate)) AS month_name,
            QUARTER(DATE(soh.OrderDate)) AS quarter_num,
            YEAR(DATE(soh.OrderDate)) AS year_num,
            WEEK(DATE(soh.OrderDate), 3) AS week_of_year,
            DAYNAME(DATE(soh.OrderDate)) AS day_name,
            CASE
                WHEN DAYOFWEEK(DATE(soh.OrderDate)) IN (1, 7) THEN 1
                ELSE 0
            END AS is_weekend
        FROM {SOURCE_DB}.salesorderheader soh
        JOIN {SOURCE_DB}.customer c
            ON soh.CustomerID = c.CustomerID
        LEFT JOIN {DW_DB}.dim_time d
            ON DATE(soh.OrderDate) = d.full_date
        WHERE c.StoreID IS NOT NULL
          AND soh.OrderDate >= '{period_start}'
          AND soh.OrderDate < '{period_end}'
          AND d.full_date IS NULL
        ORDER BY full_date ASC;
        '''
        df_time = pd.read_sql(sql=str_sql, con=dw_conn)
        print("New dates to load:", len(df_time))
        if len(df_time) > 0:
            cursor = dw_conn.cursor()
            for _, row in df_time.iterrows():
                # Convert NaN to None for MySQL NULL compatibility
                row_values = tuple(None if pd.isna(val) else val for val in row)
                cursor.execute('''
                    INSERT INTO dim_time 
                    (time_key, full_date, day_of_month, month_num, month_name, quarter_num, year_num, week_of_year, day_name, is_weekend)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', row_values)
            dw_conn.commit()
            cursor.close()

        # 4. Salesperson dimension
        str_sql = f'''
        SELECT DISTINCT
            COALESCE(soh.SalesPersonID, st.SalesPersonID) AS salesperson_id,
            TRIM(
                CONCAT(
                    COALESCE(p.FirstName, ''),
                    CASE WHEN p.MiddleName IS NOT NULL THEN CONCAT(' ', p.MiddleName) ELSE '' END,
                    CASE WHEN p.LastName IS NOT NULL THEN CONCAT(' ', p.LastName) ELSE '' END
                )
            ) AS full_name,
            p.Title AS title,
            sp.SalesQuota AS sales_quota,
            sp.Bonus AS bonus,
            sp.CommissionPct AS commission_pct,
            sp.SalesYTD AS sales_ytd,
            sp.SalesLastYear AS sales_last_year
        FROM {SOURCE_DB}.salesorderheader soh
        JOIN {SOURCE_DB}.customer c
            ON soh.CustomerID = c.CustomerID
        JOIN {SOURCE_DB}.store st
            ON c.StoreID = st.BusinessEntityID
        LEFT JOIN {SOURCE_DB}.salesperson sp
            ON sp.BusinessEntityID = COALESCE(soh.SalesPersonID, st.SalesPersonID)
        LEFT JOIN {SOURCE_DB}.person p
            ON p.BusinessEntityID = sp.BusinessEntityID
        LEFT JOIN {DW_DB}.dim_salesperson d
            ON COALESCE(soh.SalesPersonID, st.SalesPersonID) = d.salesperson_id
        WHERE c.StoreID IS NOT NULL
          AND soh.OrderDate >= '{period_start}'
          AND soh.OrderDate < '{period_end}'
          AND COALESCE(soh.SalesPersonID, st.SalesPersonID) IS NOT NULL
          AND d.salesperson_id IS NULL
        ORDER BY salesperson_id ASC;
        '''
        df_salesperson = pd.read_sql(sql=str_sql, con=dw_conn)
        print("New salespersons to load:", len(df_salesperson))
        if len(df_salesperson) > 0:
            cursor = dw_conn.cursor()
            for _, row in df_salesperson.iterrows():
                # Convert NaN to None for MySQL NULL compatibility
                row_values = tuple(None if pd.isna(val) else val for val in row)
                cursor.execute('''
                    INSERT IGNORE INTO dim_salesperson 
                    (salesperson_id, full_name, title, sales_quota, bonus, commission_pct, sales_ytd, sales_last_year)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ''', row_values)
            dw_conn.commit()
            cursor.close()

        return period_ctx

    @task(task_id="etl_sales_fact_data_for_current_period")
    def etl_sales_fact_data_for_current_period(period_ctx: dict) -> dict:
        ensure_warehouse_objects()

        period_start = period_ctx["period_start"]
        period_end = period_ctx["period_end"]
        current_period = int(period_ctx["current_period"])
        dw_conn = get_mysql_connection(DW_DB)

        # -----------------------------------------------------------------
        # Load only new raw fact rows for the current period
        # -----------------------------------------------------------------
        str_sql = f'''
        SELECT
        soh.SalesOrderID AS sales_order_id,
        sod.SalesOrderDetailID AS sales_order_detail_id,
        soh.OrderDate AS order_date,
        soh.DueDate AS due_date,
        soh.ShipDate AS ship_date,
        soh.Status AS status,
        soh.SalesOrderNumber AS sales_order_number,
        soh.PurchaseOrderNumber AS purchase_order_number,
        soh.AccountNumber AS account_number,
        soh.CustomerID AS customer_id,
        c.StoreID AS store_id,
        st.Name AS store_name,
        COALESCE(soh.SalesPersonID, st.SalesPersonID) AS salesperson_id,
        c.TerritoryID AS territory_id,
        sod.ProductID AS product_id,
        sod.OrderQty AS order_qty,
        sod.UnitPrice AS unit_price,
        sod.UnitPriceDiscount AS unit_price_discount,
        sod.LineTotal AS line_total,
        soh.SubTotal AS subtotal,
        soh.TaxAmt AS tax_amt,
        soh.Freight AS freight,
        soh.TotalDue AS total_due,
        soh.CurrencyRateID AS currency_rate_id,
        soh.ShipMethodID AS ship_method_id,
        soh.Comment AS comment
        FROM {SOURCE_DB}.salesorderheader soh
        JOIN {SOURCE_DB}.salesorderdetail sod
            ON soh.SalesOrderID = sod.SalesOrderID
        JOIN {SOURCE_DB}.customer c
            ON soh.CustomerID = c.CustomerID
        JOIN {SOURCE_DB}.store st
            ON c.StoreID = st.BusinessEntityID
        LEFT JOIN {DW_DB}.raw_store_sales r
            ON soh.SalesOrderID = r.sales_order_id
           AND sod.SalesOrderDetailID = r.sales_order_detail_id
        WHERE c.StoreID IS NOT NULL
          AND soh.OrderDate >= '{period_start}'
          AND soh.OrderDate < '{period_end}'
          AND r.sales_order_id IS NULL
        ORDER BY soh.SalesOrderID, sod.SalesOrderDetailID ASC;
        '''
        df_raw = pd.read_sql(sql=str_sql, con=dw_conn)
        print("New raw store sales rows to load:", len(df_raw))
        if len(df_raw) > 0:
            cursor = dw_conn.cursor()
            for _, row in df_raw.iterrows():
                # Convert NaN to None for MySQL NULL compatibility
                row_values = tuple(None if pd.isna(val) else val for val in row)
                cursor.execute('''
            INSERT INTO raw_store_sales 
            (sales_order_id, sales_order_detail_id, order_date, due_date, ship_date, status, 
            sales_order_number, purchase_order_number, account_number, customer_id, store_id, 
            store_name, salesperson_id, territory_id, product_id, order_qty, unit_price, 
            unit_price_discount, line_total, subtotal, tax_amt, freight, total_due, 
            currency_rate_id, ship_method_id, comment)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', row_values)
            dw_conn.commit()
            cursor.close()

        # -----------------------------------------------------------------
        # Transform current-period raw rows into fact rows
        # -----------------------------------------------------------------
        str_sql = f'''
        SELECT
        r.sales_order_id,
        r.sales_order_detail_id,
        r.sales_order_number,
        r.customer_id,
        r.product_id,
        COALESCE(r.salesperson_id, -1) AS salesperson_id,
        DATE(r.order_date) AS order_date,
        DATE(r.ship_date) AS ship_date,
        r.order_qty,
        r.unit_price,
        r.unit_price_discount,
        r.line_total,
        r.subtotal,
        r.tax_amt,
        r.freight,
        r.total_due,
        p.standard_cost
    FROM {DW_DB}.raw_store_sales r
    LEFT JOIN {DW_DB}.fact_store_sales f
        ON r.sales_order_id = f.sales_order_id
    AND r.sales_order_detail_id = f.sales_order_detail_id
    LEFT JOIN {DW_DB}.dim_product p
        ON r.product_id = p.product_id
    WHERE r.order_date >= '{period_start}'
    AND r.order_date < '{period_end}'
    AND f.sales_order_id IS NULL
    ORDER BY r.sales_order_id, r.sales_order_detail_id ASC;
        '''
        df_fact_stage = pd.read_sql(sql=str_sql, con=dw_conn)
        print("Raw rows to transform into fact rows:", len(df_fact_stage))

        if len(df_fact_stage) > 0:
            df_customer_map = pd.read_sql(
                "SELECT customer_key, customer_id FROM dim_store_customer;",
                con=dw_conn,
            )
            df_product_map = pd.read_sql(
                "SELECT product_key, product_id FROM dim_product;",
                con=dw_conn,
            )
            df_time_map = pd.read_sql(
                "SELECT time_key, full_date FROM dim_time;",
                con=dw_conn,
            )
            df_time_map["full_date"] = pd.to_datetime(df_time_map["full_date"]).dt.date
            df_salesperson_map = pd.read_sql(
                "SELECT salesperson_key, salesperson_id FROM dim_salesperson;",
                con=dw_conn,
            )

            df_fact = df_fact_stage.copy()
            df_fact['order_date'] = pd.to_datetime(df_fact['order_date']).dt.date
            df_fact['ship_date'] = pd.to_datetime(df_fact['ship_date']).dt.date

            df_fact = df_fact.merge(df_customer_map, on='customer_id', how='left')
            df_fact = df_fact.merge(df_product_map, on='product_id', how='left')
            df_fact = df_fact.merge(df_time_map, left_on='order_date', right_on='full_date', how='left')
            df_fact = df_fact.merge(df_salesperson_map, on='salesperson_id', how='left')

            # Row-level transformations
            df_fact['gross_amount'] = (df_fact['order_qty'] * df_fact['unit_price']).round(4)
            df_fact['discount_amount'] = (
                df_fact['order_qty'] * df_fact['unit_price'] * df_fact['unit_price_discount']
            ).round(4)
            # df_fact['net_sales_amount'] = (df_fact['gross_amount'] - df_fact['discount_amount']).round(4)
            df_fact['net_amount'] = df_fact['line_total']
            df_fact['margin_amount'] = (df_fact['net_amount'] - (df_fact['order_qty'] * df_fact['standard_cost']).round(4)).round(4)

            missing_customer = df_fact['customer_key'].isna().sum()
            missing_product = df_fact['product_key'].isna().sum()
            missing_time = df_fact['time_key'].isna().sum()
            missing_salesperson = df_fact['salesperson_key'].isna().sum()

            print('Missing customer keys   :', missing_customer)
            print('Missing product keys    :', missing_product)
            print('Missing time keys       :', missing_time)
            print('Missing salesperson keys:', missing_salesperson)

            if missing_customer or missing_product or missing_time or missing_salesperson:
                raise ValueError('One or more surrogate keys could not be resolved.')

            df_fact = df_fact[[
                'sales_order_id',
                'sales_order_detail_id',
                'sales_order_number',
                'customer_key',
                'product_key',
                'time_key',
                'salesperson_key',
                'order_qty',
                'unit_price',
                'unit_price_discount',
                'gross_amount',
                'discount_amount',
                'margin_amount',
                'net_amount',
                'subtotal',
                'tax_amt',
                'freight',
                'total_due'
            ]].rename(columns={
                'time_key': 'order_date_key',
                'subtotal': 'header_subtotal',
                'tax_amt': 'header_tax_amt',
                'freight': 'header_freight',
                'total_due': 'header_total_due'
            })


            if len(df_fact) > 0:
                cursor = dw_conn.cursor()
                for _, row in df_fact.iterrows():
                    # Convert NaN to None for MySQL NULL compatibility
                    row_values = tuple(None if pd.isna(val) else val for val in row)
                    cursor.execute('''
                        INSERT INTO fact_store_sales
                        (sales_order_id, sales_order_detail_id, sales_order_number, customer_key, product_key,
                        order_date_key, salesperson_key, order_qty, unit_price, unit_price_discount,
                        gross_amount, discount_amount, margin_amount, net_amount, 
                        header_subtotal, header_tax_amt, header_freight, header_total_due)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ''', row_values)
                dw_conn.commit()
                cursor.close()

        # Increment the counter only after a successful full run for the period
        Variable.set(PERIOD_COUNTER_VAR, str(current_period + 1))

        print(f"Finished loading period {current_period}. Next period = {current_period + 1}")
        return {
            "loaded_period": current_period,
            "next_period": current_period + 1,
        }

    period_ctx = determine_current_period()
    decision = check_period(period_ctx)

    start >> period_ctx >> decision
    decision >> etl_new_unique_dimension_data_for_current_period(period_ctx) >> etl_sales_fact_data_for_current_period(period_ctx) >> end
    decision >> do_nothing >> end


dag = adventureworks_store_sales_incremental_dag()
