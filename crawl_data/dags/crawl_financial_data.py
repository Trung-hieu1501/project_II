import pandas as pd
import requests
from google.cloud import storage
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging
import io

def crawl_financial_ratios(): 
    # Đọc dữ liệu từ BigQuery
    logging.info("Bắt đầu load dữ liệu")
    client = storage.Client()
    bucket = client.get_bucket("trunghieund")
    blob = bucket.blob("symbols.csv")
    data = blob.download_as_text()  # Tải file dưới dạng văn bản
    symbols = pd.read_csv(io.StringIO(data))  # Đọc vào DataFrame
    logging.info("Load dữ liệu thành công")

    financial_ratios = []
    for i in symbols['symbol']:
        try:
            response = requests.get(f"https://s.cafef.vn/Ajax/PageNew/FinanceData/fi.ashx?symbol={i}")
            if response.status_code == 200:
                records = response.json()
                for record in records:
                    financial_ratios.append({
                        'symbol': record.get("Symbol"),
                        'year': record.get('Year'),
                        'eps': record.get('EPS'),
                        'pe': record.get('PE'),
                    })
        except Exception as e:
            continue

    # Chuyển dữ liệu thành DataFrame
    financial_ratios = pd.DataFrame(financial_ratios)

    bucket = client.get_bucket("trunghieund")
    blob = bucket.blob('finance_ratio.csv')
    data = financial_ratios.to_csv(index=False)  # Không lưu index trong CSV
    
    # Tải lên Google Cloud Storage
    blob.upload_from_string(data, content_type="text/csv")

# Khởi tạo DAG
with DAG(
    dag_id='crawl_financial_ratios',
    start_date=datetime(2024, 12, 15),
    schedule_interval='0 0 2 4 *',  # Chạy vào ngày 2 tháng 4 hàng năm
    catchup=False,
) as dag:
    crawl_financial_ratios_task = PythonOperator(
        task_id="crawl_financial_ratios",
        python_callable=crawl_financial_ratios,
    )

    crawl_financial_ratios_task
