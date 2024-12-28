import pandas as pd
import requests
from google.cloud import bigquery, storage
from datetime import datetime, timedelta
import time
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging
import os
import io



# Hàm crawl lịch sử giao dịch
def crawl_stock_data():
    # Khởi tạo client Google Cloud Storage
    logging.info("Bắt đầu load dữ liệu")
    client = storage.Client()
    bucket = client.get_bucket("trunghieund")
    blob = bucket.blob("stock_data.csv")
    data = blob.download_as_text()  # Tải file dưới dạng văn bản
    df = pd.read_csv(io.StringIO(data))  # Đọc vào DataFrame
    logging.info("Load dữ liệu thành công")

    symbols = df['symbol'].unique()
    #symbols = ['VNINDEX']
    df['day'] = pd.to_datetime(df['day'], format='%Y-%m-%d')
    df = df.sort_values(by=['symbol', 'day'])

    #ngày bắt đầu lấy dữ liệu
    start_date = df['day'].iloc[-3]
    #start_date = datetime.strptime(start_date, "%d/%m/%Y")
    start_date = start_date.strftime("%m/%d/%Y")
    #data['day'] = pd.to_datetime(data['day'], format='%Y-%m-%d')   # Định dạng kiểu ngày tháng
    stock_data = []
    logging.info(start_date)
    i=0

    headers_template = {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "vi,en-US;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
        "Cookie": "favorite_stocks_state=1; ASP.NET_SessionId=g5ls2el2tnjl00kzkgip3ttw"
    }

    for symbol in symbols:
        try: 
            headers = headers_template.copy()
            headers["Referer"] = f"https://cafef.vn/du-lieu/lich-su-giao-dich-{symbol}-1.chn"
            response = requests.get(f'https://s.cafef.vn/Ajax/PageNew/DataHistory/PriceHistory.ashx?Symbol={symbol}&StartDate={start_date}&EndDate=&PageIndex=1&PageSize=80', headers=headers)
            if response.status_code == 200:
                logging.info(f"============================{i}")
                logging.info(symbol)
                records = response.json().get('Data', {}).get('Data', [])
                for record in records:
                    stock_data.append({
                    'symbol': symbol,
                    'day': record.get('Ngay'),
                    'open': record.get('GiaMoCua'),
                    'close': record.get('GiaDongCua'),
                    'volume': record.get('KhoiLuongKhopLenh'),
                    'high': record.get('GiaCaoNhat'),
                    'low': record.get('GiaThapNhat')
                    })  
            time.sleep(0.5)
            i+=1
        except: 
            i+=1
            continue
    stock_data = pd.DataFrame(stock_data)
    logging.info("============================")
    logging.info(stock_data.columns)
    stock_data['day'] = pd.to_datetime(stock_data['day'], errors='coerce', format='%d/%m/%Y')
    #stock_data['day'] = pd.to_datetime(stock_data['day'], format='%d/%m/%Y')
    data = stock_data
    # Gộp dữ liệu mới vào dữ liệu cũ và sắp xếp
    data = pd.concat([df, stock_data], ignore_index=True)
    data = data.reset_index(drop = True)
    data.to_csv('/opt/airflow/data/x.csv')
    data = data.sort_values(by=['symbol', 'day', 'close'], ascending=[True, True, False])
    data.drop_duplicates(subset=['symbol', 'day'], inplace=True)
    # data.to_csv('/opt/airflow/data/x.csv')

    bucket = client.get_bucket("trunghieund")
    blob = bucket.blob('stock_data.csv')
    data = data.to_csv(index=False)  # Không lưu index trong CSV
    
    # Tải lên Google Cloud Storage
    blob.upload_from_string(data, content_type="text/csv")



# Khởi tạo DAG
with DAG(
    dag_id='crawl_stock_data',
    start_date=datetime(2024, 12, 15),
    schedule_interval='@daily',
) as dag:
    # Task thu thập dữ liệu
    crawl_stock_data_HNX_task = PythonOperator(
        task_id="crawl_stock_data",
        python_callable=crawl_stock_data,
    )

    # Định nghĩa thứ tự thực thi
    crawl_stock_data_HNX_task 
    #>> crawl_stock_data_HSX_task >> crawl_stock_data_UpCOM_task
