FROM apache/airflow:2.10.2-python3.12

USER root
RUN apt-get update && apt-get install -y gcc

USER airflow
RUN pip install --no-cache-dir pandas requests numpy google-cloud-bigquery db-dtypes

# Tạo thư mục dữ liệu
RUN mkdir -p /opt/airflow/data

# Đặt thư mục làm việc (tuỳ chọn)
WORKDIR /opt/airflow
