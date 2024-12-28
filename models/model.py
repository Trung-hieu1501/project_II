import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Giả sử DataFrame `data` chứa các cột 'symbol', 'day', 'close'
# Đọc dữ liệu
data = pd.read_csv('stock_data.csv')
df_symbol = pd.read_csv('symbols.csv')



symbols = df_symbol['symbol']

df = data[data['symbol'].isin(symbols)]

# Chuyển cột 'day' sang kiểu datetime và lọc các mã chứng khoán
df['day'] = pd.to_datetime(df['day'], format='%Y-%m-%d')
#df = df[df['symbol'].isin(symbols)]

# Định nghĩa khoảng thời gian
train_start = '2020-01-01'
train_end = '2024-01-01'
test_start = '2024-01-02'

df_train = df[(df['day'] >= train_start) & (df['day'] <= train_end)]
#symbols = df_train['symbol'].unique()


# Lưu scaler cho từng mã chứng khoán
scalers = {}
train_sequences = []
train_labels = []

# Tạo dữ liệu huấn luyện
for symbol in symbols:
    train_data = df[df['symbol'] == symbol].sort_values('day')
    
    # Phân tách dữ liệu huấn luyện
    #train_data = df_symbol[(df_symbol['day'] >= train_start) & (df_symbol['day'] <= train_end)]
    features = train_data['close'].values.reshape(-1, 1)
    
    # Chuẩn hóa dữ liệu
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    scalers[symbol] = scaler
    
    # Tạo sequences và labels
    for i in range(90, len(features_scaled)):
        train_sequences.append(features_scaled[i-90:i])
        train_labels.append(features_scaled[i, 0])

# Chuyển đổi dữ liệu huấn luyện thành numpy array
X_train = np.array(train_sequences)
y_train = np.array(train_labels)

# Xây dựng mô hình LSTM
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=1)
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='mean_squared_error')

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=10, batch_size=32)



model.save('model_test.h5')

result_df = []

for symbol in symbols:
    try:
        # Lọc dữ liệu kiểm tra
        test_df = df[(df['symbol'] == symbol) & (df['day'] >= test_start)].sort_values('day')

        # Kiểm tra nếu DataFrame rỗng
        if test_df.empty:
            print(f"No data for {symbol}. Skipping.")
            continue

        # Lấy giá đóng cửa và kiểm tra số lượng mẫu
        test_features = test_df['close'].values.reshape(-1, 1)
        if test_features.shape[0] == 0:
            print(f"Not enough data for {symbol}. Skipping.")
            continue

        # Chuẩn hóa dữ liệu kiểm tra
        scaler = scalers[symbol]
        try:
            test_features_scaled = scaler.transform(test_features)
        except ValueError as e:
            print(f"Scaler error for {symbol}: {e}. Skipping.")
            continue

        # Tạo chuỗi đầu vào từ dữ liệu kiểm tra
        test_sequences = [
            test_features_scaled[i - 90:i]
            for i in range(90, len(test_features_scaled))
        ]

        if len(test_sequences) == 0:
            print(f"Not enough sequences for {symbol}. Skipping.")
            continue

        X_test = np.array(test_sequences)

        # Dự đoán giá trên tập kiểm tra
        predictions_scaled = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions_scaled)

        # Lấy giá thực tế
        test_actual = test_features[90:]
        actual = test_actual.flatten()
        predicted = predictions.flatten()

        # Đảm bảo không có giá trị không hợp lệ
        if np.any(np.isnan(actual)) or np.any(np.isnan(predicted)):
            print(f"Invalid values detected for {symbol}. Skipping.")
            continue

        # Đánh giá
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        result_df.append({
            'symbol': symbol,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        })

    except Exception as e:
        print(f"Error processing {symbol}: {e}")
 
result_df = pd.DataFrame(result_df)
