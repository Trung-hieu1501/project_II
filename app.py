from flask import Flask, render_template, request, session, jsonify, flash
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
import random
from flask_caching import Cache
from scipy.optimize import minimize
import json
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import os
import io
from google.cloud import storage


app = Flask(__name__)

# Cấu hình cache
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 3000})
app.secret_key = 'your-secret-key'
print("=======================1")

###############################

# Đặt biến môi trường cho key của Google Cloud
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "crawl_data/project2-441416-142ca88fe3d7.json"
# Khởi tạo client Google Cloud Storage
client = storage.Client()
def read_data(file_path):
    bucket = client.get_bucket("trunghieund")
    blob = bucket.blob(file_path)
    data = blob.download_as_text()  # Tải file dưới dạng văn bản
    df = pd.read_csv(io.StringIO(data))  # Đọc vào DataFrame
    return df

symbols_data = read_data("symbols.csv")
cache.set('symbols_data', symbols_data)

financial_data = read_data("finance_ratio.csv")
cache.set('financial_data', financial_data)

stock_data = read_data("stock_data.csv")
stock_data['day'] = pd.to_datetime(stock_data['day'], format='%Y-%m-%d')
stock_data = stock_data.sort_values(by=['symbol', 'day'])
cache.set('stock_data', stock_data)
print("=======================2")

# Hàm vẽ biểu đồ
def generate_stock_chart(df, stock_code):
    stock_data = df[df['symbol'] == stock_code]
    stock_recent_data = stock_data.tail(150)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_recent_data['day'], y=stock_recent_data['close'], mode='lines', name=stock_code))
    fig.update_layout(
        title={
            'text': f'Biểu đồ biến động {"chỉ số" if stock_code == "VNINDEX" else "giá"} - {stock_code}',
            'x': 0.5,
            'xanchor': 'center',
        },
        xaxis_title=f'{"Ngày" if stock_code != "VNINDEX" else ""}',
        yaxis_title='Giá đóng cửa',
        dragmode=False,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
    )
    return pio.to_html(fig, full_html=False)

# Route trang chủ với cache
@app.route('/')
@cache.cached()
def index():
    df = cache.get('stock_data')  # Lấy dữ liệu từ cache

    # Tạo biểu đồ VNINDEX
    vni_chart = generate_stock_chart(df, 'VNINDEX')

    # Lấy 4 mã chứng khoán ngẫu nhiên
    symbols = random.sample(df['symbol'].unique().tolist(), 4)
    stock_charts = [generate_stock_chart(df, symbol) for symbol in symbols]

    return render_template('index.html', vni_chart=vni_chart, stock_charts=stock_charts)

# Chọn danh sách mã chứng khoán
@app.route('/suggest', methods=['GET'])
def suggest():
    symbols_data = cache.get('symbols_data')  
    query = request.args.get('query', '').upper()
    print(f"Query received: {query}")  # Debug
    if query:
        suggestions = [
            {
                'symbol': row['symbol'],
                'company_name': row['company_name']  # Lấy tên công ty từ cột 'company_name'
            }
            for index, row in symbols_data.iterrows() if row['symbol'].startswith(query)
        ][:10]  # Giới hạn kết quả tối đa là 10 mã chứng khoán
        print(f"Suggestions: {suggestions}")  # Debug
    else:
        suggestions = []
    return jsonify(suggestions)

# Theo dõi mã chứng khoán
@app.route('/track', methods=['GET', 'POST'])
def track_stock():
    df = cache.get('stock_data') 
    df_symbols = cache.get('symbols_data')
    df_finance = cache.get('financial_data')

    stock_chart = "<p>Chưa có dữ liệu!</p>"
    stock_code = None
    company_name = None
    category_name = None
    table_financial = None

    if request.method == 'POST':
        stock_code = request.form.get('stock_code').upper()
        print(stock_code)
        if stock_code in df_symbols['symbol'].values:
            stock_chart = generate_stock_chart(df, stock_code)
            company_name = df_symbols[df_symbols['symbol'] == stock_code]['company_name'].values[0]     # Dữ liệu tên công ty
            category_name = df_symbols[df_symbols['symbol'] == stock_code]['category_name'].values[0]   # tên ngành hoạt động
            table_financial = df_finance[(df_finance['symbol'] == stock_code)][['year', 'capitalization', 'pe', 'eps', 'roe']]
            table_financial = table_financial[0:4]
            table_financial.rename(columns = {'year' : 'Năm', 'capitalization': 'Vốn hoán (tỷ đồng)', 'pe': 'Chỉ số PE', 'eps': 'Chỉ số EPS', 'roe': 'Chỉ số ROE'}, inplace = True)
            print(table_financial)
        else:
            stock_chart = "<p>Mã chứng khoán không tồn tại!</p>"
    

    return render_template('track.html', stock_chart=stock_chart, stock_code=stock_code, company_name = company_name, category_name= category_name, table_financial=table_financial)

# Hàm tối ưu danh mục
@app.route('/optimize_portfolio', methods=['GET', 'POST'])
def optimize_portfolio():
    df = cache.get('stock_data') 
    
    df['day'] = df['day'].dt.strftime("%Y-%m-%d")
    df_symbols = cache.get('symbols_data')
    df_finance = cache.get('financial_data')

 

    portfolio = None  # Bảng danh mục đầu tư

    if request.method == 'POST':
        stock_codes = None
        data = request.get_json()  # Lấy danh sách mã chứng khoán từ body JSON
        stock_codes = data.get('stocks', [])

        if not stock_codes:
            # Nếu không có mã chứng khoán nào, dùng danh sách mặc định và lọc các mã theo yêu cầu đặt ra
            stock_codes = df_symbols[(df_symbols['category_id'] != 0) & (df_symbols['category_id'] != 341)]['symbol'].values
            
            df_symbols = df_symbols[(df_symbols['category_id'] != 0) & (df_symbols['category_id'] != 341)][['symbol', 'category_id']]
            
            stock_codes = pd.merge(df_finance, df_symbols, how = 'inner', left_on= 'symbol', right_on='symbol')
            stock_codes.sort_values(by = ['symbol', 'year'], inplace = True)
            
            # Tỷ lệ thay đổi eps qua các năm
            stock_codes['change_eps'] = stock_codes.groupby('symbol')['eps'].transform(lambda x: x.diff() / x.shift().abs() * 100)
            stock_codes.dropna(subset=['change_eps'], inplace = True)            

            stock_codes['peg'] = stock_codes['pe'] / stock_codes['change_eps']  # Tính chỉ số PEG

            stock_codes['median_roe_symbol'] = stock_codes.groupby(['category_id', 'year'])['roe'].transform('median').round(2)     # Lấy trung vị roe theo ngành của từng năm
            stock_codes['median_peg_category'] = stock_codes.groupby(['category_id', 'year'])['peg'].transform('mean').round(2)     # Lấy trung vị peg theo ngành của từng năm

            # sử dụng các chỉ số trong năm 2023 để tính (roe cao hơn trung bình ngành và peg nhỏ hơn trung bình ngành)
            stock_codes = stock_codes[stock_codes['year'] == 2023]
            #stock_codes = stock_codes[(stock_codes['capitalization'] >= 1000) & (stock_codes['roe'] >= stock_codes['median_roe_symbol']) & (stock_codes['peg'] <= stock_codes['median_peg_category'])]
            stock_codes = stock_codes[(stock_codes['capitalization'] >= 1000) & (stock_codes['peg'] <= stock_codes['median_peg_category'])]

            stock_codes = stock_codes['symbol'].tolist()

            num_category = 5        # số lượng mã đầu ra


            # Pivot để dễ dàng xử lý theo từng mã cổ phiếu
            pivoted = df[df['day'] >= '2023-01-01']
            pivoted = pivoted.pivot(index='day', columns='symbol', values='volume')
            
            # Hàm kiểm tra chuỗi liên tiếp khối lượng = 0
            def has_long_low_streak(volume_series):
                volume_series = volume_series.fillna(0)
                low_streaks = (volume_series <= 500000).astype(int) 
                streak_count = low_streaks.groupby((low_streaks != low_streaks.shift()).cumsum()).cumsum()
                return streak_count.max() >= 20

            # Lọc các mã cổ phiếu không thỏa mãn điều kiện
            stock_codes_filter = [symbol for symbol in pivoted.columns 
                                if not has_long_low_streak(pivoted[symbol])]     


            stock_codes = [x for x in stock_codes_filter if x in stock_codes]        


        else:
            # Tính tỷ trọng giả định cho danh sách mã chứng khoán
            stock_codes = [stock_code for stock_code in stock_codes if stock_code in df_symbols['symbol'].values]
            num_category = len(stock_codes)

        df = df[df['symbol'].isin(stock_codes)]
        df = df[['symbol','day','close']]
        df = df[df['day'] >= '2023-01-01']
        df = df.pivot(index='day', columns='symbol', values='close')

        num = len(stock_codes)
        columns = df.columns
        # Fill null
        for column in columns:
            series = df[column]
            for i in range(len(series)):
                if pd.isnull(series[i]):
                    # Tìm 4 giá trị không null gần nhất
                    left = series[max(i - 4, 0): i]
                    right = series[i + 1: i + 1 + 4]
                    valid_values = left.dropna().tolist() + right.dropna().tolist()
                    
                    # Nếu có ít nhất 2 giá trị, tính trung bình
                    if len(valid_values) >= 2:
                        series[i] = np.mean(valid_values[:4])  # lấy tối đa 4 giá trị
                        
        
        ############# Tối ưu danh mục ################
        # Tính toán tỷ suất sinh lợi hàng ngày cho từng cột
        for col in columns:
            df[col] = df[col].pct_change()
            
        df.reset_index(inplace = True)

        df['year'] = df['day'].apply(lambda x: x.split('-')[0])

        yearly_mean = df.groupby('year')[columns].mean()  # Tính trung bình lợi nhuận kỳ vọng từng năm
        yearly_cov = df.groupby('year')[columns].cov()
        
        # Tính trung bình kỳ vọng các năm và ma trận hiệp phương sai
        expected_returns_annual = yearly_mean[columns].mean()
        cov_matrix_annual = yearly_cov[columns].cov()

        # expected_returns_annual = df[columns].mean()
        # cov_matrix_annual = df[columns].cov()

        risk_free_rate = 0.04 # tỷ lệ ko rủi ro (lãi suất trái phiếu chính phủ ở mức 4%)

        # Định nghĩa hàm tối ưu hóa với tỷ suất Sharpe
        def negative_sharpe_ratio(weights, returns, covariance, risk_free_rate):
            portfolio_return = np.dot(weights, returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            return -sharpe_ratio

        args = (expected_returns_annual, cov_matrix_annual, risk_free_rate)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 0.4) for _ in range(len(columns)))
        
        # Khởi tạo giá trị ban đầu và tối ưu hóa
        initial_guess = len(columns) * [1. / len(columns)]
        result = minimize(negative_sharpe_ratio, initial_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)

        print("=========================================")
        print(result.x)
        print("=========================================")
        # Lấy  mã cổ phiếu có tỷ trọng cao nhất
        top_indices = np.argsort(result.x)[-num_category:]
        top_weights = result.x[top_indices]
        top_tickers = [columns[i] for i in top_indices]
        
        # Chuẩn hóa lại trọng số cho các cổ phiếu có tỷ trọng cao nhất
        sum_weight = np.sum(top_weights)
        
        top_weights = [round(weight / sum_weight,2)*100 for weight in top_weights]

        portfolio = pd.DataFrame({
            'Mã chứng khoán': top_tickers,
            'Tỷ trọng đầu tư': top_weights
            })
        portfolio = portfolio[portfolio['Tỷ trọng đầu tư'] >0]
        portfolio['Tỷ trọng đầu tư'] = portfolio['Tỷ trọng đầu tư'].apply(lambda x: str(x) + '%')

        print("Danh mục đầu tư:", portfolio)
        return jsonify(portfolio.to_dict(orient='records'))  # Trả về dữ liệu dưới dạng JSON

    return render_template('optimize_portfolio.html', portfolio=portfolio)


# Dự báo giá chứng khoán
@app.route('/stock_price_prediction', methods=['GET', 'POST'])
def stock_price_prediction():
    df = cache.get('stock_data') 
    df_symbols = cache.get('symbols_data')
    df_finance = cache.get('financial_data')
    #models = load_model('models\model.h5')     #đọc mô hình

##############################
    # Tải file .h5 từ Google Cloud Storage
    bucket = client.get_bucket("trunghieund")
    blob = bucket.blob("model.h5")
    # Tải xuống file .h5
    blob.download_to_filename("model.h5")
    # Đọc mô hình từ file .h5
    models = tf.keras.models.load_model("model.h5")
    os.remove("model.h5")
##########################

    stock_chart = "<p>Chưa có dữ liệu!</p>"
    stock_code = None
    company_name = None
    result_evaluate = None
    result_df = None

    if request.method == 'POST':
        stock_code = request.form.get('stock_code').upper()
        print(stock_code)
        if stock_code in df_symbols['symbol'].values:
            stock_chart = generate_stock_chart(df, stock_code)
            company_name = df_symbols[df_symbols['symbol'] == stock_code]['company_name'].values[0]     # Dữ liệu tên công ty

            day = df['day'].iloc[-1]
            test_start = (pd.to_datetime(day) - timedelta(days=300)).strftime('%Y-%m-%d')
            test_df = df[(df['symbol'] == stock_code) & (df['day'] >= test_start)].sort_values('day')

            # Lấy giá đóng cửa và kiểm tra số lượng mẫu
            test_features = test_df['close'].values.reshape(-1, 1)
            scaler = MinMaxScaler()
            test_features_scaled = scaler.fit_transform(test_features)

            test_sequences = [
                test_features_scaled[i - 90:i]
                for i in range(90, len(test_features_scaled))
            ]

            X_test = np.array(test_sequences)

            # Dự đoán giá trên tập kiểm tra
            predictions_scaled = models.predict(X_test)
            predictions = scaler.inverse_transform(predictions_scaled)
            predictions = np.round(predictions, 2)
            # Lấy giá thực tế
            test_actual = test_features[90:]
            actual = test_actual.flatten()
            predicted = predictions.flatten()

            # Bảng đánh giá
            mae = mean_absolute_error(actual[1:], predicted[:-1])
            mse = mean_squared_error(actual[1:], predicted[:-1])
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((actual[1:] - predicted[:-1]) / actual[1:])) * 100

            # Dataframe giá
            result_df = pd.DataFrame({
                'Date': pd.to_datetime(test_df['day'], errors='coerce', format='%Y-%m-%d').values[90:],  # Lấy ngày từ dữ liệu kiểm tra
                'Price day t': actual,  # Giá thực tế
                'Price day t+1': predicted  # Giá dự đoán
            })

            print(result_df['Price day t+1'].head(5))
        # Biểu đồ giá
            # Dịch giá dự đoán (Price day t+1) về trục x tương ứng với thời điểm t+1
            result_df['Date t+1'] = result_df['Date'].shift(-1)
            result_df['Date t+1'] = result_df['Date t+1'].fillna((pd.to_datetime(result_df['Date'].iloc[-1]) + timedelta(days=1)).strftime('%Y-%m-%d'))

           # Tạo đồ thị
            fig = go.Figure()

            # Vẽ giá thực tế
            fig.add_trace(go.Scatter(x=result_df['Date'].shift(-1), y=result_df['Price day t'].shift(-1), mode='lines', name='Actual Price (day t)', line=dict(color='blue')))

            # Vẽ giá dự đoán
            fig.add_trace(go.Scatter(x=result_df['Date t+1'], y=result_df['Price day t+1'], mode='lines', name='Predicted Price (day t+1)', line=dict(color='orange')))

            # Thêm nhãn và tiêu đề
            fig.update_layout(
                title={
                    'text': f'Biểu đồ giá đóng cửa thực tế và dự đoán {stock_code}',
                    'x': 0.5,  # Căn giữa tiêu đề
                    'xanchor': 'center'
                },
                xaxis_title='Ngày giao dịch',
                yaxis_title='Giá',
                template='plotly',
                dragmode=False,
                xaxis=dict(fixedrange=True),  # Vô hiệu hóa thao tác kéo trên trục x
                yaxis=dict(fixedrange=True),  # Vô hiệu hóa thao tác kéo trên trục y
                legend=dict(orientation='h', yanchor='bottom',y = -0.35, xanchor='center', x=0.5),
            )

            # Lấy biểu đồ dưới dạng HTML
            stock_chart = fig.to_html(full_html=False)

            # Bảng đánh giá
            result_evaluate = pd.DataFrame({
                'Mã chứng khoán': [stock_code],  # Cần đưa vào danh sách
                'Chỉ số MAE': [np.round(mae, 2)],
                'Chỉ số RMSE': [np.round(rmse, 2)],
                'Chỉ số MAPE': [str(np.round(mape, 2)) +'%']
            })

            # Bảng giá
            result_df.drop(columns='Date t+1', inplace=True)
            result_df = result_df[-10:]
            result_df.columns =['Ngày giao dịch t', 'Giá đóng cửa ngày t', 'Giá đóng cửa dự đoán ngày t+1']

        else:
            stock_chart = "<p>Mã chứng khoán không tồn tại!</p>"
    

    return render_template('stock_price_prediction.html', stock_chart=stock_chart, stock_code=stock_code, company_name = company_name, result_evaluate = result_evaluate, result_df = result_df)

# Hàm vẽ biểu đồ
def plot_signals(signals_df, stock_code):
    # Lọc dữ liệu theo điều kiện 'position'
    buy_signals = signals_df[signals_df['signal'] == 2]
    sell_signals = signals_df[signals_df['signal'] == -1]
    
    # Tạo biểu đồ giá và dải Bollinger
    fig = make_subplots(
        rows=2, cols=1,  # Hai hàng, một cột
        shared_xaxes=True,  # Chia sẻ trục x giữa các biểu đồ
        vertical_spacing=0.005,  # Khoảng cách giữa các hàng (giảm khoảng cách giữa các biểu đồ)
        subplot_titles=(f"Biểu đồ giá đóng cửa, dải Bollinger và RSI của {stock_code}", ""),  # Tiêu đề
        row_heights=[0.7, 0.3]  # Dành 70% không gian cho biểu đồ giá và dải Bollinger, 30% cho RSI
    )

    # Biểu đồ giá và dải Bollinger
    fig.add_trace(go.Scatter(x=signals_df['day'], y=signals_df['close'], mode='lines', name='Close', line=dict(color='purple', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=signals_df['day'], y=signals_df['upper_band'], mode='lines', name='Dải Bollinger trên', line=dict(color='red', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=signals_df['day'], y=signals_df['lower_band'], mode='lines', name='Dải Bollinger dưới', line=dict(color='green', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=buy_signals['day'], y=buy_signals['close'], mode='markers', name='Điểm mua', marker=dict(color='green', symbol='triangle-up', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_signals['day'], y=sell_signals['close'], mode='markers', name='Điểm bán', marker=dict(color='red', symbol='triangle-down', size=10)), row=1, col=1)

    # Biểu đồ RSI
    fig.add_trace(go.Scatter(x=signals_df['day'], y=signals_df['rsi'], mode='lines', name='RSI', line=dict(color='blue', width=2)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", name="Quá mua (70)", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", name="Quá bán (30)", row=2, col=1)
    fig.add_trace(go.Scatter(x=buy_signals['day'], y=buy_signals['rsi'], mode='markers', name='Điểm mua', marker=dict(color='green', symbol='triangle-up', size=10)), row=2, col=1)
    fig.add_trace(go.Scatter(x=sell_signals['day'], y=sell_signals['rsi'], mode='markers', name='Điểm bán', marker=dict(color='red', symbol='triangle-down', size=10)), row=2, col=1)

    # Cập nhật layout
    fig.update_layout(
        height=800,  # Chiều cao tổng thể của biểu đồ
        dragmode=False,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.1, xanchor='center', x=0.5),
    )

    # Cập nhật trục x/y
    fig.update_xaxes(
        tickvals=signals_df['day'].iloc[::len(signals_df)//10].tolist(), 
        fixedrange=True
    )
    fig.update_yaxes(fixedrange=True)

    return pio.to_html(fig, full_html=False)
#, pio.to_html(fig2, full_html=False)


# Tín hiệu giao dịch
@app.route('/trading_signal', methods=['GET', 'POST'])
def trading_signal():
    df = cache.get('stock_data') 
    df_symbols = cache.get('symbols_data')
    # df['day'] = df['day'].dt.strftime("%Y-%m-%d")
    df = df[df['day'] >= '2023-01-01']
    models = load_model('models/model.h5')

    signal_chart = "<p>Chưa có dữ liệu!</p>"
    rsi_chart = None
    stock_code = None
    company_name = None
    category_name = None

    if request.method == 'POST':
        
        stock_code = request.form.get('stock_code').upper()
             
        if stock_code in df_symbols['symbol'].values:
            # Hàm tính RSI
            def calculate_rsi(df, period=14):
                df['price_change'] = df['close'].diff()
                df['gain'] = df['price_change'].apply(lambda x: x if x > 0 else 0)
                df['loss'] = df['price_change'].apply(lambda x: -x if x < 0 else 0)
                
                df['avg_gain'] = df['gain'].rolling(window=period, min_periods=1).mean()
                df['avg_loss'] = df['loss'].rolling(window=period, min_periods=1).mean()
                
                df['rs'] = df['avg_gain'] / df['avg_loss']
                df['rsi'] = 100 - (100 / (1 + df['rs']))
                return df[['day', 'close', 'rsi']]

            # Hàm tính Bollinger Bands
            def calculate_bollinger_bands(df, period=20, num_std=2):
                df['SMA'] = df['close'].rolling(window=period, min_periods=1).mean()
                df['std_dev'] = df['close'].rolling(window=period, min_periods=1).std()
                df['upper_band'] = df['SMA'] + (df['std_dev'] * num_std)
                df['lower_band'] = df['SMA'] - (df['std_dev'] * num_std)
                return df[['day', 'close', 'rsi', 'upper_band', 'lower_band']]

            # Hàm xác định tín hiệu giao dịch
            def generate_signals(df):
                df['signal'] = 0
                df['signal'] = np.where((df['rsi'] < 30) & (df['close'] < df['lower_band']), 2, df['signal'])
                df['signal'] = np.where((df['rsi'] >= 70) & (df['close'] >= df['upper_band']),-1, df['signal'])
                df['position'] = df['signal'].diff()
                return df
            df_signal = df[df['symbol'] == stock_code]
########################### Dự báo giá t+1
            # Chuẩn hóa dữ liệu
            predict_df = df_signal.iloc[-90:]
            scaler = MinMaxScaler(feature_range=(0, 1))
            predict_df['close_scaled'] = scaler.fit_transform(predict_df[['close']])
            
            # Lấy chuỗi thời gian gần nhất
            recent_data = predict_df['close_scaled'].values[-90:]
            recent_data = np.reshape(recent_data, (1, 90, 1))  # Định dạng (samples, sequence_length, features)
            
            # Dự đoán giá cho ngày tiếp theo
            predicted_scaled_price = models.predict(recent_data)
            predicted_price = scaler.inverse_transform(predicted_scaled_price)  # Chuyển về giá trị gốc
            predicted_price = predicted_price[0][0]

            new_row = {'day': (pd.to_datetime(df_signal['day'].iloc[-1]) + timedelta(days=1)), 'close': predicted_price}
            df_signal = pd.concat([df_signal, pd.DataFrame([new_row])], ignore_index=True)
#######################

            df_signal = calculate_rsi(df_signal)
            df_signal = calculate_bollinger_bands(df_signal)
            df_signal = generate_signals(df_signal)
           
            signal_chart= plot_signals(df_signal, stock_code)
            #signal_chart, rsi_chart = plot_signals(df_signal, stock_code)
            company_name = df_symbols[df_symbols['symbol'] == stock_code]['company_name'].values[0]     # Dữ liệu tên công ty
            category_name = df_symbols[df_symbols['symbol'] == stock_code]['category_name'].values[0]   # tên ngành hoạt động
        else:
            signal_chart = "<p>Mã chứng khoán không tồn tại!</p>"
    
    return render_template('trading_signal.html', signal_chart=signal_chart, stock_code=stock_code, company_name = company_name, category_name= category_name)




if __name__ == '__main__':

    app.run(debug=True)

