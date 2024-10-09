# Import các thư viện cần thiết
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import streamlit as st

# Đọc và tiền xử lý dữ liệu
@st.cache_data
def load_data():
    file_path = "Restaurant_revenue.csv"  # Thay bằng đường dẫn tới file của bạn
    data = pd.read_csv(file_path)
    
    # Mã hóa biến phân loại 'Cuisine_Type'
    encoder = OneHotEncoder(sparse_output=False)
    cuisine_encoded = encoder.fit_transform(data[['Cuisine_Type']])
    cuisine_encoded_df = pd.DataFrame(cuisine_encoded, columns=encoder.get_feature_names_out(['Cuisine_Type']))
    data_encoded = pd.concat([data.drop(columns=['Cuisine_Type']), cuisine_encoded_df], axis=1)
    
    # Chuẩn hóa các cột số (trừ cột mục tiêu 'Monthly_Revenue')
    scaler = MinMaxScaler()
    numerical_columns = data_encoded.columns.difference(['Monthly_Revenue'])
    data_encoded[numerical_columns] = scaler.fit_transform(data_encoded[numerical_columns])
    
    return data_encoded, encoder, numerical_columns, scaler, data['Cuisine_Type'].unique().tolist()

data_encoded, encoder, numerical_columns, scaler, cuisine_types = load_data()

# Tách dữ liệu thành Train Set và Test Set
X = data_encoded.drop(columns=['Monthly_Revenue'])
y = data_encoded['Monthly_Revenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện các mô hình
def train_models():
    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # 2. Lasso Regression
    lasso = Lasso(random_state=42, alpha=0.1)  # Sử dụng giá trị alpha đã biết
    lasso.fit(X_train, y_train)

    # 3. Neural Network (MLP)
    nn = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    nn.fit(X_train, y_train)

    # 4. Stacking Model
    stacking = StackingRegressor(
        estimators=[
            ('linear_regression', lr),
            ('lasso_best', lasso),
            ('neural_network', nn)
        ],
        final_estimator=LinearRegression()
    )
    stacking.fit(X_train, y_train)
    
    return {
        'Linear Regression': lr,
        'Lasso Regression': lasso,
        'Neural Network': nn,
        'Stacking': stacking
    }

models = train_models()

# Giao diện người dùng để nhập dữ liệu mới
st.title("Dự đoán Doanh thu Nhà hàng")
st.write("Nhập các thông tin sau để dự đoán doanh thu:")

# Nhập dữ liệu cho các cột số
input_data = {}
for i, column in enumerate(X.columns):
    if 'Cuisine_Type' not in column:  # Bỏ qua các cột mã hóa 'Cuisine_Type'
        input_data[column] = st.number_input(f'{column}', value=0.0, key=f'{column}_{i}')

# Nhập dữ liệu cho cột 'Cuisine_Type'
selected_cuisine_type = st.selectbox("Chọn loại Cuisine Type", cuisine_types)
# Đặt tất cả các loại cuisine khác là 0
for cuisine in cuisine_types:
    input_data[f'Cuisine_Type_{cuisine}'] = 0

# Đặt loại cuisine đã chọn là 1
input_data[f'Cuisine_Type_{selected_cuisine_type}'] = 1

# Chọn mô hình trước khi dự đoán
selected_model_name = st.selectbox("Chọn mô hình muốn sử dụng", models.keys())
selected_model = models[selected_model_name]

# Nút dự đoán
if st.button("Dự đoán"):
    # Tạo DataFrame từ dữ liệu nhập
    input_df = pd.DataFrame([input_data])

    # Đảm bảo `input_df` có cùng cột và thứ tự với `X_train`
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    
    # Chuẩn hóa dữ liệu đầu vào
    input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])
    
    # Dự đoán cho mô hình đã chọn
    prediction = selected_model.predict(input_df)[0]

    # Hiển thị kết quả dự đoán
    st.write(f"Kết quả dự đoán doanh thu của nhà hàng: {prediction:.2f}")

    # Tính toán báo cáo hồi quy (Regression Report)
    y_pred = selected_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f"Báo cáo hồi quy của mô hình {selected_model_name}:")
    st.write(f"MAE: {mae:.2f}")
    st.write(f"MSE: {mse:.2f}")
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"R2 Score: {r2:.2f}")

    # Vẽ biểu đồ Dự đoán vs. Thực tế cho mô hình đã chọn
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0), alpha=0.7)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)  # Đường y = x
    ax.set_xlabel('Giá trị thực tế')
    ax.set_ylabel('Giá trị dự đoán')
    ax.set_title(f'{selected_model_name} - Thực tế vs. Dự đoán')
    st.pyplot(fig)

    # Vẽ Learning Curve cho mô hình đã chọn
    train_sizes, train_scores, test_scores = learning_curve(
        selected_model, X_train, y_train, cv=5, n_jobs=-1, scoring='neg_mean_squared_error',
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
    )

    train_scores_mean = np.mean(-train_scores, axis=1)
    test_scores_mean = np.mean(-test_scores, axis=1)

    fig, ax = plt.subplots()
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    ax.set_title(f'Learning Curve - {selected_model_name}')
    ax.set_xlabel('Number of Training Examples')
    ax.set_ylabel('Mean Squared Error')
    ax.legend(loc="best")
    st.pyplot(fig)
