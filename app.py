import os
import joblib
import hashlib
import numpy as np
import pandas as pd
import streamlit as st

from src.features import add_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "UNETI21_model.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "model_features.joblib")

st.set_page_config(
    page_title="UNETI AI | Taxi Predictor",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');

:root {
    --primary: #167D75;
    --primary-dark: #11645E;
    --background: #F1F5F4;
    --surface: #FFFFFF;
    --text: #263C3A;
    --muted: #607572;
    --border: #DCE7E3;
}

html, body, [class*="css"] {
    font-family: 'Be Vietnam Pro', sans-serif;
}

.stApp {
    background: var(--background);
    color: var(--text);
}

.block-container {
    max-width: 1250px;
    padding: 2rem 2rem 3rem;
}

.hero {
    background: linear-gradient(120deg, #174B52, #167D75, #39A18B);
    padding: 38px 42px;
    border-radius: 20px;
    margin-bottom: 28px;
    box-shadow: 0 8px 24px rgba(23, 75, 82, .12);
}

.hero-label {
    color: #BCE8D9;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 12px;
}

.hero h1 {
    color: #FFFFFF;
    font-size: 32px;
    font-weight: 800;
    margin: 0;
}

.hero p {
    color: #E4F4EF;
    font-size: 14px;
    margin-top: 12px;
    margin-bottom: 0;
}

.section-title {
    color: #263C3A;
    font-size: 23px;
    font-weight: 800;
    margin: 20px 0;
}

.card-heading {
    color: #176B64;
    font-size: 17px;
    font-weight: 800;
    padding-bottom: 14px;
    margin-bottom: 18px;
    border-bottom: 1px solid var(--border);
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 4px 15px rgba(30, 65, 60, .04);
}

.stApp label,
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stWidgetLabel"] label {
    color: #304D49 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

.stApp input {
    background-color: #FFFFFF !important;
    color: #263C3A !important;
}

div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #D5E2DE !important;
    border-radius: 9px !important;
}

div[data-baseweb="select"] * {
    color: #263C3A !important;
}

div[data-testid="stNumberInput"] input {
    background: #FFFFFF !important;
    color: #263C3A !important;
    border: 1px solid #D5E2DE !important;
    border-radius: 9px !important;
}

div[data-testid="stNumberInput"] button {
    background: #EDF5F2 !important;
    color: #176B64 !important;
    border: 1px solid #D5E2DE !important;
}

.stFormSubmitButton > button {
    width: 100%;
    min-height: 54px;
    border-radius: 11px;
    background: #167D75;
    color: white;
    border: none;
    font-size: 15px;
    font-weight: 700;
    transition: .2s;
}

.stFormSubmitButton > button:hover {
    background: #11645E;
    color: white;
}

div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 4px 14px rgba(30, 65, 60, .04);
}

div[data-testid="stMetricLabel"] {
    color: #607572;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #167D75;
    font-weight: 800;
}

.result-card {
    background: #E4F3EE;
    border: 1px solid #B9DED1;
    border-radius: 16px;
    padding: 26px;
    margin-top: 20px;
    text-align: center;
}

.result-label {
    color: #476D63;
    font-size: 14px;
    font-weight: 700;
}

.result-value {
    color: #167D75;
    font-size: 38px;
    font-weight: 800;
    margin: 12px 0;
}

.result-detail {
    color: #607572;
    font-size: 14px;
}

.footer {
    text-align: center;
    color: #81938E;
    font-size: 12px;
    margin-top: 35px;
    padding: 15px;
}

@media screen and (max-width: 768px) {
    .block-container {
        padding: 1rem 1rem 2rem;
    }

    .hero {
        padding: 25px 22px;
    }

    .hero h1 {
        font-size: 24px;
    }

    .section-title {
        font-size: 19px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        padding: 15px;
    }

    div[data-testid="stMetric"] {
        padding: 13px;
    }

    .result-value {
        font-size: 29px;
    }
}

@media screen and (max-width: 480px) {
    .block-container {
        padding-left: .75rem;
        padding-right: .75rem;
    }

    .hero {
        padding: 22px 17px;
    }

    .hero h1 {
        font-size: 21px;
    }

    .result-value {
        font-size: 26px;
    }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path, features_path, model_mtime, features_mtime):
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features


if not os.path.exists(MODEL_PATH):
    st.error("Không tìm thấy mô hình. Hãy chạy src/train.py trước.")
    st.stop()

if not os.path.exists(FEATURES_PATH):
    st.error("Không tìm thấy model_features.joblib. Hãy chạy src/train.py trước.")
    st.stop()

model, features = load_model(
    MODEL_PATH,
    FEATURES_PATH,
    os.path.getmtime(MODEL_PATH),
    os.path.getmtime(FEATURES_PATH)
)

st.markdown("""
<div class="hero">
    <div class="hero-label">UNETI AI OLYMPIC 2026</div>
    <h1>Taxi Trip Duration Predictor</h1>
    <p>Hệ thống dự đoán thời gian di chuyển của chuyến taxi bằng Machine Learning.</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">🚕 Thông tin chuyến taxi</div>',
    unsafe_allow_html=True
)

with st.form("taxi_prediction_form"):

    col1, col2 = st.columns(2, gap="large")

    with col1:
        with st.container(border=True):

            st.markdown(
                '<div class="card-heading">📋 Thông tin chuyến đi</div>',
                unsafe_allow_html=True
            )

            vendor_id = st.selectbox(
                "Đơn vị cung cấp dịch vụ (Vendor ID)",
                options=[1, 2],
                index=1
            )

            pickup_month = st.selectbox(
                "Tháng đón khách (Pickup Month)",
                options=list(range(1, 13)),
                index=2
            )

            day_options = {
                "Thứ 2": 2,
                "Thứ 3": 3,
                "Thứ 4": 4,
                "Thứ 5": 5,
                "Thứ 6": 6,
                "Thứ 7": 7,
                "Chủ nhật": 8
            }

            selected_day = st.selectbox(
                "Thứ trong tuần (Pickup Day of Week)",
                options=list(day_options.keys()),
                index=0
            )

            pickup_dayofweek = day_options[selected_day]

            pickup_hour = st.slider(
                "Giờ đón khách (Pickup Hour)",
                min_value=0,
                max_value=23,
                value=10
            )

            pickup_minute_bucket = st.selectbox(
                "Khoảng phút đón khách (Minute Bucket)",
                options=[0, 15, 30, 45],
                index=0
            )

            passenger_count = st.slider(
                "Số lượng hành khách (Passenger Count)",
                min_value=1,
                max_value=6,
                value=5
            )

    with col2:
        with st.container(border=True):

            st.markdown(
                '<div class="card-heading">📍 Vị trí và khoảng cách</div>',
                unsafe_allow_html=True
            )

            pickup_lon_approx = st.number_input(
                "Kinh độ điểm đón (Pickup Longitude)",
                value=-73.97,
                format="%.5f"
            )

            pickup_lat_approx = st.number_input(
                "Vĩ độ điểm đón (Pickup Latitude)",
                value=40.8,
                format="%.5f"
            )

            dropoff_lon_approx = st.number_input(
                "Kinh độ điểm trả (Dropoff Longitude)",
                value=-74.01,
                format="%.5f"
            )

            dropoff_lat_approx = st.number_input(
                "Vĩ độ điểm trả (Dropoff Latitude)",
                value=40.74,
                format="%.5f"
            )

            straight_line_km_approx = st.number_input(
                "Khoảng cách đường thẳng (km)",
                min_value=0.0,
                value=7.0,
                format="%.2f"
            )

            store_and_fwd_flag = st.selectbox(
                "Trạng thái lưu trữ và chuyển tiếp dữ liệu",
                options=["N", "Y"],
                index=0,
                format_func=lambda x: (
                    "N - Không lưu trữ và chuyển tiếp"
                    if x == "N"
                    else "Y - Có lưu trữ và chuyển tiếp"
                )
            )

    submitted = st.form_submit_button(
        "🔍 DỰ ĐOÁN THỜI GIAN CHUYẾN ĐI"
    )


if submitted:

    input_data = pd.DataFrame([{
        "vendor_id": vendor_id,
        "pickup_month": pickup_month,
        "pickup_dayofweek": pickup_dayofweek,
        "pickup_hour": pickup_hour,
        "pickup_minute_bucket": pickup_minute_bucket,
        "passenger_count": passenger_count,
        "pickup_lon_approx": pickup_lon_approx,
        "pickup_lat_approx": pickup_lat_approx,
        "dropoff_lon_approx": dropoff_lon_approx,
        "dropoff_lat_approx": dropoff_lat_approx,
        "straight_line_km_approx": straight_line_km_approx,
        "store_and_fwd_flag": store_and_fwd_flag
    }])

    try:
        with st.spinner("Mô hình đang phân tích dữ liệu..."):

            input_processed = add_features(input_data.copy())

            missing_features = [
                col for col in features
                if col not in input_processed.columns
            ]

            if missing_features:
                raise ValueError(
                    f"Thiếu features: {missing_features}"
                )

            input_processed = input_processed.reindex(
                columns=features
            )

            if input_processed.isnull().any().any():
                missing_columns = input_processed.columns[
                    input_processed.isnull().any()
                ].tolist()

                raise ValueError(
                    f"Dữ liệu thiếu hoặc không hợp lệ: {missing_columns}"
                )

            non_numeric = input_processed.select_dtypes(
                exclude=np.number
            ).columns.tolist()

            if non_numeric:
                raise ValueError(
                    f"Các features chưa phải kiểu số: {non_numeric}"
                )

            numeric_data = input_processed.to_numpy(dtype=np.float64)

            if not np.isfinite(numeric_data).all():
                raise ValueError(
                    "Dữ liệu đầu vào chứa giá trị không hợp lệ."
                )

            pred_log = model.predict(input_processed)
            pred_sec = float(np.expm1(pred_log[0]))

            if not np.isfinite(pred_sec):
                raise ValueError("Kết quả dự đoán không hợp lệ.")

            pred_sec = max(1.0, pred_sec)

        st.success("Dự đoán hoàn tất!")

        st.markdown(
            '<div class="section-title">📊 Kết quả dự đoán</div>',
            unsafe_allow_html=True
        )

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Thời gian (giây)",
            f"{pred_sec:,.2f} s"
        )

        metric2.metric(
            "Thời gian (phút)",
            f"{pred_sec / 60:.2f} phút"
        )

        metric3.metric(
            "Thời gian (giờ)",
            f"{pred_sec / 3600:.2f} giờ"
        )

        hours = int(pred_sec // 3600)
        minutes = int((pred_sec % 3600) // 60)
        seconds = int(pred_sec % 60)

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">
                THỜI GIAN DI CHUYỂN ƯỚC TÍNH
            </div>
            <div class="result-value">
                {hours:02d}:{minutes:02d}:{seconds:02d}
            </div>
            <div class="result-detail">
                {hours} giờ {minutes} phút {seconds} giây
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.caption(
            f"Khoảng cách chuyến đi: {straight_line_km_approx:.2f} km"
        )

        with st.expander("Kiểm tra dữ liệu và mô hình"):
            st.write("Model:", MODEL_PATH)
            st.write("Thứ đã chọn:", selected_day)
            st.write("pickup_dayofweek gửi vào model:", pickup_dayofweek)
            st.write("Features:", features)
            st.write("Dữ liệu đầu vào:")
            st.dataframe(input_data)
            st.write("Dữ liệu sau xử lý:")
            st.dataframe(input_processed)
            st.write("Giá trị dự đoán log:", float(pred_log[0]))
            st.write("Giá trị dự đoán (giây):", pred_sec)

            with open(MODEL_PATH, "rb") as model_file:
                model_hash = hashlib.md5(model_file.read()).hexdigest()

            st.write("Model MD5:", model_hash)

    except Exception as e:
        st.error(f"Không thể dự đoán: {str(e)}")


st.markdown("""
<div class="footer">
    UNETI AI OLYMPIC 2026 · MACHINE LEARNING PROJECT
</div>
""", unsafe_allow_html=True)