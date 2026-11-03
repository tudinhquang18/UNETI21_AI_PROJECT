import os
import time
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from catboost import CatBoostRegressor
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from features import add_features


def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)

    return np.sqrt(
        np.mean(
            (np.log1p(y_pred) - np.log1p(y_true)) ** 2
        )
    )


def train_models(data_path, fast_mode=False):

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    print("=" * 60)
    print("UNETI AI OLYMPIC 2026 - MODEL TRAINING")
    print("=" * 60)

    print("\n[1] Đang tải dữ liệu...")

    nrows = 150000 if fast_mode else None

    df = pd.read_csv(
        data_path,
        nrows=nrows
    )

    print("Kích thước dữ liệu ban đầu:", df.shape)

    required_columns = [
        "trip_id",
        "trip_duration"
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Không tìm thấy cột {col} trong dữ liệu!")

    df["trip_duration"] = pd.to_numeric(
        df["trip_duration"],
        errors="coerce"
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=["trip_duration"]
    )

    df = df[
        df["trip_duration"] > 0
    ].copy()

    if len(df) < 100:
        raise ValueError("Không đủ dữ liệu hợp lệ để huấn luyện!")

    print("Số dòng hợp lệ:", len(df))

    print("\n[2] Đang tạo đặc trưng...")

    df = add_features(df)

    features = [
        col for col in df.columns
        if col not in ["trip_id", "trip_duration"]
    ]

    if not features:
        raise ValueError("Không tìm thấy đặc trưng đầu vào!")

    X = df[features].copy()

    y = df["trip_duration"].astype("float32")

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    if X.isnull().any().any():
        missing_columns = X.columns[
            X.isnull().any()
        ].tolist()

        raise ValueError(
            f"Các đặc trưng còn giá trị thiếu: {missing_columns}. "
            "Hãy xử lý trong features.py."
        )

    non_numeric = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    if non_numeric:
        raise ValueError(
            f"Các đặc trưng chưa được mã hóa số: {non_numeric}"
        )

    if not np.isfinite(y).all():
        raise ValueError("Nhãn trip_duration chứa giá trị không hợp lệ!")

    print("Số dòng sử dụng:", len(X))
    print("Số đặc trưng:", len(features))
    print("Danh sách đặc trưng:", features)

    print("\n[3] Đang chia Train và Validation...")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=42
    )

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)

    y_train_log = np.log1p(y_train)

    models = {
        "Baseline (DummyMean)": DummyRegressor(
            strategy="mean"
        ),

        "Linear Regression": LinearRegression(),

        "LightGBM": lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),

        "CatBoost": CatBoostRegressor(
            iterations=300,
            learning_rate=0.08,
            depth=6,
            random_seed=42,
            verbose=0,
            thread_count=-1
        )
    }

    results = []

    best_rmsle = float("inf")
    best_name = None
    best_model = None

    print("\n[4] Bắt đầu huấn luyện và đánh giá...")

    for name, model in models.items():

        print("\n" + "-" * 50)
        print("Đang huấn luyện:", name)

        start_time = time.time()

        model.fit(
            X_train,
            y_train_log
        )

        pred_log = model.predict(X_val)

        pred = np.expm1(pred_log)

        pred = np.clip(
            pred,
            1,
            None
        )

        elapsed = time.time() - start_time

        score_rmsle = rmsle(
            y_val,
            pred
        )

        score_mae = mean_absolute_error(
            y_val,
            pred
        )

        score_rmse = np.sqrt(
            mean_squared_error(
                y_val,
                pred
            )
        )

        ai_score = 70 / (1 + score_rmsle)

        results.append({
            "Model": name,
            "RMSLE": round(score_rmsle, 5),
            "MAE": round(score_mae, 2),
            "RMSE": round(score_rmse, 2),
            "Training Time (s)": round(elapsed, 2),
            "Estimated AI Score": round(ai_score, 2)
        })

        print("RMSLE:", round(score_rmsle, 5))
        print("MAE:", round(score_mae, 2))
        print("RMSE:", round(score_rmse, 2))
        print("Thời gian:", round(elapsed, 2), "giây")
        print("Điểm AI ước tính:", round(ai_score, 2))

        if score_rmsle < best_rmsle:
            best_rmsle = score_rmsle
            best_name = name
            best_model = model

    print("\n[5] Kết quả so sánh mô hình...")

    df_results = pd.DataFrame(results)

    df_results = df_results.sort_values(
        by="RMSLE",
        ascending=True
    )

    print(
        df_results.to_string(index=False)
    )

    comparison_path = "outputs/comparison.csv"

    df_results.to_csv(
        comparison_path,
        index=False
    )

    print("\nMô hình tốt nhất:", best_name)
    print("Validation RMSLE:", round(best_rmsle, 5))
    print(
        "Điểm AI ước tính:",
        round(70 / (1 + best_rmsle), 2)
    )

    print("\n[6] Đang huấn luyện lại mô hình tốt nhất trên toàn bộ Train...")

    final_model = clone(best_model)

    final_model.fit(
        X,
        np.log1p(y)
    )

    print("\n[7] Đang lưu mô hình...")

    joblib.dump(
        final_model,
        "models/UNETI21_model.joblib"
    )

    joblib.dump(
        features,
        "models/model_features.joblib"
    )

    training_info = {
        "best_model": best_name,
        "validation_rmsle": float(best_rmsle),
        "estimated_ai_score": float(70 / (1 + best_rmsle)),
        "training_rows": int(len(X)),
        "feature_count": int(len(features)),
        "fast_mode": bool(fast_mode)
    }

    joblib.dump(
        training_info,
        "models/training_info.joblib"
    )

    print("\n" + "=" * 60)
    print("HOÀN THÀNH HUẤN LUYỆN")
    print("=" * 60)

    print("Mô hình:", best_name)
    print("RMSLE Validation:", round(best_rmsle, 5))
    print("Model: models/UNETI21_model.joblib")
    print("Features: models/model_features.joblib")
    print("Training Info: models/training_info.joblib")
    print("Comparison: outputs/comparison.csv")


if __name__ == "__main__":

    train_models(
        "data/UNETI2026_TRAIN_CHUNG_1200000.csv",
        fast_mode=False
    )