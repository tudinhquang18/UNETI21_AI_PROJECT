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

    print("Đang tải dữ liệu...")

    nrows = 150000 if fast_mode else None
    df = pd.read_csv(data_path, nrows=nrows)

    print("Kích thước dữ liệu:", df.shape)

    df = df.dropna(subset=["trip_duration"])
    df = df[df["trip_duration"] > 0].copy()

    print("Đang xử lý đặc trưng...")

    df = add_features(df)

    features = [
        col for col in df.columns
        if col not in ["trip_id", "trip_duration"]
    ]

    X = df[features].copy()
    y = df["trip_duration"].astype("float32")

    X = X.replace([np.inf, -np.inf], np.nan)

    if X.isnull().any().any():
        raise ValueError("Dữ liệu đặc trưng còn giá trị thiếu.")

    if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
        raise ValueError("Tất cả đặc trưng phải được mã hóa thành số.")

    print("Số dòng:", len(df))
    print("Số đặc trưng:", len(features))

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=42
    )

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

    for name, model in models.items():

        print(f"\nĐang huấn luyện: {name}")

        start_time = time.time()

        model.fit(X_train, y_train_log)

        pred_log = model.predict(X_val)

        pred = np.expm1(pred_log)
        pred = np.clip(pred, 1, None)

        elapsed = time.time() - start_time

        score_rmsle = rmsle(y_val, pred)

        score_mae = mean_absolute_error(
            y_val, pred
        )

        score_rmse = np.sqrt(
            mean_squared_error(y_val, pred)
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

        print(
            f"RMSLE: {score_rmsle:.5f} | "
            f"MAE: {score_mae:.2f} | "
            f"RMSE: {score_rmse:.2f}"
        )

        if score_rmsle < best_rmsle:
            best_rmsle = score_rmsle
            best_name = name
            best_model = model

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("RMSLE")

    print("\n===== BẢNG SO SÁNH MÔ HÌNH =====")
    print(df_results.to_string(index=False))

    df_results.to_csv(
        "outputs/comparison.csv",
        index=False
    )

    print("\nMô hình tốt nhất:", best_name)
    print("Validation RMSLE:", round(best_rmsle, 5))
    print("Điểm AI ước tính:", round(70 / (1 + best_rmsle), 2))

    print("\nĐang huấn luyện lại mô hình tốt nhất...")

    final_model = clone(best_model)

    final_model.fit(
        X,
        np.log1p(y)
    )

    joblib.dump(
        final_model,
        "models/UNETI21_model.joblib"
    )

    joblib.dump(
        features,
        "models/model_features.joblib"
    )

    print("\nĐã lưu models/UNETI21_model.joblib")
    print("Đã lưu models/model_features.joblib")
    print("Đã lưu outputs/comparison.csv")
    print("\nHOÀN THÀNH ĐÁNH GIÁ VÀ HUẤN LUYỆN!")


if __name__ == "__main__":
    train_models(
        "data/UNETI2026_TRAIN_CHUNG_1200000.csv",
        fast_mode=False
    )
