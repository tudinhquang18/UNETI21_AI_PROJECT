
import os
import joblib
import numpy as np
import pandas as pd

from features import add_features


MODEL_PATH = "models/UNETI21_model.joblib"
FEATURES_PATH = "models/model_features.joblib"
TEST_PATH = "data/UNETI21_TEST_3000.csv"
OUTPUT_PATH = "outputs/UNETI21_submission.csv"

TARGET_ID = "T3c808997a0b47937b7"


def generate_predictions():

    os.makedirs("outputs", exist_ok=True)

    print("Dang doc du lieu Test...")

    test_df = pd.read_csv(TEST_PATH)

    if len(test_df) != 3000:
        raise ValueError(
            f"Test phai co 3000 dong, hien tai co {len(test_df)} dong"
        )

    if "trip_id" not in test_df.columns:
        raise ValueError("Khong tim thay cot trip_id")

    if test_df["trip_id"].isnull().any():
        raise ValueError("Test co trip_id bi thieu")

    if test_df["trip_id"].duplicated().any():
        raise ValueError("Test co trip_id bi trung")

    trip_ids = test_df["trip_id"].copy()

    print("Dang tai mo hinh...")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Khong tim thay model: {MODEL_PATH}"
        )

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            f"Khong tim thay features: {FEATURES_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)

    print("Dang xu ly Feature Engineering...")

    X_test = add_features(test_df.copy())

    X_test = X_test.reindex(columns=features)

    non_numeric = X_test.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    if non_numeric:
        raise ValueError(
            f"Cac cot chua phai kieu so: {non_numeric}"
        )

    if X_test.isnull().any().any():
        missing = X_test.columns[
            X_test.isnull().any()
        ].tolist()

        raise ValueError(
            f"Du lieu co gia tri thieu tai: {missing}"
        )

    if not np.isfinite(
        X_test.to_numpy(dtype=float)
    ).all():
        raise ValueError("Du lieu dau vao co NaN hoac Inf")

    print("Feature Engineering thanh cong")
    print("Kich thuoc X_test:", X_test.shape)

    print("Dang du doan...")

    pred_log = model.predict(X_test)

    preds = np.expm1(pred_log)
    preds = np.clip(preds, 1, None)

    if not np.isfinite(preds).all():
        raise ValueError("Ket qua du doan co NaN hoac Inf")

    submission = pd.DataFrame({
        "trip_id": trip_ids,
        "trip_duration": preds
    })

    assert len(submission) == 3000
    assert submission["trip_id"].equals(trip_ids)
    assert submission["trip_id"].is_unique
    assert submission["trip_duration"].notna().all()
    assert np.isfinite(submission["trip_duration"]).all()
    assert (submission["trip_duration"] > 0).all()

    print("\nKiem tra trip_id can doi chieu...")

    check = submission[
        submission["trip_id"] == TARGET_ID
    ]

    if not check.empty:
        duration = check["trip_duration"].iloc[0]

        print("Trip ID:", TARGET_ID)
        print("Trip duration:", repr(float(duration)), "giay")
        print("So phut:", round(duration / 60, 2), "phut")
    else:
        print("Khong tim thay trip_id:", TARGET_ID)

    submission.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nHOAN THANH DU DOAN")
    print("So dong:", len(submission))
    print("So cot:", len(submission.columns))
    print("So trip_id duy nhat:", submission["trip_id"].nunique())

    print(
        "Trung binh:",
        round(submission["trip_duration"].mean(), 2),
        "giay"
    )

    print(
        "Nho nhat:",
        round(submission["trip_duration"].min(), 2),
        "giay"
    )

    print(
        "Lon nhat:",
        round(submission["trip_duration"].max(), 2),
        "giay"
    )

    print("\nDa luu file:", OUTPUT_PATH)


if __name__ == "__main__":
    generate_predictions()