
import numpy as np
import pandas as pd


def add_features(df):
    df = df.copy()

    # 1. Chuyển các cột số về dạng numeric
    numeric_cols = [
        "vendor_id",
        "pickup_month",
        "pickup_dayofweek",
        "pickup_hour",
        "pickup_minute_bucket",
        "passenger_count",
        "pickup_lon_approx",
        "pickup_lat_approx",
        "dropoff_lon_approx",
        "dropoff_lat_approx",
        "straight_line_km_approx"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # 2. Mã hóa thời gian theo chu kỳ
    df["hour_sin"] = np.sin(
        2 * np.pi * df["pickup_hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["pickup_hour"] / 24
    )

    df["dow_sin"] = np.sin(
        2 * np.pi * df["pickup_dayofweek"] / 7
    )

    df["dow_cos"] = np.cos(
        2 * np.pi * df["pickup_dayofweek"] / 7
    )

    df["month_sin"] = np.sin(
        2 * np.pi * df["pickup_month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["pickup_month"] / 12
    )

    # 3. Đặc trưng giờ cao điểm
    hour = df["pickup_hour"]

    df["is_rush_hour"] = (
        ((hour >= 7) & (hour <= 9)) |
        ((hour >= 16) & (hour <= 19))
    ).astype(int)

    # 4. Đặc trưng cuối tuần
    df["is_weekend"] = (
        df["pickup_dayofweek"] >= 5
    ).astype(int)

    # 5. Phân loại buổi trong ngày
    df["time_of_day"] = np.select(
        [
            (hour >= 6) & (hour <= 11),
            (hour >= 12) & (hour <= 16),
            (hour >= 17) & (hour <= 21)
        ],
        [1, 2, 3],
        default=4
    )

    # 6. Khoảng cách tọa độ
    dlon = (
        df["dropoff_lon_approx"] -
        df["pickup_lon_approx"]
    )

    dlat = (
        df["dropoff_lat_approx"] -
        df["pickup_lat_approx"]
    )

    df["abs_dlon"] = np.abs(dlon)
    df["abs_dlat"] = np.abs(dlat)

    # Khoảng cách Manhattan xấp xỉ (km)
    df["manhattan_km_approx"] = (
        df["abs_dlat"] * 111.0 +
        df["abs_dlon"] * 85.0
    )

    # 7. Hướng di chuyển
    df["bearing"] = np.arctan2(dlon, dlat)

    # 8. Kết hợp khoảng cách và thời gian
    df["distance_x_rush"] = (
        df["straight_line_km_approx"] *
        df["is_rush_hour"]
    )

    df["distance_x_hour"] = (
        df["straight_line_km_approx"] *
        hour
    )

    # 9. Mã hóa store_and_fwd_flag
    if "store_and_fwd_flag" in df.columns:
        df["store_and_fwd_flag"] = (
            df["store_and_fwd_flag"]
            .map({
                "N": 0,
                "Y": 1,
                "n": 0,
                "y": 1,
                0: 0,
                1: 1
            })
        )

    return (
    df.replace([np.inf, -np.inf], np.nan)
      .fillna(0)
)