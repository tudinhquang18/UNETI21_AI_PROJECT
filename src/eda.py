
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


OUTPUT_DIR = "outputs/eda_charts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.unicode_minus"] = False


def run_eda(train_path):

    print("=" * 65)
    print("KHÁM PHÁ DỮ LIỆU - UNETI AI 2026")
    print("=" * 65)

    print("\nĐANG ĐỌC DỮ LIỆU...")

    df = pd.read_csv(train_path)

    print("Đọc dữ liệu thành công!")

    print("\n1. KÍCH THƯỚC DỮ LIỆU")
    print("Số dòng:", df.shape[0])
    print("Số cột:", df.shape[1])

    print("\nDanh sách các cột:")
    print(df.columns.tolist())

    print("\n2. KIỂU DỮ LIỆU")
    print(df.dtypes)

    print("\n3. THỐNG KÊ MÔ TẢ")
    print(df.describe(include="all").T)

    print("\n4. KIỂM TRA GIÁ TRỊ THIẾU")

    missing = df.isnull().sum()

    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("Không có giá trị thiếu.")

    print("\n5. KIỂM TRA DỮ LIỆU TRÙNG LẶP")
    print("Số dòng trùng lặp:", df.duplicated().sum())

    if "trip_id" in df.columns:
        print("Số trip_id bị trùng:",
              df["trip_id"].duplicated().sum())

    print("\n6. KIỂM TRA TRIP_DURATION")

    if "trip_duration" not in df.columns:
        raise ValueError("Không tìm thấy cột trip_duration!")

    target = df["trip_duration"]

    print(target.describe(
        percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]
    ))

    print("Số giá trị thiếu:", target.isna().sum())

    print("Số giá trị không hữu hạn:",
          (~np.isfinite(target)).sum())

    print("Số chuyến có thời gian <= 0:",
          (target <= 0).sum())

    print("Số chuyến có thời gian > 10000 giây:",
          (target > 10000).sum())

    valid_df = df[
        np.isfinite(target) & (target > 0)
    ].copy()

    if valid_df.empty:
        raise ValueError("Không có dữ liệu trip_duration hợp lệ!")

    print("\n7. BIỂU ĐỒ PHÂN PHỐI THỜI GIAN")

    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=valid_df,
        x="trip_duration",
        bins=100,
        kde=True,
        color="steelblue"
    )

    plt.title("Distribution of Taxi Trip Duration")
    plt.xlabel("Trip Duration (seconds)")
    plt.ylabel("Number of Trips")
    plt.xlim(0, 10000)
    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/trip_duration_dist.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\n8. BIỂU ĐỒ KHOẢNG CÁCH VÀ THỜI GIAN")

    if "straight_line_km_approx" in valid_df.columns:

        sample = valid_df.sample(
            n=min(100000, len(valid_df)),
            random_state=42
        )

        plt.figure(figsize=(10, 6))

        plt.hexbin(
            sample["straight_line_km_approx"],
            sample["trip_duration"],
            gridsize=50,
            cmap="inferno",
            bins="log",
            mincnt=1
        )

        plt.colorbar(label="Log Count")
        plt.title("Distance vs Trip Duration")
        plt.xlabel("Straight Line Distance (km)")
        plt.ylabel("Trip Duration (seconds)")
        plt.ylim(0, 10000)
        plt.tight_layout()

        plt.savefig(
            f"{OUTPUT_DIR}/distance_vs_duration.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    else:
        print("Không tìm thấy straight_line_km_approx.")

    print("\n9. BIỂU ĐỒ THỜI GIAN THEO GIỜ ĐÓN")

    if "pickup_hour" in valid_df.columns:

        hourly_avg = (
            valid_df.groupby("pickup_hour")["trip_duration"]
            .mean()
            .reset_index()
            .sort_values("pickup_hour")
        )

        plt.figure(figsize=(10, 6))

        sns.lineplot(
            data=hourly_avg,
            x="pickup_hour",
            y="trip_duration",
            marker="o",
            color="darkorange"
        )

        plt.title("Average Trip Duration by Pickup Hour")
        plt.xlabel("Pickup Hour")
        plt.ylabel("Average Duration (seconds)")
        plt.xticks(range(24))
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        plt.savefig(
            f"{OUTPUT_DIR}/hourly_avg_duration.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    else:
        print("Không tìm thấy pickup_hour.")

    print("\n10. BA NHẬN XÉT TỪ KẾT QUẢ EDA")

    mean_duration = valid_df["trip_duration"].mean()
    median_duration = valid_df["trip_duration"].median()

    long_trip_ratio = (
        valid_df["trip_duration"].gt(10000).mean() * 100
    )

    print("\nNHẬN XÉT 1: PHÂN PHỐI THỜI GIAN")

    print(f"Thời gian trung bình: {mean_duration:.2f} giây")
    print(f"Thời gian trung vị: {median_duration:.2f} giây")
    print(f"Tỷ lệ chuyến trên 10000 giây: {long_trip_ratio:.2f}%")

    if mean_duration > median_duration:
        print("Kết luận: Phân phối thời gian có xu hướng lệch phải.")
    elif mean_duration < median_duration:
        print("Kết luận: Phân phối thời gian có xu hướng lệch trái.")
    else:
        print("Kết luận: Trung bình và trung vị bằng nhau.")

    print("\nNHẬN XÉT 2: KHOẢNG CÁCH VÀ THỜI GIAN")

    if "straight_line_km_approx" in valid_df.columns:

        correlation = valid_df[
            ["straight_line_km_approx", "trip_duration"]
        ].corr().iloc[0, 1]

        print(f"Hệ số tương quan Pearson: {correlation:.4f}")

        if correlation > 0:
            print("Kết luận: Khoảng cách có tương quan tuyến tính thuận với thời gian.")
        elif correlation < 0:
            print("Kết luận: Khoảng cách có tương quan tuyến tính nghịch với thời gian.")
        else:
            print("Kết luận: Không có tương quan tuyến tính rõ ràng.")

    print("\nNHẬN XÉT 3: THỜI GIAN THEO GIỜ ĐÓN")

    if "pickup_hour" in valid_df.columns:

        hourly_avg = valid_df.groupby(
            "pickup_hour"
        )["trip_duration"].mean()

        max_hour = hourly_avg.idxmax()
        min_hour = hourly_avg.idxmin()

        print(
            f"Giờ có thời gian trung bình cao nhất: "
            f"{max_hour}h ({hourly_avg.max():.2f} giây)"
        )

        print(
            f"Giờ có thời gian trung bình thấp nhất: "
            f"{min_hour}h ({hourly_avg.min():.2f} giây)"
        )

        print(
            "Kết luận: Thời gian chuyến đi trung bình "
            "có sự khác biệt giữa các khung giờ đón."
        )

    print("\n" + "=" * 65)
    print("HOÀN THÀNH KHÁM PHÁ DỮ LIỆU!")
    print("Biểu đồ được lưu tại:", OUTPUT_DIR)
    print("=" * 65)


if __name__ == "__main__":

    run_eda(
        "data/UNETI2026_TRAIN_CHUNG_1200000.csv"
    )

