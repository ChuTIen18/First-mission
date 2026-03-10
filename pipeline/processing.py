from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import Source.utils_1 as utils_1


# Ngưỡng tách vùng địa lý (lấy theo notebook hiện tại)
# - Offshore: vùng ngoài khơi California (lat ~33.x, lon ~[-119.1, -117.9])
# - Bay: vùng vịnh (lat ~37.x, lon ~[-123.0, -121.8])
OFFSHORE_LAT_MIN = 33.2
OFFSHORE_LAT_MAX = 34.1
OFFSHORE_LON_MIN = -119.1
OFFSHORE_LON_MAX = -117.9

BAY_LAT_MIN = 37.5
BAY_LAT_MAX = 38.2
BAY_LON_MIN = -123.0
BAY_LON_MAX = -121.8

# Giới hạn số lượng tàu giống notebook: chỉ lấy top 350 MMSI mỗi vùng
TOP_MMSI_PER_REGION = 350


def split_regions_by_lat(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Tách dữ liệu thành 2 vùng: bay / offshore dựa trên LAT.
    Trả về dict: {"bay": df_bay, "offshore": df_off}.
    Các điểm nằm ngoài 2 vùng này sẽ bị loại.
    """
    if "LAT" not in df.columns or "LON" not in df.columns:
        raise ValueError("DataFrame phải có cả cột 'LAT' và 'LON' để tách vùng.")

    df = df.copy()

    mask_offshore = (
        (df["LAT"] >= OFFSHORE_LAT_MIN)
        & (df["LAT"] <= OFFSHORE_LAT_MAX)
        & (df["LON"] >= OFFSHORE_LON_MIN)
        & (df["LON"] <= OFFSHORE_LON_MAX)
    )
    mask_bay = (
        (df["LAT"] >= BAY_LAT_MIN)
        & (df["LAT"] <= BAY_LAT_MAX)
        & (df["LON"] >= BAY_LON_MIN)
        & (df["LON"] <= BAY_LON_MAX)
    )

    df_offshore = df[mask_offshore]
    df_bay = df[mask_bay]

    return {
        "offshore": df_offshore.reset_index(drop=True),
        "bay": df_bay.reset_index(drop=True),
    }


def _basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning tối thiểu để tương thích với logic trong notebook:
    - giữ lại các cột quan trọng, drop NA trên các cột đó
    - lọc SOG trong khoảng [6, 40]
    - sort theo (MMSI, BaseDateTime)
    Khoảng cách thời gian (delta_t) và các kiểm soát window sẽ
    được xử lý bên trong utils_1.build_sequence_samples_limited.
    """
    required_cols = [
        "BaseDateTime",
        "LAT",
        "LON",
        "SOG",
        "COG",
        "Heading",
        "MMSI",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu các cột bắt buộc trong raw data: {missing}")

    dfc = df.copy()
    dfc["BaseDateTime"] = pd.to_datetime(dfc["BaseDateTime"], errors="coerce")
    dfc = dfc.dropna(subset=required_cols)

    # Lọc tốc độ như trong thiết kế: 6–40
    dfc = dfc[(dfc["SOG"] >= 6.0) & (dfc["SOG"] <= 40.0)]

    dfc = dfc.sort_values(["MMSI", "BaseDateTime"])
    return dfc.reset_index(drop=True)


def build_training_dataset_for_region(
    df_region: pd.DataFrame,
    region_name: str, # Thêm tham số này để biết đang chạy cho vùng nào
    id_month: int = 0, # Tháng đang xử lý, dùng để phân biệt trong scaler JSON
    output_dir: str = ".", # Mặc định lưu scaler ở thư mục hiện tại
) -> Tuple[pd.DataFrame, dict]:
    """
    Xây dựng dataset huấn luyện (92 cột) cho một vùng (bay/offshore):
    - Cleaning cơ bản
    - Gọi utils_1.build_phase_features để tạo feature + scaler riêng
    - Gọi utils_1.build_sequence_samples_limited để cắt sliding window 10 bước

    Trả về:
    - df_train: DataFrame có 92 cột (90 feature + 2 target)
    - meta: dict chứa meta thông tin (lat_ref, lon_ref, scaler_xy, ...)
    """
    if df_region.empty:
        return df_region.copy(), {}

    # Giữ nguyên logic chọn top 350 MMSI giống các notebook:
    # tính theo dữ liệu đã lọc thô (SOG>3, vùng lat) trước khi cleaning sâu.
    counts = (
        df_region.groupby("MMSI")
        .size()
        .sort_values(ascending=False)
    )
    top_mmsi = counts.head(TOP_MMSI_PER_REGION).index
    df_region = df_region[df_region["MMSI"].isin(top_mmsi)]

    df_clean = _basic_cleaning(df_region)

    # Tạo feature + chuẩn hóa XY cho riêng vùng này (scaler riêng)
    df_feat, meta = utils_1.build_phase_features(
        df_clean,
        mmsi_col="MMSI",
        time_col="BaseDateTime",
        lat_ref=None,
        lon_ref=None,
        scaler_xy=None,
    )

    # chèn logic lưu scaler json
    scaler_info = {
        "region": region_name,
        "id_month": id_month,
        "lat_ref": meta["lat_ref"],
        "lon_ref": meta["lon_ref"],
        "mean_xm": float(meta["scaler_xy"].mean_[0]),
        "mean_ym": float(meta["scaler_xy"].mean_[1]),
        "std_xm": float(meta["scaler_xy"].scale_[0]),
        "std_ym": float(meta["scaler_xy"].scale_[1]),
    }

    # Đọc array cũ (nếu có), thêm entry mới, ghi lại
    scaler_path = f"{output_dir}/scaler_{region_name}.json"
    existing: list = []
    try:
        with open(scaler_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                existing = data
            else:
                # File cũ là single object → chuyển thành list
                existing = [data]
    except (FileNotFoundError, json.JSONDecodeError):
        existing = []

    # Loại bỏ entry trùng id_month (để idempotent khi chạy lại)
    existing = [e for e in existing if e.get("id_month") != id_month]
    existing.append(scaler_info)
    # Sắp xếp theo id_month cho dễ đọc
    existing.sort(key=lambda x: x.get("id_month", 0))

    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    print(f"    [+] Đã lưu Scaler của {region_name} (id_month={id_month}) tại: {scaler_path}")
    
    # Cắt sliding window theo đúng hàm trong utils_1
    shards = utils_1.build_sequence_samples_limited(
        df_feat,
        feature_cols=utils_1.FEATURE_INPUT,
        seq_len=10,
        stop_speed=6.0,
        max_time_gap=300.0,
        mmsi_col="MMSI",
        time_col="BaseDateTime",
        target_cols=tuple(utils_1.TARGET),  # type: ignore[arg-type]
        stride=1,
        max_sog=40.0,
        max_samples_per_group=270_000,
        max_total_groups=100,
    )

    if not shards:
        return pd.DataFrame(columns=[]), meta

    df_train = pd.concat(shards, ignore_index=True)
    return df_train, meta


