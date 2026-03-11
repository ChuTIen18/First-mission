from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

from pipeline.processing import (
    split_regions_by_lat,
    build_training_dataset_for_region,
    OFFSHORE_LAT_MIN,
    OFFSHORE_LAT_MAX,
    BAY_LAT_MIN,
    BAY_LAT_MAX,
)
from pipeline.db_utils import (
    DBConfig,
    get_or_create_engine_for_year,
    write_training_table,
    delete_month_partition,
)


# Thư mục chứa các file raw .parquet, bạn chỉ cần copy file vào đây
RAW_DATA_DIR = Path("raw_data")

# Cấu hình kết nối SQL Server.
# TODO: sửa username/password/server cho đúng môi trường của bạn.
DB_CONFIG = DBConfig(
    server="localhost",
    port=1433,
    username="sa",
    password="123Strong!",  # thay bằng mật khẩu thật trong môi trường của bạn
    driver="ODBC Driver 18 for SQL Server",
    trust_server_certificate=True,
)


FILE_PATTERN = re.compile(r"^(?P<year>\d{4})_NOAA_AIS_logs_(?P<month>\d{2})\.parquet$")


def discover_raw_files(directory: Path) -> list[Path]:
    """
    Tìm tất cả file .parquet hợp lệ trong thư mục raw_data.
    Chỉ nhận những file đúng pattern YYYY_NOAA_AIS_logs_MM.parquet.
    """
    if not directory.exists():
        return []

    files = []
    for p in directory.glob("*.parquet"):
        if FILE_PATTERN.match(p.name):
            files.append(p)
    return sorted(files)


def parse_year_month_from_name(filename: str) -> tuple[int, int]:
    m = FILE_PATTERN.match(filename)
    if not m:
        raise ValueError(
            f"Tên file không đúng pattern YYYY_NOAA_AIS_logs_MM.parquet: {filename}"
        )
    year = int(m.group("year"))
    month = int(m.group("month"))
    return year, month


def load_raw_subset(path: Path) -> pd.DataFrame:
    """
    Đọc file parquet lớn bằng DuckDB và chỉ lấy subset cần dùng
    để tránh tràn RAM:
    - Chỉ các cột cần thiết cho pipeline
    - Chỉ 2 dải LAT: offshore + bay
    - Lọc thêm SOG > 3, góc hợp lệ (0–360)
    """
    con = duckdb.connect()

    query = f"""
    SELECT
        BaseDateTime,
        LAT,
        LON,
        SOG,
        COG,
        Heading,
        MMSI
    FROM '{path.as_posix()}'
    WHERE
        SOG > 3
        AND COG >= 0 AND COG < 360
        AND Heading >= 0 AND Heading < 360
        AND (
            (LAT BETWEEN {OFFSHORE_LAT_MIN} AND {OFFSHORE_LAT_MAX})
            OR
            (LAT BETWEEN {BAY_LAT_MIN} AND {BAY_LAT_MAX})
        )
    """
    df = con.execute(query).df()
    con.close()
    return df


def process_single_file(path: Path, db_config: DBConfig) -> None:
    """
    Xử lý một file raw .parquet:
    - đọc dữ liệu
    - tách bay/offshore
    - feature engineering + sliding window (92 cột)
    - ghi vào đúng database + bảng theo năm/tháng/vùng.
    """
    year, month = parse_year_month_from_name(path.name)
    print(f"\n=== Processing file: {path.name} (year={year}, month={month}) ===")

    # 1) Load raw parquet (subset để tránh tràn RAM)
    print("  - Loading raw parquet (filtered subset via DuckDB)...")
    df_raw = load_raw_subset(path)
    print(f"    Raw shape: {df_raw.shape}")

    # 2) Split thành 2 vùng: bay / offshore
    regions = split_regions_by_lat(df_raw)

    for region_name, df_region in regions.items():
        if df_region.empty:
            print(f"  - Region {region_name}: empty, skip.")
            continue

        print(f"  - Region {region_name}: input rows = {len(df_region)}")

        # 3) Xử lý đặc trưng + sliding window để ra dataset 92 cột
        df_train, meta = build_training_dataset_for_region(df_region)
        print(f"    -> Training dataset shape: {df_train.shape}")

        # 4) Ghi vào SQL Server
        # Thiết kế: mỗi DB năm có 2 bảng cố định:
        #   - training_offshore
        #   - training_bay
        # và thêm cột id_month để phân biệt tháng.
        engine = get_or_create_engine_for_year(year, db_config)

        table_name = f"training_{region_name}"  # không encode tháng trong tên bảng

        # Gắn thêm cột id_month để truy vấn theo tháng sau này
        df_to_write = df_train.copy()
        df_to_write["id_month"] = month

        # Idempotent theo tháng: xoá dữ liệu cũ của tháng này rồi ghi lại
        delete_month_partition(engine, table_name, month)

        print(f"    -> Writing to table: {table_name} (id_month={month}) in DB Data_{year}")

        write_training_table(
            df=df_to_write,
            engine=engine,
            table_name=table_name,
            replace_if_exists=False,  # append các tháng khác, tháng này đã xoá trước
        )

    print(f"=== Done file: {path.name} ===")


def main(files: Iterable[Path] | None = None) -> None:
    """
    Điểm vào chính của pipeline.
    - Nếu `files` None: tự động quét folder raw_data.
    - Nếu truyền list file cụ thể: chỉ xử lý các file đó.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if files is None:
        files_to_process = discover_raw_files(RAW_DATA_DIR)
    else:
        files_to_process = list(files)

    if not files_to_process:
        print(f"Không tìm thấy file .parquet nào trong {RAW_DATA_DIR.resolve()}")
        return

    print("Sẽ xử lý các file sau:")
    for p in files_to_process:
        print(f"  - {p.name}")

    for p in files_to_process:
        process_single_file(p, DB_CONFIG)

    print("\nHoàn thành toàn bộ pipeline.")


if __name__ == "__main__":
    main()

