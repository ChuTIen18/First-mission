from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


@dataclass
class DBConfig:
    server: str
    port: int
    username: str
    password: str
    driver: str = "ODBC Driver 18 for SQL Server"
    trust_server_certificate: bool = True

    def make_url(self, database: str) -> str:
        """
        Tạo SQLAlchemy URL cho một database cụ thể.
        """
        # Ví dụ:
        # mssql+pyodbc://sa:Pass@localhost:1433/Data_2023
        #   ?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes
        base = (
            f"mssql+pyodbc://{self.username}:{self.password}"
            f"@{self.server}:{self.port}/{database}"
        )
        params = f"driver={self.driver.replace(' ', '+')}"
        if self.trust_server_certificate:
            params += "&TrustServerCertificate=yes"
        return f"{base}?{params}"


def _get_master_engine(cfg: DBConfig) -> Engine:
    """
    Engine kết nối tới database 'master' để có thể tạo DB mới.
    """
    url = cfg.make_url("master")
    # Dùng AUTOCOMMIT để có thể chạy CREATE DATABASE ngoài transaction
    return create_engine(
        url,
        fast_executemany=True,
        isolation_level="AUTOCOMMIT",
    )


def get_or_create_engine_for_year(year: int, cfg: DBConfig) -> Engine:
    """
    Đảm bảo tồn tại database Data_<year>, sau đó trả về Engine kết nối vào DB đó.
    """
    db_name = f"Data_{year}"

    # 1) Tạo DB nếu chưa có
    master_engine = _get_master_engine(cfg)
    # CREATE DATABASE không được nằm trong multi-statement transaction,
    # nên ta chạy bằng AUTOCOMMIT và không dùng transaction context.
    create_sql = f"IF DB_ID('{db_name}') IS NULL CREATE DATABASE [{db_name}]"
    with master_engine.connect() as conn:
        conn.execute(text(create_sql))

    # 2) Engine trỏ vào DB Data_<year>
    data_engine = create_engine(cfg.make_url(db_name), fast_executemany=True)
    return data_engine


def write_training_table(
    df: pd.DataFrame,
    engine: Engine,
    table_name: str,
    *,
    replace_if_exists: bool = True,
) -> None:
    """
    Ghi DataFrame training (92 cột) vào bảng SQL Server.

    - Nếu replace_if_exists=True: sẽ DROP bảng cũ (nếu có) trước khi ghi lại.
      Điều này giúp pipeline idempotent cho cùng (year, month, region).
    - Nếu replace_if_exists=False: dùng if_exists='append'.
    """
    if df.empty:
        return

    if replace_if_exists:
        drop_stmt = text(
            "IF OBJECT_ID(:tbl, 'U') IS NOT NULL "
            "BEGIN DROP TABLE [dbo]." + table_name + " END"
        )
        with engine.begin() as conn:
            conn.execute(drop_stmt, {"tbl": table_name})
        if_exists_mode = "fail"  # sau khi drop, bảng chưa tồn tại
    else:
        if_exists_mode = "append"

    # Chia chunksize để tránh lỗi giới hạn tham số
    chunksize = max(10, 2000 // max(1, len(df.columns)))

    df.to_sql(
        name=table_name,
        con=engine,
        if_exists=if_exists_mode,
        index=False,
        chunksize=chunksize,
        method="multi",
    )


def delete_month_partition(engine: Engine, table_name: str, id_month: int) -> None:
    """
    Xoá dữ liệu của một tháng (id_month) trong bảng partition theo cột id_month,
    nếu bảng đã tồn tại. Dùng để đảm bảo idempotent trên (year, month, region):
    chạy lại cùng năm/tháng sẽ thay data tháng đó thay vì append trùng.
    """
    # Không dùng tham số cho tên bảng trong SQL Server, nên chèn trực tiếp table_name
    # (table_name được code sinh ra, không phải input từ user).
    sql = text(
        f"""
        IF OBJECT_ID(:tbl, 'U') IS NOT NULL
        BEGIN
            DELETE FROM [dbo].{table_name} WHERE id_month = :m
        END
        """
    )
    with engine.begin() as conn:
        conn.execute(sql, {"tbl": table_name, "m": id_month})


