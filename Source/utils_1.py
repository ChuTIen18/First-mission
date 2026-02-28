# utils.py
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler
from typing import Iterable
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import folium
from pyproj import Transformer
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings("ignore")

# đặc trưng đầu vào
FEATURE_INPUT = [
    "SOG",
    "Heading_sin","Heading_cos",
    "COG_sin","COG_cos",
    "X_norm","Y_norm",
    "hour_sin","hour_cos",
]

# biến mục tiêu
TARGET = ["X_norm","Y_norm"]

# biến đầu vào
x = ['SOG_t0','Heading_sin_t0','Heading_cos_t0','COG_sin_t0','COG_cos_t0','X_norm_t0','Y_norm_t0','hour_sin_t0','hour_cos_t0',
 'SOG_t1','Heading_sin_t1','Heading_cos_t1','COG_sin_t1','COG_cos_t1','X_norm_t1','Y_norm_t1','hour_sin_t1','hour_cos_t1',
 'SOG_t2','Heading_sin_t2','Heading_cos_t2','COG_sin_t2','COG_cos_t2','X_norm_t2','Y_norm_t2','hour_sin_t2','hour_cos_t2',
 'SOG_t3','Heading_sin_t3','Heading_cos_t3','COG_sin_t3','COG_cos_t3','X_norm_t3','Y_norm_t3','hour_sin_t3','hour_cos_t3',
 'SOG_t4','Heading_sin_t4','Heading_cos_t4','COG_sin_t4','COG_cos_t4','X_norm_t4','Y_norm_t4','hour_sin_t4','hour_cos_t4',
 'SOG_t5','Heading_sin_t5','Heading_cos_t5','COG_sin_t5','COG_cos_t5','X_norm_t5','Y_norm_t5','hour_sin_t5','hour_cos_t5',
 'SOG_t6','Heading_sin_t6','Heading_cos_t6','COG_sin_t6','COG_cos_t6','X_norm_t6','Y_norm_t6','hour_sin_t6','hour_cos_t6',
 'SOG_t7','Heading_sin_t7','Heading_cos_t7','COG_sin_t7','COG_cos_t7','X_norm_t7','Y_norm_t7','hour_sin_t7','hour_cos_t7',
 'SOG_t8','Heading_sin_t8','Heading_cos_t8','COG_sin_t8','COG_cos_t8','X_norm_t8','Y_norm_t8','hour_sin_t8','hour_cos_t8',
 'SOG_t9','Heading_sin_t9','Heading_cos_t9','COG_sin_t9','COG_cos_t9','X_norm_t9','Y_norm_t9','hour_sin_t9','hour_cos_t9']

# hàm xử lí góc và thời gian

def angle_to_sin_cos(angle_deg: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Chuyển góc (độ) -> (sin, cos)."""
    ang = (angle_deg.astype(float) % 360.0) * np.pi / 180.0
    return np.sin(ang.values), np.cos(ang.values) #type: ignore 

def add_heading_cog_sin_cos(df: pd.DataFrame) -> pd.DataFrame:
    df["Heading_sin"], df["Heading_cos"] = angle_to_sin_cos(df["Heading"])
    df["COG_sin"], df["COG_cos"] = angle_to_sin_cos(df["COG"])
    return df

def add_hour_sin_cos(df: pd.DataFrame, time_col: str = "BaseDateTime") -> pd.DataFrame:
    dt = pd.to_datetime(df[time_col], errors="coerce")
    h = dt.dt.hour + dt.dt.minute/60.0 + dt.dt.second/3600.0
    df["hour_sin"] = np.sin(2*np.pi*h/24.0)
    df["hour_cos"] = np.cos(2*np.pi*h/24.0)
    return df

def equirect_xy_m(lat0: float, lon0: float,
                  lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R = 6371000.0
    lat0r = np.deg2rad(lat0)
    x = (np.deg2rad(lon) - np.deg2rad(lon0)) * np.cos(lat0r) * R
    y = (np.deg2rad(lat)  - np.deg2rad(lat0)) * R
    return x, y

def add_xy_m(df: pd.DataFrame,
             lat_ref: Optional[float] = None,
             lon_ref: Optional[float] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if lat_ref is None:
        lat_ref = float(df["LAT"].median())
    if lon_ref is None:
        lon_ref = float(df["LON"].median())
    X_m, Y_m = equirect_xy_m(lat_ref, lon_ref,
                             df["LAT"].to_numpy(), df["LON"].to_numpy())
    df["X_m"], df["Y_m"] = X_m, Y_m
    return df, {"lat_ref": lat_ref, "lon_ref": lon_ref}

def fit_xy_scaler(df_train: pd.DataFrame) -> StandardScaler:
    sc = StandardScaler()
    sc.fit(df_train[["X_m","Y_m"]].to_numpy())
    return sc

def apply_xy_scaler(df: pd.DataFrame, sc: StandardScaler,
                    out_cols: Tuple[str,str] = ("X_norm","Y_norm")) -> pd.DataFrame:
    xy = sc.transform(df[["X_m","Y_m"]].to_numpy())
    df[out_cols[0]], df[out_cols[1]] = xy[:,0], xy[:,1]
    return df

# cấu hình tất cả vào một hàm, tạo các đặc trưng cần thiết. 
def build_phase_features(df_all: pd.DataFrame,
                         mmsi_col: str = "MMSI_copy",
                         time_col: str = "BaseDateTime",
                         lat_ref: Optional[float] = None,
                         lon_ref: Optional[float] = None,
                         scaler_xy: Optional[StandardScaler] = None
                         ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Tối giản: chỉ tạo các feature input và target,
    """
    df = df_all.copy().sort_values([mmsi_col,time_col])
    df = add_heading_cog_sin_cos(df)
    df = add_hour_sin_cos(df, time_col)

    # tọa độ phẳng + chuẩn hóa
    df, meta = add_xy_m(df, lat_ref, lon_ref)
    if scaler_xy is None:
        scaler_xy = fit_xy_scaler(df)
    df = apply_xy_scaler(df, scaler_xy)

    meta.update({"scaler_xy":scaler_xy})
    return df, meta


# tạo window slide để  tạo data train.
# chia shard tránh tràn RAM.
def build_sequence_samples_limited(
    df: pd.DataFrame,
    feature_cols: list,
    seq_len: int = 10,
    stop_speed: float = 6.0,
    max_time_gap: float = 360.0,
    *,
    mmsi_col: str = "MMSI",
    time_col: str = "BaseDateTime",
    target_cols: tuple[str, str] = ("X_norm", "Y_norm"),
    stride: int = 1,
    min_time_gap: float = 1.0,
    max_sog: float | None = None,
    max_windows_per_mmsi: int | None = None,
    max_samples_per_group: int = 500_000,
    max_total_groups: int = 3,
) -> list[pd.DataFrame]:
    """
    Cắt AIS thành các mẫu (t0..t{L-1} -> tL).
    Có chia shard theo max_samples_per_group.
    """
    import numpy as np
    import pandas as pd

    if mmsi_col not in df.columns:
        if "MMSI_copy" in df.columns:
            mmsi_col = "MMSI_copy"
        else:
            raise ValueError(f"Không tìm thấy cột MMSI: '{mmsi_col}' hoặc 'MMSI_copy'")

    dfx = df.copy()
    dfx[time_col] = pd.to_datetime(dfx[time_col], errors="coerce")
    dfx = dfx.dropna(subset=[time_col]).sort_values([mmsi_col, time_col])

    groups: list[pd.DataFrame] = []
    current_rows: list[np.ndarray] = []
    current_count = 0

    # tên cột sequence
    feature_names = [f"{col}_t{t}" for t in range(seq_len) for col in feature_cols]
    out_cols = feature_names + list(target_cols)

    for mmsi, g in dfx.groupby(mmsi_col, sort=False):
        gm = g[g["SOG"] > float(stop_speed)]
        if max_sog is not None:
            gm = gm[gm["SOG"] <= float(max_sog)]
        if len(gm) < seq_len + 1:
            continue

        dt = gm[time_col].diff().dt.total_seconds().fillna(np.inf).to_numpy()
        used_for_mmsi = 0
        limit_for_mmsi = max_windows_per_mmsi if max_windows_per_mmsi is not None else np.inf

        for i in range(0, len(gm) - (seq_len + 1) + 1, max(1, int(stride))):
            if used_for_mmsi >= limit_for_mmsi:
                break
            j = i + seq_len
            win_dt = dt[i + 1 : j + 1]
            if win_dt.size != seq_len:
                continue
            if np.any(win_dt <= float(min_time_gap)) or np.any(win_dt > float(max_time_gap)):
                continue

            block_feats = gm.iloc[i : i + seq_len][feature_cols].to_numpy(dtype=np.float32)
            block_tgt   = gm.iloc[j][list(target_cols)].to_numpy(dtype=np.float32)

            if not np.isfinite(block_feats).all() or not np.isfinite(block_tgt).all():
                continue

            row = np.concatenate([block_feats.ravel(), block_tgt])
            current_rows.append(row.astype(np.float32))
            current_count += 1
            used_for_mmsi += 1

            # shard theo RAM
            if current_count >= int(max_samples_per_group):
                groups.append(pd.DataFrame(current_rows, columns=out_cols))
                current_rows = []
                current_count = 0
                if len(groups) >= int(max_total_groups):
                    return groups

    if current_rows:
        groups.append(pd.DataFrame(current_rows, columns=out_cols))

    return groups

class ShipLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attn = nn.Linear(hidden_size, 1)
        self.fc   = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        w = torch.softmax(self.attn(out), dim=1)
        ctx = (w * out).sum(dim=1)
        return self.fc(ctx)


def predict(df_input, model_path, batch_size=1024, seq_len=10, X_cols=None, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # keep original column order (must match train order)
    if X_cols is None:
        X_cols = x
    if len(X_cols) % seq_len != 0:
        raise ValueError(f"len(X_cols)={len(X_cols)} not divisible by seq_len={seq_len}")

    input_size = len(X_cols) // seq_len

    X_np = df_input[X_cols].to_numpy(dtype=np.float32).reshape(-1, seq_len, input_size)

    model = ShipLSTM(input_size=input_size).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state)
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            xb = torch.from_numpy(X_np[i:i+batch_size]).to(device, non_blocking=True)
            yb = model(xb)
            preds.append(yb.detach().cpu())
    return torch.cat(preds, dim=0).numpy()

def predict_to_df(df_input, model_path="best_model.pt", batch_size=1024, seq_len=10, X_cols=None,
                  out_cols=("pred_X_norm","pred_Y_norm"), device=None):
    preds = predict(df_input, model_path, batch_size, seq_len, X_cols, device)
    df_out = df_input.copy()
    df_out[out_cols[0]] = preds[:,0]
    df_out[out_cols[1]] = preds[:,1]
    return df_out

def predict_point(df_row, model, SEQ_LEN=10, device='cpu'):
    X_cols = x
    X = df_row[X_cols].values.reshape(1, SEQ_LEN, -1)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(X_tensor).cpu().numpy()
    return tuple(pred.squeeze())  # (X_norm, Y_norm)

def denormalize_and_convert(norm_points, scaler, meta):
    """
    Chuyển từ (X_norm,Y_norm) -> lat/lon
    dùng lại equirectangular projection với lat_ref, lon_ref trong meta.
    """
    # 1. Denormalize về mét
    xy_array = scaler.inverse_transform(np.array(norm_points))  # (N,2)
    x_m, y_m = xy_array[:,0], xy_array[:,1]

    # 2. Lấy gốc tham chiếu
    lat0 = meta["lat_ref"]
    lon0 = meta["lon_ref"]
    R = 6371000.0
    lat0r = np.deg2rad(lat0)

    # 3. Chuyển ngược equirectangular -> lat/lon
    lat = np.rad2deg(y_m / R + np.deg2rad(lat0))
    lon = np.rad2deg(x_m / (R * np.cos(lat0r)) + np.deg2rad(lon0))

    return list(zip(lat, lon))  # [(lat, lon), ...]


def plot_ship_trajectory(df, idx, model_path, scaler, meta, device='cpu', zoom_start=11):
    # Load model
    SEQ_LEN = 10
    X_cols = x
    input_size = len(X_cols) // SEQ_LEN
    model = ShipLSTM(input_size=input_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Lấy sample
    row = df.iloc[idx]

    # Các điểm dữ liệu xanh (t1..t9)
    coords = [(float(row[f'X_norm_t{i}']), float(row[f'Y_norm_t{i}'])) for i in range(SEQ_LEN)]

    # Điểm dự đoán đỏ (t+1)
    pred_xy = predict_point(row, model, SEQ_LEN=SEQ_LEN, device=device)
    coords.append(pred_xy)

    # Đổi sang lat/lon
    lonlat_points = denormalize_and_convert(coords, scaler, meta)
    latlon_points = lonlat_points

    # Tạo map
    center = [np.mean([lat for lat, _ in latlon_points]),
              np.mean([lon for _, lon in latlon_points])]
    m = folium.Map(location=center, zoom_start=zoom_start) #type: ignore 

    for i, (lat, lon) in enumerate(latlon_points):
        if i == len(latlon_points) - 1:
            color, popup = 'red', 'Prediction'
        else:
            color, popup = 'blue', f'Point {i+1}'
        folium.CircleMarker(
            location=(lat, lon),
            radius=5,
            color=color, fill=True, fill_color=color, popup=popup
        ).add_to(m)

    folium.PolyLine(latlon_points, color="green", weight=2.5).add_to(m)
    return m


# QUAN TRỌNG
# ----------- các phương pháp và chỉ số đánh giá--------------
def to_xy_m(xy_norm: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Inverse-normalize XY (last-dim=2) from StandardScaler back to meters."""
    arr = np.asarray(xy_norm)
    if arr.shape[-1] != 2:
        raise ValueError("xy_norm must have last dimension = 2")
    flat = arr.reshape(-1, 2)
    xy_m = scaler.inverse_transform(flat)
    return xy_m.reshape(arr.shape)

def pairwise_l2_m(pred_m: np.ndarray, true_m: np.ndarray) -> np.ndarray:
    """Euclidean distance(s) in meters between pred and true (supports (...,2))."""
    diff = np.asarray(pred_m) - np.asarray(true_m)
    return np.linalg.norm(diff, axis=-1)

def compute_point_metrics_norm(pred_norm: np.ndarray,
                               true_norm: np.ndarray,
                               scaler: StandardScaler,
                               hit_threshold_m: float = 100.0) -> Dict[str, float]:
    """Point-wise metrics (RMSE/MAE/Median/P90/Hit@R) for single-step (N,2) given scaler."""
    pred_m = to_xy_m(pred_norm, scaler)
    true_m = to_xy_m(true_norm, scaler)
    errs = pairwise_l2_m(pred_m, true_m)                # (N,)
    mse_vec_m2 = np.mean(errs**2)                       # mean of squared L2
    rmse_m = float(np.sqrt(mse_vec_m2))
    out = {
        "rmse_m": rmse_m,
        "mae_m": float(np.mean(errs)),
        "median_m": float(np.median(errs)),
        "p90_m": float(np.percentile(errs, 90)),
        "p95_m": float(np.percentile(errs, 95)),
        "mse_vec_m2": float(mse_vec_m2),
        "hit@{}_m".format(int(hit_threshold_m)): float(np.mean(errs <= hit_threshold_m)),
        "n": float(errs.size),
    }
    return out

def compute_seq_metrics_norm(pred_seq_norm: np.ndarray,
                             true_seq_norm: np.ndarray,
                             scaler: StandardScaler,
                             hit_threshold_m: float = 100.0) -> Dict[str, Any]:
    """Sequence metrics (ADE/FDE/per-step RMSE/Hit@R) for multi-step (N,T,2) given scaler."""
    pred_m = to_xy_m(pred_seq_norm, scaler)  # (N,T,2)
    true_m = to_xy_m(true_seq_norm, scaler)
    errs = pairwise_l2_m(pred_m, true_m)     # (N,T)
    ade = float(np.mean(errs))
    fde = float(np.mean(errs[:, -1]))
    # per-step RMSE over L2 distances
    per_step_rmse = np.sqrt(np.mean(errs**2, axis=0))   # (T,)
    hit = float(np.mean(errs <= hit_threshold_m))
    return {
        "ADE_m": ade,
        "FDE_m": fde,
        "per_step_RMSE_m": per_step_rmse,
        "hit@{}_m".format(int(hit_threshold_m)): hit,
        "n": float(errs.shape[0]),
        "T": float(errs.shape[1]),
    }

def speed_consistency_from_norm(last_obs_norm: np.ndarray,
                                next_pred_norm: np.ndarray,
                                next_true_norm: np.ndarray,
                                delta_t_sec: np.ndarray,
                                scaler: StandardScaler) -> Dict[str, float]:
    """Speed consistency (m/s) using last observed XY, predicted next, true next, and Δt per sample."""
    last_m = to_xy_m(last_obs_norm, scaler)
    pred_m = to_xy_m(next_pred_norm, scaler)
    true_m = to_xy_m(next_true_norm, scaler)

    d_pred = np.linalg.norm(pred_m - last_m, axis=-1)
    d_true = np.linalg.norm(true_m - last_m, axis=-1)

    dt = np.asarray(delta_t_sec).astype(float)
    dt = np.where(dt <= 0, np.nan, dt)

    v_pred = d_pred / dt
    v_true = d_true / dt
    valid = np.isfinite(v_pred) & np.isfinite(v_true)

    if not np.any(valid):
        return {"speed_mae_mps": float("nan"),
                "speed_rmse_mps": float("nan"),
                "pred_speed_mean_mps": float("nan"),
                "true_speed_mean_mps": float("nan"),
                "n": 0.0}

    dv = v_pred[valid] - v_true[valid]
    return {
        "speed_mae_mps": float(np.mean(np.abs(dv))),
        "speed_rmse_mps": float(np.sqrt(np.mean(dv**2))),
        "pred_speed_mean_mps": float(np.mean(v_pred[valid])),
        "true_speed_mean_mps": float(np.mean(v_true[valid])),
        "n": float(valid.sum()),
    }

def bin_stats(errors_m: np.ndarray,
              by_values: Iterable[float],
              bins: Iterable[float],
              labels: Iterable[str] | None = None) -> pd.DataFrame:
    """Aggregate mean/median/p90 per bin of 'by_values' for given distance errors (m)."""
    s_err = pd.Series(np.asarray(errors_m))
    s_by = pd.Series(np.asarray(by_values))
    cat = pd.cut(s_by, bins=bins, labels=labels, include_lowest=True) #type: ignore
    df = pd.DataFrame({"err": s_err, "bin": cat}).dropna()
    agg = df.groupby("bin")["err"].agg(
        count="count", mean_m="mean",
        median_m="median",
        p90_m=lambda x: np.percentile(x, 90)
    ).reset_index()
    return agg

def macro_group_stats(errors_m: np.ndarray,
                      group_ids: Iterable[Any]) -> Dict[str, float]:
    """Macro vs micro stats over groups (e.g., MMSI): macro=avg over group means, micro=over all samples."""
    s_err = pd.Series(np.asarray(errors_m))
    s_grp = pd.Series(list(group_ids))
    df = pd.DataFrame({"err": s_err, "g": s_grp}).dropna()
    # micro
    micro_mean = float(df["err"].mean())
    micro_rmse = float(np.sqrt(np.mean(df["err"].values**2))) #type: ignore
    # macro (mean of per-group means / rmses)
    g_means = df.groupby("g")["err"].mean().values
    g_rmses = df.groupby("g")["err"].apply(lambda x: np.sqrt(np.mean(x.values**2))).values
    macro_mean = float(np.mean(g_means)) #type: ignore
    macro_rmse = float(np.mean(g_rmses)) #type: ignore
    return {
        "micro_mean_m": micro_mean,
        "micro_rmse_m": micro_rmse,
        "macro_mean_m": macro_mean,
        "macro_rmse_m": macro_rmse,
        "n_groups": float(len(g_means)),
        "n_samples": float(len(df)),
    }

def evaluate_phaseA(df_input,
                    model_path: str,
                    scaler: StandardScaler,
                    hit_threshold_m: float = 100.0,
                    seq_len: int = 10) -> Dict[str, float]:
    """
    Wrapper Phase A: predict 1-step trên df_input rồi tính metrics RMSE/MAE/... bằng scaler.
    - df_input: DataFrame windows t0..t9 + X_norm,Y_norm
    - model_path: checkpoint .pt đã train (state_dict)
    - scaler: meta["scaler_xy"]
    """
    # 1) Predict (chuẩn hoá)
    pred_norm = predict(df_input, model_path=model_path)  # (N,2)
    true_norm = df_input[["X_norm","Y_norm"]].to_numpy(dtype="float32")

    # 2) Metrics (mét)
    metrics = compute_point_metrics_norm(pred_norm, true_norm, scaler, hit_threshold_m=hit_threshold_m)
    return metrics