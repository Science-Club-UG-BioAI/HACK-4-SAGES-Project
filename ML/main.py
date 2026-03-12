import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm

from ML.download import InaraDataset

from torchinfo import summary

SEED = 2137
EPOCHS = 0 #35
BATCH_SIZE = 128
LR = 3e-4
WEIGHT_DECAY = 1e-4

DATA_DIR = Path("data/inara")
RAW_INDEX_CSV = DATA_DIR / "index.csv"
SPLIT_DIR = DATA_DIR / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = SPLIT_DIR / "train_norm.csv"
VAL_CSV = SPLIT_DIR / "val_norm.csv"
TEST_CSV = SPLIT_DIR / "test_norm.csv"

NORMALIZER_JSON = DATA_DIR / "target_normalizer.json"
BEST_MODEL_PATH = Path("best_model.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REAL_COLS = [
    "planet_radius_(Earth_radii)",
    "planet_density_(g/cm3)",
    "planet_surface_pressure_(bar)",
    "planet_surface_temperature_(Kelvin)",
]

BOUNDED_COLS = [
    "H2O",
    "CO2",
    "O2",
    "N2",
    "CH4",
    "N2O",
    "CO",
    "O3",
    "SO2",
    "NH3",
    "C2H6",
    "NO2",
    "planet's_mean_surface_albedo_(unitless)",
]


TRANSFORM_TYPES = {
    "planet_radius_(Earth_radii)": "real_standard",
    "planet_density_(g/cm3)": "real_standard",
    "planet_surface_pressure_(bar)": "real_standard",
    "planet_surface_temperature_(Kelvin)": "real_standard",
    "H2O": "bounded_identity",
    "CO2": "bounded_identity",
    "O2": "bounded_identity",
    "N2": "bounded_identity",
    "CH4": "bounded_identity",
    "planet's_mean_surface_albedo_(unitless)": "bounded_identity",
    "N2O": "bounded_log1p_scaled",
    "CO": "bounded_log1p_scaled",
    "O3": "bounded_log1p_scaled",
    "SO2": "bounded_log1p_scaled",
    "NH3": "bounded_log1p_scaled",
    "C2H6": "bounded_log1p_scaled",
    "NO2": "bounded_log1p_scaled",
}

K_VALUES = {
    "N2O": 1e3,
    "CO": 1e3,
    "O3": 1e4,
    "SO2": 1e3,
    "NH3": 1e3,
    "C2H6": 1e4,
    "NO2": 1e5,
}


def make_3channel_spectrum(
    x: torch.Tensor, wavelengths: torch.Tensor | None = None
) -> torch.Tensor:
    single = False
    if x.ndim == 1:
        x = x.unsqueeze(0)
        single = True

    b, _, l = x.shape
    if wavelengths is None:
        d1 = torch.diff(x, dim=-1, prepend=x[..., :1])
        d2 = torch.diff(d1, dim=-1, prepend=d1[..., :1])
    else:
        if wavelengths.ndim != 1:
            raise ValueError("wavelengths need to have shape: [L]")

        wavelengths = wavelengths.to(x.device, x.dtype)
        dx = x[..., 1:] - x[..., :-1]
        dl = wavelengths[1:] - wavelengths[:-1]

        d1_inner = dx / dl.unsqueeze(0)
        d1 = torch.cat([d1_inner[..., :1], d1_inner], dim=-1)

        ddx = d1[..., 1:] - d1[..., :-1]
        ddl = wavelengths[1:] - wavelengths[:-1]
        d2_inner = ddx / ddl.unsqueeze(0)
        d2 = torch.cat([d2_inner[..., :1], d2_inner], dim=-1)

    out = torch.stack([x, d1, d2], dim=1)

    if single:
        out = out.squeeze(0)

    return out.reshape(b, 3, l)


target_cols = [
    # "star_temperature_(Kelvin)",
    # "star_radius_(Solar_radii)",
    # "distance_from_Earth_to_the_system_(parsec)",
    # "semimajor_axis_of_the_planet_(AU)",
    "planet_radius_(Earth_radii)",
    "planet_density_(g/cm3)",
    "planet_surface_pressure_(bar)",
    # "kappa",
    # "gamma1",
    # "gamma2",
    # "alpha",
    # "beta",
    "planet_surface_temperature_(Kelvin)",
    "H2O",
    "CO2",
    "O2",
    "N2",
    "CH4",
    "N2O",
    "CO",
    "O3",
    "SO2",
    "NH3",
    "C2H6",
    "NO2",
    "planet's_mean_surface_albedo_(unitless)",
]


def set_seed(seed: int = 2137):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_fixed_grid(min_w: float = 0.5, max_w: float = 20.0, n_points: int = 2048):
    return np.linspace(min_w, max_w, n_points, dtype=np.float32)


def get_valid_num_groups(num_channels: int, preferred_groups: int = 8) -> int:
    for g in range(min(preferred_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


def split_dataframe(df: pd.DataFrame, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)

    n_total = len(df)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    return df_train, df_val, df_test


class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1)
        return x * w


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 8,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=((kernel_size - 1) // 2) * dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(num_groups=min(get_valid_num_groups(out_ch, groups), out_ch), num_channels=out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiScaleResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()

        c1 = out_ch // 3
        c2 = out_ch // 3
        c3 = out_ch - c1 - c2

        self.b1 = ConvGNAct(in_ch, c1, kernel_size=3, stride=stride, dilation=1)
        self.b2 = ConvGNAct(in_ch, c2, kernel_size=7, stride=stride, dilation=1)
        self.b3 = ConvGNAct(in_ch, c3, kernel_size=15, stride=stride, dilation=1)

        self.mix = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.GELU(),
        )

        self.se = SEBlock1D(out_ch)

        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        out = self.mix(out)
        out = self.se(out)
        out = out + self.skip(x)
        return F.gelu(out)


class AdaptiveConcatPool1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.max = nn.AdaptiveMaxPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.avg(x), self.max(x)], dim=1)


class HeadReal(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class HeadBounded(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.first = nn.Sequential(
            nn.Linear(in_c, out_c),
            nn.GELU(),
            nn.Dropout(0.05),
        )

        self.second = nn.Sequential(
            nn.Linear(out_c, out_c),
            nn.GELU(),
            nn.Dropout(0.05),
        )

        self.third = nn.Sequential(
            nn.Linear(out_c, out_c),
            nn.GELU(),
            nn.Dropout(0.05),
        )

        self.fourth = nn.Sequential(
            nn.Linear(out_c, out_c),
            nn.GELU(),
            nn.Dropout(0.05),
        )

        self.skip = nn.Linear(in_c, out_c) if in_c != out_c else nn.Identity()

    def forward(self, x):
        long_res = self.skip(x)

        out = self.first(x)
        short_res = out

        out = self.second(out)
        out = self.third(out) + short_res

        out = self.fourth(out) + long_res
        return out


class MultiHeadInaraRegressor(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()

        self.stem = nn.Sequential(
            ConvGNAct(in_channels, 64, kernel_size=9, stride=1),
            ConvGNAct(64, 64, kernel_size=7, stride=1),
        )

        self.encoder = nn.Sequential(
            MultiScaleResBlock1D(64, 128, stride=2),
            MultiScaleResBlock1D(128, 128, stride=1),
            MultiScaleResBlock1D(128, 256, stride=2),
            MultiScaleResBlock1D(256, 256, stride=1),
            MultiScaleResBlock1D(256, 512, stride=2),
            MultiScaleResBlock1D(512, 512, stride=1),
        )

        self.pool = AdaptiveConcatPool1d()

        self.shared = nn.Sequential(
            nn.Flatten(),
            *[ResBlock(1024, 1024) for _ in range(3)],
            ResBlock(1024, 512),
            ResBlock(512, 256),
        )

        self.heads = nn.ModuleList(
            [
                *[HeadReal(in_dim=256, hidden_dim=512) for _ in range(4)],
                *[HeadBounded(in_dim=256, hidden_dim=512) for _ in range(13)],
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)

        feat = self.stem(x)
        feat = self.encoder(feat)
        feat = self.pool(feat)
        feat = self.shared(feat)

        out = [x(feat) for x in self.heads]
        out = torch.cat(out, dim=-1)
        return out
    
@dataclass
class TargetNormalizer:
    target_cols: list[str]
    transform_types: dict[str, str]
    k_values: dict[str, float]
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit_from_df(
        cls,
        df: pd.DataFrame,
        target_cols: list[str],
        transform_types: dict[str, str],
        k_values: dict[str, float] | None = None,
    ):
        if k_values is None:
            k_values = {}

        missing = [c for c in target_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Column not found: {missing}")

        mean = np.zeros(len(target_cols), dtype=np.float32)
        std = np.ones(len(target_cols), dtype=np.float32)

        for i, col in enumerate(target_cols):
            mode = transform_types.get(col, "identity")
            values = df[col].to_numpy(dtype=np.float32)

            if mode == "real_standard":
                mean[i] = np.mean(values, dtype=np.float64)
                s = np.std(values, ddof=0, dtype=np.float64)
                std[i] = 1.0 if (not np.isfinite(s) or s == 0) else float(s)

            elif mode in ("bounded_identity", "bounded_log1p_scaled"):
                mean[i] = 0.0
                std[i] = 1.0

            else:
                raise ValueError(f"Unknown transform type  {col}: {mode}")

        return cls(
            target_cols=target_cols,
            transform_types=transform_types,
            k_values=k_values,
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
        )

    def _transform_column_np(self, col: str, x: np.ndarray) -> np.ndarray:
        mode = self.transform_types.get(col, "identity")

        if mode == "real_standard":
            idx = self.target_cols.index(col)
            return (x - self.mean[idx]) / self.std[idx]

        elif mode == "bounded_identity":
            return x

        elif mode == "bounded_log1p_scaled":
            k = float(self.k_values[col])
            x = np.clip(x, 0.0, 1.0)
            return np.log1p(k * x) / np.log1p(k)

        else:
            raise ValueError(f"Unknown transform type  {col}: {mode}")

    def _inverse_transform_column_np(self, col: str, x: np.ndarray) -> np.ndarray:
        mode = self.transform_types.get(col, "identity")

        if mode == "real_standard":
            idx = self.target_cols.index(col)
            return x * self.std[idx] + self.mean[idx]

        elif mode == "bounded_identity":
            return np.clip(x, 0.0, 1.0)

        elif mode == "bounded_log1p_scaled":
            k = float(self.k_values[col])
            x = np.clip(x, 0.0, 1.0)
            return np.expm1(x * np.log1p(k)) / k

        else:
            raise ValueError(f"Unknown transform type  {col}: {mode}")

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.target_cols:
            arr = df[col].to_numpy(dtype=np.float32)
            df[col] = self._transform_column_np(col, arr)
        return df

    def inverse_transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.target_cols:
            arr = df[col].to_numpy(dtype=np.float32)
            df[col] = self._inverse_transform_column_np(col, arr)
        return df

    def transform_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != len(self.target_cols):
            raise ValueError(
                f"Last dim of tensor must be {len(self.target_cols)}, "
                f"is :{x.shape[-1]}"
            )

        out = x.clone()

        for i, col in enumerate(self.target_cols):
            mode = self.transform_types.get(col, "identity")

            if mode == "real_standard":
                mean = torch.tensor(self.mean[i], dtype=x.dtype, device=x.device)
                std = torch.tensor(self.std[i], dtype=x.dtype, device=x.device)
                out[..., i] = (out[..., i] - mean) / std

            elif mode == "bounded_identity":
                out[..., i] = out[..., i].clamp(0.0, 1.0)

            elif mode == "bounded_log1p_scaled":
                k = torch.tensor(float(self.k_values[col]), dtype=x.dtype, device=x.device)
                out[..., i] = torch.log1p(k * out[..., i].clamp(0.0, 1.0)) / torch.log1p(k)

            else:
                raise ValueError(f"Unknown transform type {col}: {mode}")

        return out

    def inverse_transform_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != len(self.target_cols):
            raise ValueError(
                f"LAst dim of tensor must be {len(self.target_cols)}, "
                f"is: {x.shape[-1]}"
            )

        out = x.clone()

        for i, col in enumerate(self.target_cols):
            mode = self.transform_types.get(col, "identity")

            if mode == "real_standard":
                mean = torch.tensor(self.mean[i], dtype=x.dtype, device=x.device)
                std = torch.tensor(self.std[i], dtype=x.dtype, device=x.device)
                out[..., i] = out[..., i] * std + mean

            elif mode == "bounded_identity":
                out[..., i] = out[..., i].clamp(0.0, 1.0)

            elif mode == "bounded_log1p_scaled":
                k = torch.tensor(float(self.k_values[col]), dtype=x.dtype, device=x.device)
                out[..., i] = torch.expm1(out[..., i].clamp(0.0, 1.0) * torch.log1p(k)) / k

            else:
                raise ValueError(f"Unknown transform type {col}: {mode}")

        return out

    def save(self, path: str | Path):
        path = Path(path)
        payload = {
            "target_cols": self.target_cols,
            "transform_types": self.transform_types,
            "k_values": self.k_values,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            target_cols=payload["target_cols"],
            transform_types=payload["transform_types"],
            k_values={k: float(v) for k, v in payload["k_values"].items()},
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
        )


def prepare_splits_and_normalization(index_csv_path: Path, target_cols: list[str]):
    df_all = pd.read_csv(index_csv_path)

    df_train, df_val, df_test = split_dataframe(df_all, seed=SEED)

    normalizer = TargetNormalizer.fit_from_df(
        df_train,
        target_cols=target_cols,
        transform_types=TRANSFORM_TYPES,
        k_values=K_VALUES,
    )

    df_train_norm = normalizer.transform_df(df_train)
    df_val_norm = normalizer.transform_df(df_val)
    df_test_norm = normalizer.transform_df(df_test)

    df_train_norm.to_csv(TRAIN_CSV, index=False)
    df_val_norm.to_csv(VAL_CSV, index=False)
    df_test_norm.to_csv(TEST_CSV, index=False)

    normalizer.save(NORMALIZER_JSON)

    return df_train, df_val, df_test, normalizer


def build_datasets(fixed_grid, target_cols):
    train_dataset = InaraDataset(
        index_csv=str(TRAIN_CSV),
        target_cols=target_cols,
        include_noise=True,
        include_stellar_signal=False,
        fixed_grid=fixed_grid,
        log_targets=False,
    )

    val_dataset = InaraDataset(
        index_csv=str(VAL_CSV),
        target_cols=target_cols,
        include_noise=False,
        include_stellar_signal=False,
        fixed_grid=fixed_grid,
        log_targets=False,
    )

    test_dataset = InaraDataset(
        index_csv=str(TEST_CSV),
        target_cols=target_cols,
        include_noise=False,
        include_stellar_signal=False,
        fixed_grid=fixed_grid,
        log_targets=False,
    )

    return train_dataset, val_dataset, test_dataset


def build_loaders(train_dataset, val_dataset, test_dataset):
    pin_memory = DEVICE.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, train_eval_loader


def run_epoch(model, loader, criterion, fixed_grid_t, optimizer=None):
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    loss_sum = 0.0
    n_samples = 0

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch in loader:
            x = torch.log1p(batch["x"].float().to(DEVICE, non_blocking=True) * 1e27)
            y = batch["y"].float().to(DEVICE, non_blocking=True)
            x = make_3channel_spectrum(x, wavelengths=fixed_grid_t.to(DEVICE))
            pred = model(x)
            loss = criterion(pred, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            bs = x.size(0)
            loss_sum += loss.item() * bs
            n_samples += bs

    return loss_sum / max(n_samples, 1)


def evaluate_and_collect(model, loader, criterion, fixed_grid_t):
    model.eval()

    loss_sum = 0.0
    n_samples = 0

    preds_all = []
    targets_all = []

    with torch.no_grad():
        for batch in loader:
            x = torch.log1p(batch["x"].float().to(DEVICE, non_blocking=True) * 1e27)
            y = batch["y"].float().to(DEVICE, non_blocking=True)
            x = make_3channel_spectrum(x, wavelengths=fixed_grid_t.to(DEVICE))
            pred = model(x)
            loss = criterion(pred, y)

            bs = x.size(0)
            loss_sum += loss.item() * bs
            n_samples += bs

            preds_all.append(pred.detach().cpu())
            targets_all.append(y.detach().cpu())

    avg_loss = loss_sum / max(n_samples, 1)
    preds_all = torch.cat(preds_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)

    return avg_loss, preds_all, targets_all


def print_one_train_example(
    model, train_eval_loader, normalizer: TargetNormalizer, fixed_grid_t, idx: int
):
    model.eval()

    batch = next(iter(train_eval_loader))
    x = torch.log1p(batch["x"].float().to(DEVICE) * 1e27)
    x = make_3channel_spectrum(x, wavelengths=fixed_grid_t.to(DEVICE))
    y_norm = batch["y"].float().to(DEVICE)

    with torch.no_grad():
        pred_norm = model(x)

    y_true = normalizer.inverse_transform_tensor(y_norm).cpu()
    y_pred = normalizer.inverse_transform_tensor(pred_norm).cpu()

    print("\n" + "=" * 100)
    print("EXAMPLE SAMPLE")
    print("=" * 100)

    for col_name, true_val, pred_val in zip(target_cols, y_true[idx], y_pred[idx]):
        print(
            f"{col_name:<55} | "
            f"true = {float(true_val):>14.6f} | "
            f"pred = {float(pred_val):>14.6f}"
        )

    mae = torch.mean(torch.abs(y_true[idx] - y_pred[idx])).item()
    rmse = torch.sqrt(torch.mean((y_true[idx] - y_pred[idx]) ** 2)).item()

    print("-" * 100)
    print(f"Sample MAE  = {mae:.6f}")
    print(f"Sample RMSE = {rmse:.6f}")
    print("=" * 100 + "\n")


def plot_test_boxplots(
    model,
    test_loader,
    criterion,
    fixed_grid_t,
    normalizer: TargetNormalizer,
    target_cols: list[str],
    save_path: str | Path = "test_boxplots.png",
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    _, preds_norm, targets_norm = evaluate_and_collect(
        model=model,
        loader=test_loader,
        criterion=criterion,
        fixed_grid_t=fixed_grid_t,
    )

    preds = normalizer.inverse_transform_tensor(preds_norm).cpu().numpy()
    targets = normalizer.inverse_transform_tensor(targets_norm).cpu().numpy()

    n_targets = len(target_cols)
    n_cols = 4
    n_rows = int(np.ceil(n_targets / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 5.5, n_rows * 4.5),
        squeeze=False,
    )

    axes = axes.flatten()

    for i, col_name in enumerate(target_cols):
        ax = axes[i]

        true_vals = targets[:, i]
        pred_vals = preds[:, i]

        valid_true = np.isfinite(true_vals)
        valid_pred = np.isfinite(pred_vals)

        true_vals = true_vals[valid_true]
        pred_vals = pred_vals[valid_pred]

        ax.boxplot(
            [true_vals, pred_vals],
            tick_labels=["true", "pred"],
            showfliers=False,
        )
        ax.set_title(col_name, fontsize=10)
        ax.grid(True, alpha=0.3)

    for j in range(n_targets, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Boxplots of ttru vs predicted values in test dataset", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Boxplots saved: {save_path.resolve()}")


def overfit_one_batch(
    model,
    loader,
    fixed_grid_t,
    steps: int = 500,
    lr: float = 1e-3,
):
    model = model.to(DEVICE)
    model.train()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    batch = next(iter(loader))
    x = torch.log1p(batch["x"].float().to(DEVICE) * 1e27)
    y = batch["y"].float().to(DEVICE)
    x = make_3channel_spectrum(x, wavelengths=fixed_grid_t.to(DEVICE))

    for step in range(steps):
        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"step={step:04d} loss={loss.item():.8f}")

    return model


def inspect_3channel_stats(loader, fixed_grid_t):
    batch = next(iter(loader))
    x = torch.log1p(batch["x"].float() * 1e27)
    x3 = make_3channel_spectrum(x, wavelengths=fixed_grid_t)

    names = ["raw", "d1", "d2"]
    for i, name in enumerate(names):
        ch = x3[:, i, :]
        print(
            f"{name:>4} | mean={ch.mean().item(): .6f} | "
            f"std={ch.std().item(): .6f} | "
            f"maxabs={ch.abs().max().item(): .6f}"
        )


class HybridRegressionLoss(nn.Module):
    def __init__(
        self,
        delta: float = 1.0,
        w_huber: float = 1.0,
        w_corr: float = 0.2,
        w_var: float = 0.05,
        eps: float = 1e-8,
        target_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.delta = delta
        self.w_huber = w_huber
        self.w_corr = w_corr
        self.w_var = w_var
        self.eps = eps

        if target_weights is not None:
            target_weights = target_weights.float()
            target_weights = target_weights / target_weights.sum().clamp_min(eps)
            self.register_buffer("target_weights", target_weights)
        else:
            self.target_weights = None

    def _reduce_targets(self, x: torch.Tensor) -> torch.Tensor:
        if self.target_weights is None:
            return x.mean()
        return (x * self.target_weights).sum()

    def huber_component(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        per_elem = F.huber_loss(
            pred, target, reduction="none", delta=self.delta
        )
        per_target = per_elem.mean(dim=0)
        return self._reduce_targets(per_target), per_target

    def pearson_component(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_c = pred - pred.mean(dim=0, keepdim=True)
        targ_c = target - target.mean(dim=0, keepdim=True)

        pred_std = pred_c.std(dim=0, unbiased=False).clamp_min(self.eps)
        targ_std = targ_c.std(dim=0, unbiased=False).clamp_min(self.eps)

        cov = (pred_c * targ_c).mean(dim=0)
        corr = cov / (pred_std * targ_std)

        per_target = 1.0 - corr
        return self._reduce_targets(per_target), per_target

    def variance_component(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_std = pred.std(dim=0, unbiased=False).clamp_min(self.eps)
        targ_std = target.std(dim=0, unbiased=False).clamp_min(self.eps)

        per_target = F.smooth_l1_loss(pred_std, targ_std, reduction="none")
        return self._reduce_targets(per_target), per_target

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_dict: bool = False,
    ):
        if pred.shape != target.shape:
            raise RuntimeError(
                f"Shape mismatch: pred={pred.shape}, target={target.shape}"
            )

        if pred.ndim != 2:
            raise ValueError(f"Expected [B, D], got pred.ndim={pred.ndim}")

        huber_loss, huber_per_target = self.huber_component(pred, target)
        corr_loss, corr_per_target = self.pearson_component(pred, target)
        var_loss, var_per_target = self.variance_component(pred, target)

        total = (
            self.w_huber * huber_loss + self.w_corr * corr_loss + self.w_var * var_loss
        )

        if not return_dict:
            return total

        return {
            "loss": total,
            "huber": huber_loss.detach(),
            "corr": corr_loss.detach(),
            "var": var_loss.detach(),
            "huber_per_target": huber_per_target.detach(),
            "corr_per_target": corr_per_target.detach(),
            "var_per_target": var_per_target.detach(),
        }


def main():
    set_seed(SEED)

    fixed_grid = make_fixed_grid(min_w=0.5, max_w=20.0, n_points=2048)
    fixed_grid_t = torch.tensor(fixed_grid, dtype=torch.float32)

    df_train_raw, df_val_raw, df_test_raw, normalizer = (
        prepare_splits_and_normalization(
            index_csv_path=RAW_INDEX_CSV,
            target_cols=target_cols,
        )
    )

    train_dataset, val_dataset, test_dataset = build_datasets(
        fixed_grid=fixed_grid,
        target_cols=target_cols,
    )

    print(
        f"dataset={len(df_train_raw) + len(df_val_raw) + len(df_test_raw)} | "
        f"train={len(train_dataset)} | "
        f"val={len(val_dataset)} | "
        f"test={len(test_dataset)}"
    )

    train_loader, val_loader, test_loader, train_eval_loader = build_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )

    model = MultiHeadInaraRegressor(in_channels=3, out_dim=len(target_cols)).to(DEVICE)

    # print(
    #     summary(
    #         model,
    #         input_size=(64, 3, 2048),
    #         col_names=["input_size", "output_size", "num_params", "trainable"],
    #         col_width=18,
    #         depth=4,
    #         row_settings=["var_names"],
    #     )
    # )

    target_weights = torch.tensor([1, 1, 1.3, 1.5, 1.3, 1.5, 3, 1.3, 1.1, 1.3, 1.3, 1, 1, 1, 1.3, 1, 1.3]).to(DEVICE)
    criterion = HybridRegressionLoss(
        delta=1.0, w_huber=1.0, w_corr=0.4, w_var=0.12, target_weights=target_weights
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    # inspect_3channel_stats(train_loader, fixed_grid_t)
    # overfit_one_batch(model, train_loader, fixed_grid_t)
    # inspect_dataset_sample(train_dataset, 0)
    # inspect_dataset_sample(train_dataset, 1)
    # inspect_dataset_sample(train_dataset, 2)

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        train_bar = tqdm(
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{EPOCHS} [train]",
            leave=False,
        )
        model.train()

        train_loss_sum = 0.0
        train_n = 0

        for batch in train_loader:
            x = torch.log1p(batch["x"].float().to(DEVICE, non_blocking=True) * 1e27)
            y = batch["y"].float().to(DEVICE, non_blocking=True)

            x = make_3channel_spectrum(x, wavelengths=fixed_grid_t.to(DEVICE))

            pred = model(x)
            loss_dict = criterion(pred, y, True)
            loss = loss_dict["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = x.size(0)
            train_loss_sum += loss.item() * bs
            train_n += bs

            train_bar.set_postfix(
                loss=f"{loss.item():.6f}",
                huber=f"{loss_dict['huber'].item():.4f}",
                corr=f"{loss_dict['corr'].item():.4f}",
                var=f"{loss_dict['var'].item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
            train_bar.update(1)

        train_bar.close()
        avg_train_loss = train_loss_sum / max(train_n, 1)

        val_loss, _, _ = evaluate_and_collect(
            model, val_loader, criterion, fixed_grid_t
        )

        scheduler.step(val_loss)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)

        print(
            f"Epoch {epoch + 1:03d}/{EPOCHS:03d} | "
            f"train_loss: {avg_train_loss:.6f} | "
            f"val_loss: {val_loss:.6f} | "
            f"best_val: {best_val_loss:.6f} | "
            f"lr: {optimizer.param_groups[0]['lr']:.2e}"
        )

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()

    test_loss, test_preds_norm, test_targets_norm = evaluate_and_collect(
        model, test_loader, criterion, fixed_grid_t
    )
    print(f"\nFinal test_loss: {test_loss:.6f}")

    test_preds = normalizer.inverse_transform_tensor(test_preds_norm)
    test_targets = normalizer.inverse_transform_tensor(test_targets_norm)

    test_mae = torch.mean(torch.abs(test_preds - test_targets)).item()
    test_rmse = torch.sqrt(torch.mean((test_preds - test_targets) ** 2)).item()

    print(f"Final test_MAE_denorm:  {test_mae:.6f}")
    print(f"Final test_RMSE_denorm: {test_rmse:.6f}")

    plot_test_boxplots(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        fixed_grid_t=fixed_grid_t,
        normalizer=normalizer,
        target_cols=target_cols,
        save_path="plots/test_boxplots.png",
    )

    print_one_train_example(model, train_eval_loader, normalizer, fixed_grid_t, 0)
    print_one_train_example(model, train_eval_loader, normalizer, fixed_grid_t, 1)
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training history")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
