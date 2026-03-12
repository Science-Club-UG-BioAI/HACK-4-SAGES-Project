import json
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
import numpy as np
from pathlib import Path
from ML.download import InaraDataset


def make_fixed_grid(min_w=0.5, max_w=20.0, n_points=2048):
    return np.linspace(min_w, max_w, n_points, dtype=np.float32)


def _ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _savefig(path, dpi=160):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def _safe_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _channel_names(dataset: InaraDataset):
    names = ["signal"]
    if dataset.include_noise:
        names.append("noise")
    if dataset.include_stellar_signal:
        names.append("stellar_signal")
    return names


def _plot_hist(series: pd.Series, title: str, out_path: Path, bins: int = 40):
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if len(vals) == 0:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(vals, bins=bins)
    plt.title(title)
    plt.xlabel(series.name if series.name else "value")
    plt.ylabel("count")
    _savefig(out_path)


def _plot_box(series: pd.Series, title: str, out_path: Path):
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if len(vals) == 0:
        return
    plt.figure(figsize=(8, 4))
    plt.boxplot(vals, vert=False)
    plt.title(title)
    plt.xlabel(series.name if series.name else "value")
    _savefig(out_path)


def _plot_corr_heatmap(df: pd.DataFrame, title: str, out_path: Path):
    if df.shape[1] < 2:
        return
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr.values, aspect="auto")
    plt.colorbar(im)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title(title)
    _savefig(out_path)


def _plot_scatter_matrix(
    df: pd.DataFrame, title: str, out_path: Path, max_rows: int = 1000
):
    df = df.dropna()
    if df.shape[1] < 2 or len(df) < 2:
        return
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)
    _ = plt.figure(figsize=(12, 12))
    scatter_matrix(df, diagonal="hist", figsize=(12, 12))
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close("all")


def _plot_mean_std(wavelength, mean, std, title, out_path):
    plt.figure(figsize=(10, 5))
    plt.plot(wavelength, mean, label="mean")
    plt.fill_between(wavelength, mean - std, mean + std, alpha=0.25, label="± std")
    plt.xlabel("Wavelength")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    _savefig(out_path)


def _plot_quantiles(wavelength, q10, q50, q90, title, out_path):
    plt.figure(figsize=(10, 5))
    plt.plot(wavelength, q50, label="median")
    plt.fill_between(wavelength, q10, q90, alpha=0.25, label="10-90%")
    plt.xlabel("Wavelength")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    _savefig(out_path)


def _plot_example_spectra(wavelength, Xc, title, out_path, n_examples=20):
    n = min(n_examples, Xc.shape[0])
    if n == 0:
        return
    idx = np.linspace(0, Xc.shape[0] - 1, n, dtype=int)
    plt.figure(figsize=(10, 5))
    for i in idx:
        plt.plot(wavelength, Xc[i], alpha=0.6)
    plt.xlabel("Wavelength")
    plt.ylabel("Value")
    plt.title(title)
    _savefig(out_path)


def _plot_hist_from_array(arr, title, out_path, bins=40, log10_if_positive=False):
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    if log10_if_positive and np.all(arr > 0):
        arr = np.log10(arr)
        xlabel = "log10(value)"
    else:
        xlabel = "value"

    plt.figure(figsize=(8, 5))
    plt.hist(arr, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    _savefig(out_path)


def analyze_inara_dataset(
    dataset: InaraDataset,
    target_cols,
    out_dir="reports/inara_analysis",
    max_spectra=1000,
    pca_max_samples=500,
    example_spectra=20,
):
    out_dir = _ensure_dir(out_dir)
    plots_dir = _ensure_dir(out_dir / "plots")
    tables_dir = _ensure_dir(out_dir / "tables")

    df = dataset.df.copy()
    numeric_df = _safe_numeric_df(df)

    summary = numeric_df.describe().T
    summary["missing_count"] = numeric_df.isna().sum()
    summary["missing_frac"] = numeric_df.isna().mean()
    summary.to_csv(tables_dir / "numeric_summary.csv")

    df.dtypes.astype(str).to_csv(tables_dir / "column_dtypes.csv", header=["dtype"])
    df.isna().sum().sort_values(ascending=False).to_csv(
        tables_dir / "missing_values.csv", header=["missing_count"]
    )

    missing = df.isna().sum().sort_values(ascending=False)
    plt.figure(figsize=(12, 5))
    plt.bar(missing.index.astype(str), missing.values)
    plt.xticks(rotation=90)
    plt.ylabel("missing count")
    plt.title("Missing values per column")
    _savefig(plots_dir / "missing_values.png")

    if "n_points" in df.columns:
        _plot_hist(
            df["n_points"], "Distribution of n_points", plots_dir / "hist_n_points.png"
        )
        _plot_box(df["n_points"], "Boxplot of n_points", plots_dir / "box_n_points.png")

    available_targets = [c for c in target_cols if c in df.columns]
    target_df = (
        _safe_numeric_df(df[available_targets]) if available_targets else pd.DataFrame()
    )

    if not target_df.empty:
        target_df.describe().T.to_csv(tables_dir / "target_summary.csv")
        target_df.corr().to_csv(tables_dir / "target_correlation.csv")

        for c in target_df.columns:
            _plot_hist(target_df[c], f"Histogram of {c}", plots_dir / f"hist_{c}.png")
            _plot_box(target_df[c], f"Boxplot of {c}", plots_dir / f"box_{c}.png")

        _plot_corr_heatmap(
            target_df,
            "Target correlation heatmap",
            plots_dir / "target_correlation_heatmap.png",
        )

        _plot_scatter_matrix(
            target_df,
            "Scatter matrix of target variables",
            plots_dir / "target_scatter_matrix.png",
        )

        target_sum = target_df.sum(axis=1)
        target_sum.to_csv(
            tables_dir / "target_row_sums.csv", index=False, header=["sum"]
        )
        _plot_hist(
            target_sum,
            "Sum of target abundances per sample",
            plots_dir / "target_sum_hist.png",
        )
        _plot_box(
            target_sum,
            "Sum of target abundances per sample",
            plots_dir / "target_sum_box.png",
        )

    indices = np.linspace(
        0, len(dataset) - 1, min(len(dataset), max_spectra), dtype=int
    )

    X_list = []
    wavelength = None
    sample_ids = []
    expected_shape = None
    skipped = 0

    for idx in indices:
        item = dataset[int(idx)]
        x = item["x"].detach().cpu().numpy().astype(np.float32)
        w = item["wavelength"].detach().cpu().numpy().astype(np.float32)

        if expected_shape is None:
            expected_shape = x.shape
            wavelength = w

        if x.shape != expected_shape:
            skipped += 1
            continue

        X_list.append(x)
        sample_ids.append(item["sample_id"])

    if len(X_list) == 0:
        raise RuntimeError(
            "Nie udało się wczytać żadnego spójnego sample'a do analizy."
        )

    X = np.stack(X_list, axis=0)
    channel_names = _channel_names(dataset)
    used_channel_names = channel_names[: X.shape[1]]

    stats_meta = {
        "n_dataset_total": int(len(dataset)),
        "n_spectra_loaded": int(X.shape[0]),
        "n_spectra_skipped_due_to_shape_mismatch": int(skipped),
        "x_shape": list(X.shape),
        "channel_names": used_channel_names,
    }

    with open(out_dir / "analysis_meta.json", "w", encoding="utf-8") as f:
        json.dump(stats_meta, f, indent=2, ensure_ascii=False)

    channel_rows = []
    for c, name in enumerate(used_channel_names):
        Xc = X[:, c, :]

        mean_curve = Xc.mean(axis=0)
        std_curve = Xc.std(axis=0)
        q10 = np.quantile(Xc, 0.10, axis=0)
        q50 = np.quantile(Xc, 0.50, axis=0)
        q90 = np.quantile(Xc, 0.90, axis=0)

        _plot_mean_std(
            wavelength,
            mean_curve,
            std_curve,
            f"{name}: mean ± std",
            plots_dir / f"{name}_mean_std.png",
        )

        _plot_quantiles(
            wavelength,
            q10,
            q50,
            q90,
            f"{name}: median and 10-90% band",
            plots_dir / f"{name}_quantiles.png",
        )

        _plot_example_spectra(
            wavelength,
            Xc,
            f"{name}: example spectra",
            plots_dir / f"{name}_example_spectra.png",
            n_examples=example_spectra,
        )

        integrated = np.trapezoid(Xc, x=wavelength, axis=1)
        _plot_hist_from_array(
            integrated,
            f"{name}: integrated intensity",
            plots_dir / f"{name}_integrated_hist.png",
            bins=40,
            log10_if_positive=True,
        )

        channel_rows.append(
            {
                "channel": name,
                "global_mean": float(np.mean(Xc)),
                "global_std": float(np.std(Xc)),
                "global_min": float(np.min(Xc)),
                "global_max": float(np.max(Xc)),
                "integrated_mean": float(np.mean(integrated)),
                "integrated_std": float(np.std(integrated)),
                "integrated_min": float(np.min(integrated)),
                "integrated_max": float(np.max(integrated)),
            }
        )

    pd.DataFrame(channel_rows).to_csv(tables_dir / "channel_summary.csv", index=False)

    n_pca = min(pca_max_samples, X.shape[0])
    X_pca = X[:n_pca].reshape(n_pca, -1)

    X_pca = np.log10(np.clip(X_pca, 1e-30, None))

    if n_pca >= 3:
        pca = PCA(n_components=2, random_state=42)
        Z = pca.fit_transform(X_pca)

        plt.figure(figsize=(7, 6))
        plt.scatter(Z[:, 0], Z[:, 1], s=12, alpha=0.8)
        plt.xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.2f}%)")
        plt.ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.2f}%)")
        plt.title("PCA of spectra")
        _savefig(plots_dir / "spectra_pca.png")

        pd.DataFrame(
            {
                "PC1": Z[:, 0],
                "PC2": Z[:, 1],
                "sample_id": sample_ids[:n_pca],
            }
        ).to_csv(tables_dir / "spectra_pca_projection.csv", index=False)

        with open(
            tables_dir / "spectra_pca_explained_variance.json", "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                    "explained_variance": pca.explained_variance_.tolist(),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    report_lines = []
    report_lines.append(f"Liczba próbek w dataset: {len(dataset)}")
    report_lines.append(f"Liczba próbek użytych do analizy widm: {X.shape[0]}")
    report_lines.append(f"Liczba pominiętych próbek przez mismatch shape: {skipped}")
    report_lines.append(f"Kształt tensora widm: {X.shape}")
    report_lines.append(f"Kanały: {used_channel_names}")

    if not target_df.empty:
        report_lines.append("")
        report_lines.append("Statystyki targetów:")
        for c in target_df.columns:
            vals = target_df[c].dropna()
            if len(vals) == 0:
                continue
            report_lines.append(
                f"{c}: mean={vals.mean():.6g}, std={vals.std():.6g}, "
                f"min={vals.min():.6g}, max={vals.max():.6g}"
            )

    with open(out_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"[OK] Zapisano analizę do: {out_dir}")


target_cols = ["H2O", "CO2", "O2", "N2", "CH4", "O3"]
fixed_grid = make_fixed_grid(min_w=0.5, max_w=20.0, n_points=2048)
dataset = InaraDataset(
    index_csv="data/inara/index.csv",
    target_cols=target_cols,
    include_noise=True,
    include_stellar_signal=False,
    fixed_grid=fixed_grid,
    log_targets=False,
)

analyze_inara_dataset(
    dataset,
    target_cols=target_cols,
    out_dir="reports/",
    max_spectra=1000,
    pca_max_samples=500,
    example_spectra=20,
)
