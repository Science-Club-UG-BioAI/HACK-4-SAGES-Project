import torch
import numpy as np
from typing import Tuple
import math
from pathlib import Path

from ML.main import (
    MultiHeadInaraRegressor,
    make_3channel_spectrum,
    TargetNormalizer,
    target_cols,
)
from matplotlib import pyplot as plt

OLD_GRID = np.linspace(0.5, 20, 15346)
FIXED_GRID = np.linspace(0.5, 20, 2048)
BATCH_SIZE = 64
normalizer = TargetNormalizer.load("target_normalizer.json")


def load_model(model_path: str, device: str | torch.device) -> MultiHeadInaraRegressor:
    model = MultiHeadInaraRegressor(in_channels=3).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict(
    model: MultiHeadInaraRegressor,
    spectrum: torch.Tensor | np.ndarray,
    num_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(spectrum, torch.Tensor):
        spectrum = spectrum.detach().cpu().numpy()

    spectrum = np.interp(FIXED_GRID, OLD_GRID, spectrum).astype(np.float32)
    spectrum = torch.tensor(spectrum, dtype=torch.float32)

    model.eval()

    if spectrum.ndim == 1:
        spectrum = spectrum.unsqueeze(0)

    device = next(model.parameters()).device
    spectrum = spectrum.to(device)
    spectrum = torch.log1p(spectrum.unsqueeze(0) * 1e27)
    spectrum = make_3channel_spectrum(spectrum, torch.Tensor(FIXED_GRID))
    all_predictions = []
    samples_left = num_samples

    with torch.no_grad():
        while samples_left > 0:
            current_real = min(BATCH_SIZE, samples_left)
            noise_std = 1e-2
            batch = spectrum.repeat(current_real, *([1] * (spectrum.ndim - 1)))
            batch = batch + noise_std * torch.randn_like(batch)
            if current_real < BATCH_SIZE:
                pad_shape = (BATCH_SIZE - current_real, *batch.shape[1:])
                padding = torch.zeros(pad_shape, dtype=batch.dtype, device=device)
                batch = torch.cat([batch, padding], dim=0)

            predicted = model(batch)
            predicted = predicted[:current_real]
            all_predictions.append(predicted.detach().cpu())

            samples_left -= current_real

    predicted = torch.cat(all_predictions, dim=0)
    predicted = normalizer.inverse_transform_tensor(predicted)
    mean = predicted.mean(dim=0)
    std = predicted.std(dim=0, unbiased=False)
    stderr = std / math.sqrt(num_samples)

    return (
        predicted.numpy(),
        mean.numpy(),
        std.numpy(),
        stderr.numpy(),
    )


def save_prediction_boxplots(
    all_preds: np.ndarray,
    mean: np.ndarray,
    stderr: np.ndarray,
    target_cols: list[str],
    output_path: str = "prediction_boxplots.png",
    ncols: int = 3,
    figsize_per_subplot: tuple[float, float] = (5.5, 4.0),
) -> None:
    all_preds = np.asarray(all_preds)
    mean = np.asarray(mean)
    stderr = np.asarray(stderr)

    if all_preds.ndim != 2:
        raise ValueError(
            f"all_preds musi mieć shape [num_samples, num_targets], a ma {all_preds.shape}"
        )

    _, num_targets = all_preds.shape

    if len(target_cols) != num_targets:
        raise ValueError(
            f"Liczba target_cols ({len(target_cols)}) nie zgadza się z liczbą targetów ({num_targets})"
        )

    if mean.shape != (num_targets,):
        raise ValueError(f"mean musi mieć shape ({num_targets},), a ma {mean.shape}")

    if stderr.shape != (num_targets,):
        raise ValueError(
            f"stderr musi mieć shape ({num_targets},), a ma {stderr.shape}"
        )

    nrows = math.ceil(num_targets / ncols)
    fig_w = figsize_per_subplot[0] * ncols
    fig_h = figsize_per_subplot[1] * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)

    for i in range(num_targets):
        ax = axes[i]
        vals = all_preds[:, i]

        ax.boxplot([vals], tick_labels=[""], showfliers=True)

        ax.scatter([1], [mean[i]], marker="o", zorder=3)
        ax.errorbar(
            [1],
            [mean[i]],
            yerr=[stderr[i]],
            fmt="none",
            capsize=4,
            elinewidth=1.5,
            zorder=4,
        )

        ax.set_title(target_cols[i], fontsize=10, pad=10)
        ax.set_ylabel("Predicted value")
        ax.tick_params(axis="x", length=0)

    for j in range(num_targets, len(axes)):
        axes[j].axis("off")

    plt.subplots_adjust(
        left=0.06,
        right=0.98,
        bottom=0.05,
        top=0.96,
        wspace=0.35,
        hspace=0.55,
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    model = load_model("best_model.pt", "cuda")
    spec = np.fromstring(
        Path(
            "/home/tezriem/Documents/-HACK-4-SAGES-Project/data/inara/raw/unpacked/0000000-0010000/0000000-0010000/0009997_planet_signal.csv"
        ).read_text(encoding="utf-8"),
        sep=",",
    )

    all_preds, mean, std, stderr = predict(model, spec, 10)

    save_prediction_boxplots(all_preds, mean, stderr, target_cols)

    print("all_preds shape:", all_preds)
    print("mean shape:", mean.shape)
    print("std shape:", std.shape)
    print("stderr shape:", stderr.shape)
    print("mean:", mean)
    print("std:", std)
    print("stderr:", stderr)
