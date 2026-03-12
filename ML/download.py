from __future__ import annotations

import argparse
import hashlib
import re
import shlex
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import torch
from torch.utils.data import Dataset

URL_RE = re.compile(r"https?://[^\s'\"\\]+")
DEFAULT_TIMEOUT = (15, 300)
PLANET_INDEX_RE = re.compile(r"(?<!\d)(\d{7})(?!\d)")


@dataclass
class DownloadJob:
    url: str
    filename: str


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def safe_filename(name: str) -> str:
    name = name.strip().replace("\\", "_").replace("/", "_")
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return name or "file.bin"


def parse_wget_script(script_path: str | Path) -> List[DownloadJob]:
    script_path = Path(script_path)
    jobs: List[DownloadJob] = []
    seen = set()

    for raw_line in script_path.read_text(
        encoding="utf-8", errors="ignore"
    ).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "wget" not in line:
            continue

        urls = URL_RE.findall(line)
        if not urls:
            continue
        url = urls[-1]

        tokens = shlex.split(line, comments=True, posix=True)
        filename = None

        for i, tok in enumerate(tokens):
            if tok == "-O" and i + 1 < len(tokens):
                filename = tokens[i + 1]
                break
            if tok.startswith("--output-document="):
                filename = tok.split("=", 1)[1]
                break

        if filename is None:
            filename = Path(urlparse(url).path).name
            if not filename:
                filename = f"{sha1_text(url)[:16]}.bin"

        filename = safe_filename(filename)
        key = (url, filename)
        if key not in seen:
            jobs.append(DownloadJob(url=url, filename=filename))
            seen.add(key)

    if not jobs:
        raise RuntimeError(f"Nie znaleziono żadnych wpisów wget w pliku: {script_path}")
    return jobs


def download_file(
    session: requests.Session,
    job: DownloadJob,
    out_dir: Path,
    overwrite: bool = False,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / job.filename
    tmp_path = out_dir / (job.filename + ".part")

    if out_path.exists() and not overwrite:
        return out_path

    with session.get(job.url, stream=True, timeout=DEFAULT_TIMEOUT) as resp:
        resp.raise_for_status()
        with tmp_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    tmp_path.replace(out_path)
    return out_path


def download_from_wget_script(
    wget_script: str | Path,
    out_dir: str | Path,
    workers: int = 4,
    limit: Optional[int] = None,
    overwrite: bool = False,
) -> List[Path]:
    jobs = parse_wget_script(wget_script)
    if limit is not None:
        jobs = jobs[:limit]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded: List[Path] = []
    with requests.Session() as session:
        session.headers.update({"User-Agent": "inara-planet-signal-downloader/0.2"})
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(download_file, session, job, out_dir, overwrite): job
                for job in jobs
            }
            for fut in as_completed(futures):
                job = futures[fut]
                try:
                    path = fut.result()
                    downloaded.append(path)
                    print(f"[OK] {job.filename}")
                except Exception as e:
                    print(f"[ERR] {job.filename}: {e}")

    return sorted(downloaded)


def unpack_archives(
    raw_dir: str | Path, unpack_dir: Optional[str | Path] = None
) -> List[Path]:
    raw_dir = Path(raw_dir)
    unpack_dir = Path(unpack_dir) if unpack_dir is not None else raw_dir / "unpacked"
    unpack_dir.mkdir(parents=True, exist_ok=True)

    archives = sorted(raw_dir.glob("*.tar.gz"))
    out_dirs: List[Path] = []
    for arc in archives:
        target = unpack_dir / arc.name.replace(".tar.gz", "")
        if not target.exists() or not any(target.iterdir()):
            target.mkdir(parents=True, exist_ok=True)
            with tarfile.open(arc, "r:gz") as tf:
                tf.extractall(target)
            print(f"[UNPACK] {arc.name} -> {target}")
        else:
            print(f"[UNPACK-SKIP] {arc.name} -> {target} (już istnieje)")
        out_dirs.append(target)
    return out_dirs


def _normalize_colname(name: str) -> str:
    name = name.strip().lower()
    name = name.replace("-", " ")
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name)
    return name


WAVELENGTH_CANDIDATES = [
    "Wavelength",
    "lambda",
    "wavelength",
    "wave",
    "wl",
]
SIGNAL_CANDIDATES = [
    "Planet Signal",
    "planet signal",
    "signal",
    "flux",
    "radiance",
    "value",
    "y",
]
NOISE_CANDIDATES = ["Noise", "noise", "sigma", "uncertainty"]
STELLAR_CANDIDATES = ["Stellar Signal", "stellar signal", "star signal"]


def _resolve_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    normalized = {_normalize_colname(c): c for c in columns}
    for cand in candidates:
        key = _normalize_colname(cand)
        if key in normalized:
            return normalized[key]
    return None


def extract_planet_index_from_path(path: str | Path) -> str:
    s = str(path)
    hits = PLANET_INDEX_RE.findall(s)
    if not hits:
        raise ValueError(f"Nie umiem wyciągnąć planet_index z nazwy/ścieżki: {path}")
    return hits[-1]


def read_signal_table(file_path: str | Path) -> Dict[str, torch.Tensor]:
    file_path = Path(file_path)

    text = file_path.read_text(encoding="utf-8").strip()

    arr = np.fromstring(text, sep=",", dtype=np.float32)

    if arr.size == 0:
        arr = np.fromstring(text.replace("\n", " "), sep=" ", dtype=np.float32)

    if arr.size == 0:
        df = pd.read_csv(file_path, header=None)
        arr = df.to_numpy(dtype=np.float32).reshape(-1)

    if arr.size == 0:
        raise ValueError(f"Nie udało się odczytać sygnału z {file_path}")

    mask = np.isfinite(arr)
    arr = arr[mask]

    if arr.size == 0:
        raise ValueError(f"Po filtracji brak poprawnych punktów w {file_path}")

    wavelength = np.arange(arr.size, dtype=np.float32)

    return {
        "wavelength": torch.from_numpy(wavelength),
        "signal": torch.from_numpy(arr),
    }


def read_inara_parameters_tbl(
    path: str | Path, normalize_names: bool = False
) -> pd.DataFrame:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    lines = [ln.rstrip("\n") for ln in lines if ln.strip()]

    header_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("|") and ln.count("|") >= 3:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Nie znalazłem nagłówka '|' w pliku {path}")

    header = lines[header_idx]

    bar_pos = [i for i, ch in enumerate(header) if ch == "|"]

    colspecs = []
    names = []
    for a, b in zip(bar_pos[:-1], bar_pos[1:]):
        name = header[a + 1 : b].strip()
        if name:
            colspecs.append((a + 1, b))
            names.append(name)

    data_lines = []
    for ln in lines[header_idx + 1 :]:
        s = ln.strip()

        if not s:
            continue
        if s.startswith("|"):
            continue
        if set(s) <= {"-", " "}:
            continue

        data_lines.append(ln)

    text = "\n".join(data_lines)
    df = pd.read_fwf(StringIO(text), colspecs=colspecs, names=names)

    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()

    if "planet_index" in df.columns:
        df["planet_index"] = df["planet_index"].astype(str).str.strip().str.zfill(7)

    for c in df.columns:
        if c == "planet_index":
            continue

        if df[c].dtype == object:
            converted = pd.to_numeric(df[c], errors="coerce")

            if converted.notna().sum() > 0:
                df[c] = converted

    if normalize_names:
        rename_map = {
            "star_class_(F=3_G=4_K=5_M=6)": "star_class",
            "star_temperature_(Kelvin)": "star_temperature",
            "star_radius_(Solar_radii)": "star_radius",
            "distance_from_Earth_to_the_system_(parsec)": "distance_pc",
            "semimajor_axis_of_the_planet_(AU)": "planet_semimajor_axis_au",
            "planet_radius_(Earth_radii)": "planet_radius_earth",
            "planet_density_(g/cm3)": "planet_density_g_cm3",
            "planet_surface_pressure_(bar)": "planet_surface_pressure_bar",
            "planet_surface_temperature_(Kelvin)": "planet_surface_temperature",
            "planet_atmosphere's_avg_mol_wgt_(g/mol)": "planet_avg_mol_wgt",
            "planet's_mean_surface_albedo_(unitless)": "planet_surface_albedo",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    return df


def locate_parameters_tbl(raw_dir: Path, unpack_dir: Optional[Path] = None) -> Path:
    candidates = []
    if unpack_dir is not None:
        candidates.extend(sorted(unpack_dir.rglob("parameters.tbl")))
    candidates.extend(sorted(raw_dir.rglob("parameters.tbl")))
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Nie znaleziono parameters.tbl. Rozpakuj parameters.tar.gz albo podaj --summary-csv do parameters.tbl"
    )


def find_signal_files(search_root: Path) -> List[Path]:
    exts = {".csv", ".txt", ".dat", ".tbl"}
    files: List[Path] = []
    for p in search_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        low = p.name.lower()
        if low in {"parameters.tbl", "pyatmos_summary.csv", "pyatmos_models.csv"}:
            continue
        if "parameter" in low or "summary" in low:
            continue
        files.append(p)
    return sorted(files)


def convert_inara_planet_signal_to_pt(
    raw_dir: str | Path,
    processed_dir: str | Path,
    index_csv: str | Path,
    summary_csv: Optional[str | Path] = None,
    summary_id_col: str = "planet_index",
    target_cols: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    unpack_first: bool = True,
    unpack_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    unpack_dir_path: Optional[Path] = None
    if unpack_first:
        unpack_dir_path = (
            Path(unpack_dir) if unpack_dir is not None else raw_dir / "unpacked"
        )
        unpack_archives(raw_dir, unpack_dir_path)

    if summary_csv is None:
        summary_path = locate_parameters_tbl(raw_dir, unpack_dir_path)
    else:
        summary_path = Path(summary_csv)

    if summary_path.suffix.lower() == ".tbl":
        summary_df = read_inara_parameters_tbl(summary_path)
    else:
        summary_df = pd.read_csv(summary_path)

    if summary_id_col not in summary_df.columns:
        raise KeyError(
            f"Brakuje kolumny ID '{summary_id_col}' w summary. Dostępne: {list(summary_df.columns)[:10]}..."
        )
    summary_df[summary_id_col] = (
        summary_df[summary_id_col].astype(str).str.strip().str.zfill(7)
    )

    search_root = unpack_dir_path if unpack_dir_path is not None else raw_dir
    files = find_signal_files(search_root)
    if limit is not None:
        files = files[:limit]
    if not files:
        raise RuntimeError(f"Nie znaleziono żadnych plików sygnału w {search_root}")

    rows = []
    for file_path in files:
        try:
            sample = read_signal_table(file_path)
            sample_id = extract_planet_index_from_path(file_path)

            pt_path = processed_dir / f"{sample_id}.pt"
            torch.save(sample, pt_path)

            row = {
                "sample_id": sample_id,
                "pt_path": str(pt_path.resolve()),
                "raw_signal_path": str(file_path.resolve()),
                "n_points": int(sample["wavelength"].numel()),
                "has_noise": int("noise" in sample),
                "has_stellar_signal": int("stellar_signal" in sample),
            }

            hit = summary_df[summary_df[summary_id_col] == sample_id]
            if len(hit) == 1:
                hit_row = hit.iloc[0]
                if target_cols is None:
                    for c in summary_df.columns:
                        if c != summary_id_col:
                            row[c] = hit_row[c]
                else:
                    for c in target_cols:
                        if c in hit_row.index:
                            row[c] = hit_row[c]
            elif len(hit) > 1:
                raise ValueError(
                    f"Więcej niż jeden wpis summary dla sample_id={sample_id}"
                )
            else:
                row["missing_summary"] = 1

            rows.append(row)
            print(f"[PT] {file_path.name} -> {pt_path.name}")
        except Exception as e:
            print(f"[SKIP] {file_path}: {e}")

    index_df = pd.DataFrame(rows)
    if len(index_df) == 0:
        raise RuntimeError(
            "Nie udało się sparsować żadnego pliku sygnału. Sprawdź format plików po rozpakowaniu."
        )
    index_csv = Path(index_csv)
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(index_csv, index=False)
    return index_df


class InaraDataset(Dataset):
    def __init__(
        self,
        index_csv: str | Path,
        target_cols: Optional[Sequence[str]] = None,
        include_noise: bool = True,
        include_stellar_signal: bool = False,
        fixed_grid: Optional[Sequence[float]] = None,
        log_targets: bool = False,
        eps: float = 1e-12,
    ) -> None:
        self.df = pd.read_csv(index_csv)
        self.target_cols = list(target_cols) if target_cols is not None else None
        self.include_noise = include_noise
        self.include_stellar_signal = include_stellar_signal
        self.fixed_grid = (
            None if fixed_grid is None else np.asarray(fixed_grid, dtype=np.float32)
        )
        self.log_targets = log_targets
        self.eps = float(eps)

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _interp(x_old: np.ndarray, y_old: np.ndarray, x_new: np.ndarray) -> np.ndarray:
        return np.interp(x_new, x_old, y_old).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        row = self.df.iloc[idx]
        sample = torch.load(row["pt_path"], map_location="cpu")

        wavelength = sample["wavelength"].numpy().astype(np.float32)
        signal = sample["signal"].numpy().astype(np.float32)

        channels = []
        if self.fixed_grid is not None:
            w = self.fixed_grid
            signal = self._interp(wavelength, signal, w)
            channels.append(torch.from_numpy(signal))
        else:
            w = wavelength
            channels.append(sample["signal"].float())

        if self.include_noise and "noise" in sample:
            noise = sample["noise"].numpy().astype(np.float32)
            if self.fixed_grid is not None:
                noise = self._interp(wavelength, noise, w)
                channels.append(torch.from_numpy(noise))
            else:
                channels.append(sample["noise"].float())

        if self.include_stellar_signal and "stellar_signal" in sample:
            ss = sample["stellar_signal"].numpy().astype(np.float32)
            if self.fixed_grid is not None:
                ss = self._interp(wavelength, ss, w)
                channels.append(torch.from_numpy(ss))
            else:
                channels.append(sample["stellar_signal"].float())

        x = torch.stack(channels, dim=0)
        out: Dict[str, torch.Tensor | str] = {
            "x": x,
            "wavelength": torch.from_numpy(w)
            if isinstance(w, np.ndarray)
            else torch.tensor(w),
            "sample_id": str(row["sample_id"]),
        }

        if self.target_cols is not None:
            vals = []
            for c in self.target_cols:
                if c not in row.index:
                    raise KeyError(f"Brakuje target_col='{c}' w index.csv")
                v = float(row[c])
                if self.log_targets:
                    v = np.log10(max(v, self.eps))
                vals.append(v)
            out["y"] = torch.tensor(vals, dtype=torch.float32)

        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download")
    p_dl.add_argument("--wget-script", type=str, required=True)
    p_dl.add_argument("--out-dir", type=str, required=True)
    p_dl.add_argument("--workers", type=int, default=4)
    p_dl.add_argument("--limit", type=int, default=None)
    p_dl.add_argument("--overwrite", action="store_true")

    p_unpack = sub.add_parser("unpack")
    p_unpack.add_argument("--raw-dir", type=str, required=True)
    p_unpack.add_argument("--unpack-dir", type=str, default=None)

    p_conv = sub.add_parser("convert")
    p_conv.add_argument("--raw-dir", type=str, required=True)
    p_conv.add_argument("--processed-dir", type=str, required=True)
    p_conv.add_argument("--index-csv", type=str, required=True)
    p_conv.add_argument("--summary-csv", type=str, default=None)
    p_conv.add_argument("--summary-id-col", type=str, default="planet_index")
    p_conv.add_argument("--target-cols", type=str, default=None)
    p_conv.add_argument("--limit", type=int, default=None)
    p_conv.add_argument("--no-unpack", action="store_true")
    p_conv.add_argument("--unpack-dir", type=str, default=None)

    args = parser.parse_args()

    if args.cmd == "download":
        download_from_wget_script(
            wget_script=args.wget_script,
            out_dir=args.out_dir,
            workers=args.workers,
            limit=args.limit,
            overwrite=args.overwrite,
        )
    elif args.cmd == "unpack":
        unpack_archives(args.raw_dir, args.unpack_dir)
    elif args.cmd == "convert":
        target_cols = None
        if args.target_cols:
            target_cols = [c.strip() for c in args.target_cols.split(",") if c.strip()]

        convert_inara_planet_signal_to_pt(
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            index_csv=args.index_csv,
            summary_csv=args.summary_csv,
            summary_id_col=args.summary_id_col,
            target_cols=target_cols,
            limit=args.limit,
            unpack_first=not args.no_unpack,
            unpack_dir=args.unpack_dir,
        )


if __name__ == "__main__":
    main()
