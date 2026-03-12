#!/bin/sh
set -eu

PYTHON="${PYTHON:-python}"
TOOLS="${TOOLS:-./download.py}"
WGET_SCRIPT="${WGET_SCRIPT:-./build_inara.sh}"
WORKERS="${WORKERS:-4}"
LIMIT="${LIMIT:-}"

ROOT="${1:-./data/inara}"
RAW_DIR="${ROOT}/raw"
UNPACK_DIR="${RAW_DIR}/unpacked"
PROCESSED_DIR="${ROOT}/processed"
INDEX_CSV="${ROOT}/index.csv"

echo "[INFO] ROOT          = ${ROOT}"
echo "[INFO] RAW_DIR       = ${RAW_DIR}"
echo "[INFO] UNPACK_DIR    = ${UNPACK_DIR}"
echo "[INFO] PROCESSED_DIR = ${PROCESSED_DIR}"
echo "[INFO] INDEX_CSV     = ${INDEX_CSV}"
echo "[INFO] WGET_SCRIPT   = ${WGET_SCRIPT}"

if [ ! -f "${TOOLS}" ]; then
  echo "[ERR] File not found: ${TOOLS}" >&2
  exit 1
fi

if [ ! -f "${WGET_SCRIPT}" ]; then
  echo "[ERR] File not found: ${WGET_SCRIPT}" >&2
  exit 1
fi

mkdir -p "${RAW_DIR}" "${UNPACK_DIR}" "${PROCESSED_DIR}"

echo "[STEP 1/3] Download"
if [ -n "${LIMIT}" ]; then
  "${PYTHON}" "${TOOLS}" download \
    --wget-script "${WGET_SCRIPT}" \
    --out-dir "${RAW_DIR}" \
    --workers "${WORKERS}" \
    --limit "${LIMIT}"
else
  "${PYTHON}" "${TOOLS}" download \
    --wget-script "${WGET_SCRIPT}" \
    --out-dir "${RAW_DIR}" \
    --workers "${WORKERS}"
fi

echo "[STEP 2/3] Unpack"
"${PYTHON}" "${TOOLS}" unpack \
  --raw-dir "${RAW_DIR}" \
  --unpack-dir "${UNPACK_DIR}"

echo "[STEP 3/3] Convert"
if [ -n "${LIMIT}" ]; then
  "${PYTHON}" "${TOOLS}" convert \
    --raw-dir "${RAW_DIR}" \
    --unpack-dir "${UNPACK_DIR}" \
    --no-unpack \
    --processed-dir "${PROCESSED_DIR}" \
    --index-csv "${INDEX_CSV}" \
    --limit "${LIMIT}"
else
  "${PYTHON}" "${TOOLS}" convert \
    --raw-dir "${RAW_DIR}" \
    --unpack-dir "${UNPACK_DIR}" \
    --no-unpack \
    --processed-dir "${PROCESSED_DIR}" \
    --index-csv "${INDEX_CSV}"
fi

echo "[DONE] Ready"
echo "[DONE] Index: ${INDEX_CSV}"
echo "[DONE] Processed dir: ${PROCESSED_DIR}"