#!/usr/bin/env bash
# Offline TPU v7x (`tpu7x:2x2x1`) AOT compilation & Final LLO bundle verification
# using local Docker (`linux/amd64`) + `libtpu` (no physical TPU needed).
#
# Usage:
#   ./validate/run_local_libtpu_docker.sh
#
# Optional env vars:
#   IMAGE   : Docker image with jax + libtpu (default: us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.9.0-rev1,
#             or python:3.12-slim when WHEELS is set or pip install jax[tpu] is used)
#   WHEELS  : Optional local directory of pre-downloaded linux/amd64 wheels (jax, jaxlib, libtpu)
set -euo pipefail
cd "$(dirname "$0")"

SGL=${SGLANG_TPU_ROOT:-../sglang-torchtpu}
SGL_JAX=${SGLANG_JAX_ROOT:-../sglang-jax}
MXT=${MAXTEXT_ROOT:-../maxtext}
OUT_DIR=${OUT_DIR:-$(pwd)/aot_v7x_out}

rm -rf bundle && mkdir -p \
  bundle/src/sglang_tpu/kernels \
  bundle/src/src/MaxText/kernels/attention \
  bundle/src/src/MaxText/kernels/megablox \
  bundle/src/sgl_jax \
  bundle/frontend

cp "$SGL/sglang_tpu/kernels/flash_attention.py" bundle/src/sglang_tpu/kernels/
cp "$SGL/sglang_tpu/kv_page_write.py" bundle/src/sglang_tpu/
cp "$SGL/sglang_tpu/qk_norm_rope.py" bundle/src/sglang_tpu/
cp "$SGL/sglang_tpu/kimi_attn_residual.py" bundle/src/sglang_tpu/
cp "$MXT/src/MaxText/kernels/attention/ragged_attention.py" bundle/src/src/MaxText/kernels/attention/
cp "$MXT/src/MaxText/kernels/megablox/backend.py" bundle/src/src/MaxText/kernels/megablox/
cp "$SGL_JAX/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/common.py" bundle/src/sgl_jax/
cp "$SGL_JAX/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py" bundle/src/sgl_jax/
cp "$SGL_JAX/python/sgl_jax/srt/kernels/update_kv_cache/update_kv_cache.py" bundle/src/sgl_jax/
cp -R ../frontend/tpuverify bundle/frontend/
cp aot_compile_v7x.py bundle/

export PATH="/usr/local/bin:/opt/homebrew/bin:$HOME/.orbstack/bin:$HOME/.docker/bin:/Applications/Docker.app/Contents/Resources/bin:/Applications/OrbStack.app/Contents/MacOS/xbin:$PATH"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: 'docker' binary not found in PATH." >&2
  echo "Install/start OrbStack ('brew install --cask orbstack') or Docker Desktop / Colima ('brew install docker colima && colima start --arch x86_64'), then re-run this script." >&2
  exit 1
fi

if [[ -n "${WHEELS:-}" ]]; then
  IMAGE=${IMAGE:-python:3.12-slim}
  docker run --rm --platform linux/amd64 \
    -e JAX_PLATFORMS=cpu \
    -v "$(pwd)/bundle:/job" \
    -v "$(cd "$WHEELS" && pwd):/wheels:ro" \
    -v "$OUT_DIR:/out" \
    "$IMAGE" sh -c '
      python3 -m pip install --no-index --find-links=/wheels jax jaxlib libtpu numpy &&
      PYTHONPATH=/job/frontend python3 /job/aot_compile_v7x.py --src /job/src --out /out --target tpu7x:2x2x1
    '
elif [[ -n "${IMAGE:-}" ]]; then
  docker run --rm --platform linux/amd64 \
    -e JAX_PLATFORMS=cpu \
    -v "$(pwd)/bundle:/job" \
    -v "$OUT_DIR:/out" \
    "$IMAGE" sh -c '
      PYTHONPATH=/job/frontend python3 /job/aot_compile_v7x.py --src /job/src --out /out --target tpu7x:2x2x1
    '
else
  # Lightweight path: python:3.12-slim + cached pip venv with jax[tpu] (linux/amd64 libtpu.so)
  mkdir -p "$(pwd)/.docker_tpu_venv"
  PROXY_ARGS=()
  if [[ -n "${https_proxy:-${HTTPS_PROXY:-}}" ]]; then
    P_VAL="${https_proxy:-${HTTPS_PROXY}}"
    P_VAL="${P_VAL/127.0.0.1/192.168.5.2}"
    P_VAL="${P_VAL/localhost/192.168.5.2}"
    PROXY_ARGS+=(-e "http_proxy=$P_VAL" -e "https_proxy=$P_VAL" -e "HTTP_PROXY=$P_VAL" -e "HTTPS_PROXY=$P_VAL")
  fi
  docker run --rm --platform linux/amd64 \
    -e JAX_PLATFORMS=cpu \
    ${PROXY_ARGS[@]+"${PROXY_ARGS[@]}"} \
    -v "$(pwd)/bundle:/job" \
    -v "$(pwd)/.docker_tpu_venv:/opt/tpuvenv" \
    -v "$OUT_DIR:/out" \
    python:3.12-slim sh -c '
      if [ ! -x /opt/tpuvenv/bin/python ]; then
        python3 -m venv /opt/tpuvenv &&
        /opt/tpuvenv/bin/pip install --upgrade pip &&
        /opt/tpuvenv/bin/pip install "jax[tpu]==0.9.0.1" numpy -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
      fi
      PYTHONPATH=/job/frontend /opt/tpuvenv/bin/python /job/aot_compile_v7x.py --src /job/src --out /out --target tpu7x:2x2x1
    '
fi

echo "TPU v7x AOT LLO verification results saved to: $OUT_DIR/aot_v7x_results.json"
