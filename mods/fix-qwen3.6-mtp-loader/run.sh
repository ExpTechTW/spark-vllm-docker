#!/bin/bash
#
# Fix Qwen3.6 MTP loader KeyError 'layers.0.mlp.experts.w2_weight'
#
# 套用 vllm-project/vllm PR #39475 — Qwen3_5MTP 在 moe_wna16/AutoRound 量化下
# inherits 主模型 quant_config，導致 FusedMoE 找 w13_qweight 但 checkpoint 是 w13_weight
# (BF16)。Fix: 把 MTP prefix 加進 modules_to_not_convert。
#
# Upstream: https://github.com/vllm-project/vllm/pull/39475
# Issue:    https://github.com/vllm-project/vllm/issues/36954
#
# 適用 stack:
#   - Qwen3.5/Qwen3.6 family
#   - INT4 AutoRound / GPTQ-Int4 / moe_wna16
#   - VLLM_SPEC=mtp 啟用時
#
set -euo pipefail

PYTHON_ROOT="/usr/local/lib/python3.12/dist-packages"
VLLM_ROOT="$PYTHON_ROOT/vllm"
PR="39475"

if [ ! -d "$VLLM_ROOT" ]; then
    echo "[mtp-fix] vLLM not installed at $VLLM_ROOT"
    exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

cd "$PYTHON_ROOT"

DIFF="$TMP_DIR/pr-${PR}.diff"
CHECK_LOG="$TMP_DIR/pr-${PR}.check.log"

echo "[mtp-fix] downloading PR #${PR} diff..."
curl -fsSL "https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/${PR}.diff" \
    -o "$DIFF"

# 檢查是否已套用（reverse 能 apply 表示已在 tree 內）
if git apply --reverse --check --exclude="tests/*" --exclude="examples/*" "$DIFF" 2>/dev/null; then
    echo "[mtp-fix] PR #${PR} already present in installed vLLM; skipping"
    exit 0
fi

# 嘗試正向 apply
echo "[mtp-fix] applying PR #${PR}..."
if git apply --check --exclude="tests/*" --exclude="examples/*" "$DIFF" 2>"$CHECK_LOG"; then
    git apply --exclude="tests/*" --exclude="examples/*" "$DIFF"
    echo "[mtp-fix] PR #${PR} applied successfully"
else
    echo "[mtp-fix] standard apply failed, trying 3-way merge..."
    if git apply --3way --check --exclude="tests/*" --exclude="examples/*" "$DIFF" 2>>"$CHECK_LOG"; then
        git apply --3way --exclude="tests/*" --exclude="examples/*" "$DIFF"
        echo "[mtp-fix] PR #${PR} applied via 3-way merge (verify behavior)"
    else
        echo "[mtp-fix] PR #${PR} could not be applied"
        cat "$CHECK_LOG"
        exit 1
    fi
fi

# 清 pycache 讓 patch 生效
find "$VLLM_ROOT" -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# 驗證：load qwen3_5_mtp.py 不會炸 + 確認包含修補關鍵字
TARGET="$VLLM_ROOT/model_executor/models/qwen3_5_mtp.py"
if [ -f "$TARGET" ]; then
    if grep -q "modules_to_not_convert" "$TARGET" 2>/dev/null; then
        echo "[mtp-fix] ✓ verified: modules_to_not_convert hook present in qwen3_5_mtp.py"
    else
        echo "[mtp-fix] ⚠️  patch applied but expected hook not found — manual inspect:"
        echo "         $TARGET"
    fi

    if python3 -c "from vllm.model_executor.models import qwen3_5_mtp" 2>&1; then
        echo "[mtp-fix] ✓ qwen3_5_mtp module imports cleanly"
    else
        echo "[mtp-fix] ⚠️  import test failed"
        exit 1
    fi
else
    echo "[mtp-fix] ⚠️  $TARGET not found"
fi

echo ""
echo "[mtp-fix] Done."
echo "[mtp-fix] You can now use VLLM_SPEC=mtp with INT4 AutoRound + moe_wna16."
echo "[mtp-fix] Recommended pair: --kv-cache-dtype turboquant_4bit_nc (主線最穩 hybrid GDN)"
