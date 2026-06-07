#!/bin/bash
#
# EXPERIMENTAL — KVarN KV cache quantization backend cherry-pick
#
# 把 huawei-csl/KVarN 的 KV cache backend cherry-pick 進已安裝的 vLLM
# （目標：你 build 的 0.22.1rc1.dev124 + GB10 b12x / GDN Blackwell / FlashInfer 0.6.12）
#
# KVarN 機制：Hadamard rotation + iterative variance normalization + 4-bit K / 2-bit V
# 啟用方式：--kv-cache-dtype kvarn_k4v2_g128
#
# 已知限制：
#   - KVarN paper 只測 dense Qwen3-32B，hybrid GDN (Qwen3.6) 未驗證
#   - 跟 DFlash 不相容（同 #41559，等 PR #43081）
#   - 跟 MTP 互動未驗證
#   - 在 VLLM_SPEC=none 模式比 fp8 / turboquant_4bit_nc 更激進
#
# Reference: https://github.com/huawei-csl/KVarN
#
set -e

SITE_PACKAGES="/usr/local/lib/python3.12/dist-packages"
VLLM_ROOT="$SITE_PACKAGES/vllm"
KVARN_REPO="${KVARN_REPO:-https://github.com/huawei-csl/KVarN.git}"
KVARN_REF="${KVARN_REF:-main}"                     # 建議 pin 到特定 commit
KVARN_BASELINE="${KVARN_BASELINE:-v0.22.0}"        # KVarN fork 自此 tag
WORK_DIR="${WORKSPACE_DIR:-/tmp}/kvarn-work"
DRY_RUN="${KVARN_DRY_RUN:-0}"                      # =1 只報告不改動

echo "=== EXPERIMENTAL KVarN KV cache backend mod ==="
echo "[kvarn] target vLLM root: $VLLM_ROOT"
echo "[kvarn] KVarN repo:       $KVARN_REPO @ $KVARN_REF"
echo "[kvarn] baseline vLLM:    $KVARN_BASELINE"
echo "[kvarn] dry-run:          $DRY_RUN"

# ─── 0. 前置檢查 ────────────────────────────────────────────────
if [ ! -d "$VLLM_ROOT" ]; then
    echo "[kvarn ERROR] vLLM not installed at $VLLM_ROOT"
    exit 1
fi

INSTALLED_VLLM_VERSION=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
echo "[kvarn] installed vLLM version: $INSTALLED_VLLM_VERSION"

if ! command -v git &> /dev/null; then
    echo "[kvarn ERROR] git not found in container"
    exit 1
fi

# ─── 1. 取得 KVarN repo ─────────────────────────────────────────
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

if [ ! -d "kvarn" ]; then
    echo "[kvarn] cloning $KVARN_REPO ..."
    git clone --quiet "$KVARN_REPO" kvarn
else
    echo "[kvarn] updating existing clone"
    (cd kvarn && git fetch --quiet --all)
fi

cd kvarn
git checkout --quiet "$KVARN_REF"
KVARN_COMMIT=$(git rev-parse --short HEAD)
echo "[kvarn] checked out KVarN @ $KVARN_COMMIT"

# 加 upstream vLLM remote 取 baseline tag
git remote add upstream https://github.com/vllm-project/vllm.git 2>/dev/null || true
echo "[kvarn] fetching upstream $KVARN_BASELINE ..."
if ! git fetch --quiet upstream "$KVARN_BASELINE" 2>/dev/null; then
    echo "[kvarn] tag fetch failed, trying full upstream fetch"
    git fetch --quiet upstream
fi

# 解析 baseline ref
BASELINE_REF=$(git rev-parse "upstream/$KVARN_BASELINE" 2>/dev/null \
            || git rev-parse "$KVARN_BASELINE" 2>/dev/null \
            || git rev-parse FETCH_HEAD)
echo "[kvarn] baseline commit: $(git rev-parse --short "$BASELINE_REF")"

# ─── 2. 列出 KVarN 對 vLLM 改了什麼 ──────────────────────────────
echo ""
echo "[kvarn] === KVarN modifications to vllm/ vs upstream $KVARN_BASELINE ==="
git diff "$BASELINE_REF" --stat -- 'vllm/**' 2>/dev/null \
    | grep -v '^ tests/' \
    | tail -30
echo ""

# ─── 3. 分類：NEW vs MODIFIED ──────────────────────────────────
NEW_FILES=$(git diff "$BASELINE_REF" --diff-filter=A --name-only -- 'vllm/**' \
            | grep -v '^tests/' | grep -v '^examples/' || true)
MOD_FILES=$(git diff "$BASELINE_REF" --diff-filter=M --name-only -- 'vllm/**' \
            | grep -v '^tests/' | grep -v '^examples/' || true)
DEL_FILES=$(git diff "$BASELINE_REF" --diff-filter=D --name-only -- 'vllm/**' \
            | grep -v '^tests/' | grep -v '^examples/' || true)

NEW_COUNT=$(echo "$NEW_FILES" | grep -c . || echo 0)
MOD_COUNT=$(echo "$MOD_FILES" | grep -c . || echo 0)
DEL_COUNT=$(echo "$DEL_FILES" | grep -c . || echo 0)

echo "[kvarn] file classification: $NEW_COUNT new, $MOD_COUNT modified, $DEL_COUNT deleted"

if [ "$DEL_COUNT" -gt 0 ]; then
    echo "[kvarn WARN] KVarN deletes $DEL_COUNT vLLM files — manual review required:"
    echo "$DEL_FILES" | sed 's/^/    /'
fi

# ─── 4. Dry-run: 報告完就退出 ───────────────────────────────────
if [ "$DRY_RUN" = "1" ]; then
    echo ""
    echo "[kvarn] === DRY RUN — files that WOULD be added ==="
    echo "$NEW_FILES" | head -20 | sed 's/^/  + /'
    echo ""
    echo "[kvarn] === DRY RUN — files that WOULD be patched ==="
    echo "$MOD_FILES" | head -20 | sed 's/^/  ~ /'
    echo ""
    echo "[kvarn] dry-run complete; no changes applied"
    echo "[kvarn] re-run with KVARN_DRY_RUN=0 to apply"
    exit 0
fi

# ─── 5. 套用 NEW files（零衝突風險）─────────────────────────────
echo ""
echo "[kvarn] === Phase 1/3: copying NEW files ==="
NEW_APPLIED=0
NEW_FAILED=0
if [ -n "$NEW_FILES" ]; then
    while IFS= read -r f; do
        [ -z "$f" ] && continue
        target="$SITE_PACKAGES/$f"
        if [ -e "$target" ]; then
            echo "  ! $f already exists in installed vLLM (was added by 124 commits?), skipping"
            NEW_FAILED=$((NEW_FAILED + 1))
            continue
        fi
        mkdir -p "$(dirname "$target")"
        cp "$f" "$target"
        echo "  + $f"
        NEW_APPLIED=$((NEW_APPLIED + 1))
    done <<< "$NEW_FILES"
fi
echo "[kvarn] phase 1: $NEW_APPLIED added, $NEW_FAILED collisions"

# ─── 6. 套用 MODIFIED files via git apply --3way ────────────────
echo ""
echo "[kvarn] === Phase 2/3: patching MODIFIED files (3-way merge) ==="
MOD_APPLIED=0
MOD_FAILED=0
FAILED_FILES=""

if [ -n "$MOD_FILES" ]; then
    while IFS= read -r f; do
        [ -z "$f" ] && continue
        target="$SITE_PACKAGES/$f"
        if [ ! -f "$target" ]; then
            echo "  ! $f doesn't exist in installed vLLM (renamed?), skipping"
            MOD_FAILED=$((MOD_FAILED + 1))
            FAILED_FILES="$FAILED_FILES $f"
            continue
        fi

        SINGLE_DIFF="$WORK_DIR/patch-$(echo "$f" | tr / _).diff"
        git diff "$BASELINE_REF" -- "$f" > "$SINGLE_DIFF"

        # 嘗試套用 (cd 到 SITE_PACKAGES 讓 vllm/foo/bar.py 路徑解析正確)
        if (cd "$SITE_PACKAGES" && git apply --check "$SINGLE_DIFF" 2>/dev/null); then
            (cd "$SITE_PACKAGES" && git apply "$SINGLE_DIFF")
            echo "  ~ $f"
            MOD_APPLIED=$((MOD_APPLIED + 1))
        elif (cd "$SITE_PACKAGES" && git apply --3way --check "$SINGLE_DIFF" 2>/dev/null); then
            (cd "$SITE_PACKAGES" && git apply --3way "$SINGLE_DIFF") || true
            echo "  ~ $f (3-way merge — verify manually)"
            MOD_APPLIED=$((MOD_APPLIED + 1))
        else
            echo "  ✗ $f conflict; left untouched"
            MOD_FAILED=$((MOD_FAILED + 1))
            FAILED_FILES="$FAILED_FILES $f"
            # 留下 diff 給 user 手動處理
            cp "$SINGLE_DIFF" "$WORK_DIR/CONFLICT-$(basename "$f").diff"
        fi
    done <<< "$MOD_FILES"
fi
echo "[kvarn] phase 2: $MOD_APPLIED patched, $MOD_FAILED conflicts"

# ─── 7. 清 pycache ──────────────────────────────────────────────
echo ""
echo "[kvarn] === Phase 3/3: clear pycache ==="
find "$VLLM_ROOT" -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "[kvarn] pycache cleared"

# ─── 8. 驗證可 import ───────────────────────────────────────────
echo ""
echo "[kvarn] === Validation: can KVarN backend be imported? ==="

# 找出 KVarN 新增的 attention backend 檔（多半在 vllm/v1/attention/backends/ 或類似位置）
KVARN_BACKEND_MODULE=$(echo "$NEW_FILES" | grep -E "kvarn|backends/.*kv" | head -1 | sed 's|^|vllm/|' | sed 's|/|.|g' | sed 's|\.py$||' || true)

if [ -n "$KVARN_BACKEND_MODULE" ]; then
    echo "[kvarn] attempting: import $KVARN_BACKEND_MODULE"
    if python3 -c "import importlib; importlib.import_module('$KVARN_BACKEND_MODULE')" 2>&1; then
        echo "[kvarn] ✓ KVarN backend module imports cleanly"
    else
        echo "[kvarn ERROR] import failed — check conflicts above"
    fi
else
    echo "[kvarn WARN] couldn't auto-detect KVarN backend module path"
    echo "[kvarn WARN] manual check: ls $VLLM_ROOT/v1/attention/backends/ | grep -i kv"
fi

# 驗證 CLI 認 kv-cache-dtype
echo ""
echo "[kvarn] verifying --kv-cache-dtype kvarn_k4v2_g128 is recognized..."
if python3 -c "
from vllm.engine.arg_utils import EngineArgs
import argparse
parser = argparse.ArgumentParser()
EngineArgs.add_cli_args(parser)
args = parser.parse_args(['--model', 'dummy', '--kv-cache-dtype', 'kvarn_k4v2_g128'])
print(f'[kvarn] kv_cache_dtype parsed as: {args.kv_cache_dtype}')
" 2>&1; then
    echo "[kvarn] ✓ CLI accepts --kv-cache-dtype kvarn_k4v2_g128"
else
    echo "[kvarn WARN] CLI rejected the new dtype — registration may be missing"
fi

# ─── 9. Summary ─────────────────────────────────────────────────
echo ""
echo "[kvarn] ═══════════════════════════════════════════════════════"
echo "[kvarn] SUMMARY"
echo "[kvarn]   KVarN commit:     $KVARN_COMMIT"
echo "[kvarn]   vLLM baseline:    $KVARN_BASELINE"
echo "[kvarn]   Files added:      $NEW_APPLIED / $NEW_COUNT"
echo "[kvarn]   Files patched:    $MOD_APPLIED / $MOD_COUNT"
echo "[kvarn]   Conflicts:        $((NEW_FAILED + MOD_FAILED))"
echo "[kvarn] ═══════════════════════════════════════════════════════"

if [ "$MOD_FAILED" -gt 0 ] || [ "$NEW_FAILED" -gt 0 ]; then
    echo ""
    echo "[kvarn] ⚠️  Conflicts on these files (manual integration needed):"
    for f in $FAILED_FILES; do
        echo "       $f"
        if [ -f "$WORK_DIR/CONFLICT-$(basename "$f").diff" ]; then
            echo "         → diff saved at $WORK_DIR/CONFLICT-$(basename "$f").diff"
        fi
    done
    echo ""
    echo "[kvarn] common conflict targets (你 build 的 124 commits 大概也改了)："
    echo "       - vllm/v1/kv_cache_interface.py  (dtype 註冊)"
    echo "       - vllm/engine/arg_utils.py       (CLI flag)"
    echo "       - vllm/v1/worker/gpu_model_runner.py (dispatch)"
    echo ""
    echo "[kvarn] 後續：手動 inspect $WORK_DIR/CONFLICT-*.diff 套用即可"
    exit 2
fi

echo ""
echo "[kvarn] ✓ All KVarN changes applied successfully"
echo "[kvarn] Usage: vllm serve ... --kv-cache-dtype kvarn_k4v2_g128"
echo ""
echo "[kvarn] ⚠️ Compatibility caveats (重要)："
echo "[kvarn]   - Hybrid GDN (Qwen3.6) 未經 KVarN 作者驗證 — 同 TurboQuant 早期 #41560 風險"
echo "[kvarn]   - DFlash 仍被 #41559 擋（KVarN 改不了這個）"
echo "[kvarn]   - 建議先在 VLLM_SPEC=none + max_num_seqs 高並發場景試"
echo "[kvarn]   - fallback：拔掉 --kv-cache-dtype kvarn_k4v2_g128 改成 turboquant_4bit_nc"
