# exp-kvarn — KVarN KV cache quantization cherry-pick mod

把 [huawei-csl/KVarN](https://github.com/huawei-csl/KVarN) 的 KV cache backend 套進你 build 的 vLLM（保留 124 commits 的 GB10 b12x MoE / GDN Blackwell prefill / FlashInfer 0.6.12）。

## 何時用

- ✅ 你跑 `VLLM_SPEC=none` 多用戶 aggregate / 長 context 場景
- ✅ 想拿 5× KV 容量 + 1.3× FP16 throughput（KVarN [paper claim](https://arxiv.org/html/2606.03458v1)）
- ❌ DFlash 路徑用不到（[#41559](https://github.com/vllm-project/vllm/issues/41559) 擋）
- ⚠️ Hybrid GDN（Qwen3.6/Qwen3-Next/Qwen3.5）KVarN 作者未驗，有 [#41560](https://github.com/vllm-project/vllm/issues/41560) 同款風險

## 用法

### 啟用流程

```bash
# 1. 先 dry-run 看 KVarN 會改哪些檔（不動 installed vLLM）
./launch-cluster.sh \
    --apply-mod ./mods/exp-kvarn \
    -e KVARN_DRY_RUN=1 \
    ... 其他你原本的 flag

# 2. 確認 conflict 數量可接受後，正式套用
./launch-cluster.sh \
    --apply-mod ./mods/exp-kvarn \
    ... 其他你原本的 flag

# 3. 在 vllm serve 帶上新 dtype
# 你 server.py 的 SPEC=none 分支改：
#   --kv-cache-dtype kvarn_k4v2_g128
```

### 可調 env

| Env | 預設 | 說明 |
|---|---|---|
| `KVARN_REPO` | `https://github.com/huawei-csl/KVarN.git` | KVarN repo URL |
| `KVARN_REF` | `main` | 建議 pin 特定 commit（reproducibility） |
| `KVARN_BASELINE` | `v0.22.0` | KVarN fork 的 vLLM baseline tag |
| `KVARN_DRY_RUN` | `0` | =1 只報告不改動 |

### Reproducibility 範例

```bash
# Pin KVarN 到 release 日 commit（建議）
./launch-cluster.sh \
    --apply-mod ./mods/exp-kvarn \
    -e KVARN_REF=<SPECIFIC_COMMIT_SHA> \
    ...
```

## 預期輸出

成功：

```
[kvarn] ═══════════════════════════════════════════════════════
[kvarn] SUMMARY
[kvarn]   KVarN commit:     abc1234
[kvarn]   vLLM baseline:    v0.22.0
[kvarn]   Files added:      8 / 8
[kvarn]   Files patched:    3 / 3
[kvarn]   Conflicts:        0
[kvarn] ═══════════════════════════════════════════════════════
[kvarn] ✓ All KVarN changes applied successfully
```

衝突（典型情境，你 build 的 124 commits 改過同檔）：

```
[kvarn] ✗ vllm/v1/worker/gpu_model_runner.py conflict; left untouched
...
[kvarn] ⚠️  Conflicts on these files (manual integration needed):
       vllm/v1/worker/gpu_model_runner.py
         → diff saved at /workspace/kvarn-work/CONFLICT-gpu_model_runner.py.diff
```

衝突檔通常是這三類，手動套用 5-30 分鐘可解：

1. **`vllm/v1/kv_cache_interface.py`** — dtype enum 加 `kvarn_k4v2_g128`
2. **`vllm/engine/arg_utils.py`** — CLI `--kv-cache-dtype` choices 加入
3. **`vllm/v1/worker/gpu_model_runner.py`** — backend dispatch logic

衝突 diff 留在 `$WORKSPACE_DIR/kvarn-work/CONFLICT-*.diff` 給你手動 inspect。

## 驗證步驟

套用後驗：

```bash
# 1. 模組能 import
docker exec vllm-server python3 -c "from vllm.v1.attention.backends import kvarn; print('OK')"

# 2. CLI 認新 dtype
docker exec vllm-server vllm serve --help | grep kvarn

# 3. 實際啟動測試（用 VLLM_SPEC=none，最小 context）
VLLM_SPEC=none docker compose up vllm-server  # 看啟動 log 有無 KVarN backend 字樣

# 4. 跑你 production typical agent prompt 比品質
```

## Rollback

```bash
# KVarN mod 不改 installed wheel，只改 runtime files
# 重建 container 即可清除：
docker compose down
docker compose up -d vllm-server  # 從 image 重新啟動，KVarN patch 沒了
```

## 已知風險（讀過再用）

1. **Qwen3.6 hybrid GDN 未驗證**：KVarN paper 只測 dense Qwen3-32B。可能像 TurboQuant 初版一樣在 hybrid page-size alignment 上炸（[#41560](https://github.com/vllm-project/vllm/issues/41560)）。如果啟動 assertion 失敗，**回退 TurboQuant**（已 [PR #39931](https://github.com/vllm-project/vllm/pull/39931) 修了 hybrid）。

2. **DFlash 不相容**：[#41559](https://github.com/vllm-project/vllm/issues/41559) 是 backend 層級的 non-causal attention vs KV quant 衝突，**所有 KV quant 包括 KVarN 都中**。等 [PR #43081](https://github.com/vllm-project/vllm/pull/43081) merge。

3. **MTP 未驗證**：[#40880](https://github.com/vllm-project/vllm/issues/40880) 已 closed for TurboQuant，但 KVarN 是新方法，CUDA graph capture 互動沒人測過。

4. **124 commits ahead 衝突風險**：你 build 是 0.22.1rc1.dev124，KVarN base 是 0.22.0。任何被 124 commits 改過的檔都可能衝突 — script 會偵測並列出。

5. **Hadamard rotation 對 GQA-8 head_dim=128 假設**：KVarN paper 用 Qwen3-32B（也是 GQA），head_dim 應該相同 — 但沒明說過 32B → 35B-A3B MoE 的差異。

## 退路選項

依「保險程度」由高到低排序：

| 退路 | 收益 | 工程成本 |
|---|---|---|
| `--kv-cache-dtype turboquant_4bit_nc` | 3.8× 容量 | 0（主線已有） |
| `--kv-cache-dtype fp8` (e4m3) | 2× 容量 | 0（主線已有） |
| `--kv-cache-dtype auto` (BF16) | baseline | 0 |

KVarN 失敗就用上面三個之一。
