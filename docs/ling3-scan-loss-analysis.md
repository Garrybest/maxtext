# Ling3-Tiny Scan 模式 Loss 偏差分析

> 现象：开启 `scan_layers=true` 后，ling3-tiny 训练 loss 比 unscan 模式高 / 收敛差。
> 状态：**已通过运行时 dump + ckpt 实测确认根因**（2026-05-12 更新）。
> 作者：Claude（基于 branch `chore/ling3-submit-helpers-and-pallas-bump`）

---

## 0. 实测根因总结（2026-05-12 验证）

### 0.1 一句话结论

`scan_layers=true` 下，ling3 的 **23 个 backbone MoE 层全部** 在 trainer 侧失效：
- `moe_z_loss` 不计入总 loss（`backbone_sum=0`，只剩 1 个 MTP 层贡献）；
- `routed_bias`（loss-free balancing）**完全不更新**，永远是初始值。

unscan 没有此问题。差异随训练步数累积，与"scan loss 偏高/收敛差"现象吻合。

> 注：原 §2 推测"只有 3 个 Phase 1b prefix 层丢失"是错的——实际 Phase 2 scan 区域 也整段丢失（更严重），原因是路径假设和真实结构不匹配。下文修正。

### 0.2 三处错位 bug（`src/maxtext/trainers/pre_train/train.py`）

| # | 函数 / 位置 | 错误路径 | 真实路径 | 后果 |
|---|---|---|---|---|
| ① | `_collect_moe_intermediate_sum` (L131-148) | `intermediates/decoder/moe_layers/{key}` | `intermediates/decoder/moe_layers/layers_{0..3}/{key}` 是 dict | 落到 default `0.0`，backbone z_loss = 0 |
| ② | `moe_expert_counts` 收集 (L477-499) | 同上路径 | 同上 | 返回 `None`，`_apply_moe_bias_updates` 跳过 backbone 分支 |
| ③ | `_update_deepseek_bias` scan 分支 (L222-232) | `params/decoder/moe_layers/mlp/MoeBlock_0/gate/bias` | `params/decoder/moe_layers/layers_{0..3}/mlp/MoeBlock_0/gate/bias` | `has_nested_key` 返回 False，静默跳过 |

另：以上三处 scan 分支 **完全没有处理 Phase 1b prefix**（`moe_layers_{0,1,2}`），即使把 Phase 2 路径修对也还会丢这 3 层。

### 0.3 实测内存布局（关键证据）

intermediates 树（实测，`tools/dev/dump_ling3_intermediates.py` 在 Pod TPU 上）：
```
intermediates/decoder/moe_layers_0/{moe_z_loss, moe_expert_counts}        # 标量元组 / (128,)
intermediates/decoder/moe_layers_1/{...}
intermediates/decoder/moe_layers_2/{...}
intermediates/decoder/moe_layers/layers_0/moe_z_loss        shape=(5,)     # ← scan-stacked
intermediates/decoder/moe_layers/layers_0/moe_expert_counts shape=(5,128)
intermediates/decoder/moe_layers/layers_{1,2,3}/...
```

params 树（实测，`gs://data_ant/pretrain/ling3/maxtext_ckpt/ling3-tiny-scan-2/`）：
```
params/decoder/moe_layers_{0,1,2}/mlp/MoeBlock_0/gate/bias       shape=(128,)
params/decoder/moe_layers/layers_{0..3}/mlp/MoeBlock_0/gate/bias shape=(128, 5)
```

→ Phase 2 是 **嵌套 layout B**（`moe_layers/layers_X/{key}`），不是 trainer 假设的 flat layout A（`moe_layers/{key}`）。

### 0.4 为什么 DeepSeek / LING2 不踩

- DeepSeek 一般 `inhomogeneous_layer_cycle_interval=1` → 没有 Phase 1b prefix；
- DeepSeek/LING2 用单一 `moe_layer` 直接 scan → Phase 2 结构是 **flat** `decoder/moe_layers/{key}`，恰好匹配当前代码；
- 只有 Ling3 用 `Ling3ScannableBlock` 把 4 个子层打包进 scan 单元 → 结构变成 nested，三段路径全错。

### 0.5 证据链

1. **逻辑 repro**（`tools/dev/repro_ling3_scan_aux_bug.py`，纯 numpy）：构造 mock layout A/B，BUGGY 函数在 nested layout 下 backbone_sum=0，FIXED 函数正确。
2. **运行时 dump**（`tools/dev/dump_ling3_intermediates.py`，TPU + ling3-tiny init）：实证内存布局 = layout B。
3. **Ckpt inspection**（`tools/dev/inspect_ling3_scan_ckpt.py`，Pod 上拉 `ling3-tiny-scan-2/`）：实证 bias 真实路径 = `moe_layers/layers_X/.../bias`。
4. **静态代码追踪**（下方 §1-§3）：路径不对称已可定位，但范围被实测扩大了。

---

## 1. 模型结构（已验证）

ling3-tiny.yml 关键字段：
- `num_decoder_layers: 24`
- `first_num_dense_layers: 1`
- `inhomogeneous_layer_cycle_interval: 4`
- `routed_bias: true`，`routed_bias_update_rate: 0.001`，`enable_routed_bias_grad: false`
- `moe_z_loss_weight: 2.9e-06`，`load_balance_loss_weight`（继承 base.yml）
- `mtp_num_layers: 1`，`scan_layers: true`

`Decoder._apply_ling3_scan_layers` (`src/maxtext/layers/decoders.py:1227-1314`) 把 24 层切成三段：

| 段 | 层数 | 命名 | 路径 |
|---|---|---|---|
| Phase 1a Dense prefix | 1 | `dense_layers_0` | unscan |
| **Phase 1b MoE prefix** | **3** | `moe_layers_0`, `moe_layers_1`, `moe_layers_2` | **unscan** |
| Phase 2 ScannableBlock | 5 × 4 = **20** | `moe_layers/layers_{0..3}`（沿 `param_scan_axis` 堆叠 5 次） | scan |

总 MoE = 3 unscan + 20 scan = 23（= `num_decoder_layers - first_num_dense_layers`）✓

unscan 模式则是 1 个 `dense_layers_0` + 23 个 `moe_layers_{0..22}`，全部 unscan。

---

## 2. 头号嫌疑：Phase 1b 的 MoE prefix 在 trainer 端被遗漏

### 2.1 `_collect_moe_intermediate_sum` 只读 scan 区域

`src/maxtext/trainers/pre_train/train.py:113-122`：

```python
if config.decoder_block in (DecoderBlockType.DEEPSEEK, DecoderBlockType.LING2, DecoderBlockType.LING3):
    if config.scan_layers:
        nested_key = ("intermediates", "decoder", "moe_layers", key)   # 只看 scan 区
        values = maxtext_utils.get_nested_value(intermediate_outputs, nested_key, 0.0)
    else:
        num_moe_layers = config.num_decoder_layers - config.first_num_dense_layers
        values = []
        for i in range(num_moe_layers):                                # unscan 全部 23 层
            nested_key = ("intermediates", "decoder", f"moe_layers_{i}", key)
            values.append(_get_sow_scalar(nested_key))
```

`Ling3GenericLayer.post_process` (`src/maxtext/models/ling3.py:213-238`) 对 Phase 1b 的 3 个 `moe_layers_0/1/2` **会正常 sow** `moe_lb_loss`、`moe_z_loss`、`moe_expert_counts`，只是上面这段把它们丢了。

### 2.2 `_update_deepseek_bias` 同样只更新 scan 区域

`src/maxtext/trainers/pre_train/train.py:175-202`：

```python
def _update_deepseek_bias(config, new_state, moe_block_name, moe_expert_counts):
    if config.scan_layers:
        target_path = (
            "params",
            "decoder",
            "moe_layers",                       # 只 touch scan 区
            moe_block_name,
            "MoeBlock_0",
            "gate",
            "bias",
        )
        new_state = _try_update_bias(...)
    else:
        num_moe_layers = config.num_decoder_layers - config.first_num_dense_layers
        for i in range(num_moe_layers):         # unscan 23 层全部更新
            target_path = ("params", "decoder", f"moe_layers_{i}", ...)
            new_state = _try_update_bias(...)
```

### 2.3 `_count_moe_layers` 用的是配置层数（24=23+MTP），分母不变

`src/maxtext/trainers/pre_train/train.py:83-97`：返回 `(num_decoder_layers - first_num_dense_layers) + mtp_num_layers = 24`。

mean 路径（`router_bias_mean` 等）`= sum / 24`，但 sum 在 scan 模式只覆盖 21 层（20 scan + 1 MTP）→ mean 报告偏低。

z_loss 在 `src/maxtext/trainers/pre_train/train.py:666-670` 同样除以 `num_moe_layers=24`，但被加进 loss 的 raw sum 也只覆盖 21 层。

### 2.4 后果（推论，待 ckpt 实测）

ling3-tiny scan 模式下，3 / 23 ≈ 13% 的 MoE 层：

1. **lb_loss 不计入 loss**：router 没有约束，专家可塌缩。
2. **z_loss 不计入 loss**：router logits 没有正则，可能爆炸 / drift。
3. **routed_bias 永远是初始值**（`enable_routed_bias_grad=false` 同时禁了 grad 路径）：loss-free balancing 完全失效。

差异会**随训练步数累积**——开始一致，越往后差距越大。这与"scan 训练 loss 比 unscan 高/收敛差"的现象吻合。

unscan 模式无此问题，全部 23 层都被正确处理。

### 2.5 旁证

- DeepSeek-V3 不撞这个坑：DeepSeek 一般 `inhomogeneous_layer_cycle_interval=1`，导致 `unscan_prefix == first_num_dense_layers`，**Phase 1b MoE prefix 数 = 0**。同样的代码对 DeepSeek 是正确的。
- Ling2 同样有理论风险（共用 `_collect_moe_intermediate_sum` 和 `_update_deepseek_bias` 的同一分支），但需要看 ling2 实际配置 `first_num_dense_layers` 与 `inhomogeneous_layer_cycle_interval` 是否触发非零 MoE prefix。**待确认**。

---

## 3. 次要嫌疑（暂未发现 smoking gun）

### 3.1 `Ling3ScannableBlock.__call__` 不传 `global_layer_idx`

`src/maxtext/models/ling3.py:438-447`：scan 内部调用子层时没传 `global_layer_idx`，与 docstring 在 `ling3.py:386-388` 的明确说明矛盾。

**当前无害**：`attention_kda.py:364` 是 `del layer_idx  # Not used`，且 `self.layer_idx` 仅存储不读。一旦 KDA 内部加任何 per-layer 行为（slope decay、A_log scale），scan 内 KDA 全用 local `layer_idx ∈ {0,1,2}`，会和 unscan 路径下 global `{0,1,2,4,5,6,8,9,10,…}` 的行为不一致。

### 3.2 RNG 顺序

`Decoder.scan_decoder_layers` (`decoders.py:571-595`) 设了 `split_rngs={"params": True}`。
- unscan：每层 fresh key，按层序顺次拆。
- scan：scan 内每个 iteration fresh key，但拆分顺序与 unscan 不同。

**暂时无害**：只影响"同 master seed 不同模式"的初始化分布，不影响"加载同一 ckpt 的训练动力学"。但意味着**严格的 scan vs unscan 等价性测试必须从同一组 params 出发**，不能从 master seed 出发。

### 3.3 测试缺口

`tests/unit/ling3_decoder_test.py` 与 `tests/unit/ling3_checkpoint_conversion_test.py` 仅覆盖 dispatch / 构造 / boundary math。**没有任何 numerical equivalence 测试**（同 params, scan vs unscan 输出/梯度应一致），这就是 §2 的 bug 能逃过 CI 的根本原因。

---

## 4. 排查清单（按你"一条一条排查"要求顺序排）

### Step 1 — 代码层快速 sanity check（不需要 ckpt）

- [ ] **C1**：在本地 `grep "moe_layers"` 看 `train.py` 是否还有别的分支处理 ling3 scan 的 unscan prefix（例如 metric_logger 那侧）。预期：没有；如果有就要重看。
- [ ] **C2**：`metric_logger.py` 里读 `moe_lb_loss` / `moe_z_loss` / `moe_expert_counts` 的代码，是否同样有遗漏 unscan prefix 的现象。
- [ ] **C3**：跑一次单元测试快速过一下：`pytest tests/unit/ling3_decoder_test.py -v`，确认现状基线。

### Step 2 — Ckpt 层验证（需要 GCS 访问）

从 `gs://data_ant/pretrain/ling3/maxtext_ckpt/ling3-tiny-scan-2/` 拉一个非初始 step 的 ckpt：

- [ ] **K1**：用 orbax 读出来，定位以下 6 个 bias 的范数：
  - `decoder/moe_layers_0/mlp/MoeBlock_0/gate/bias`
  - `decoder/moe_layers_1/mlp/MoeBlock_0/gate/bias`
  - `decoder/moe_layers_2/mlp/MoeBlock_0/gate/bias`
  - `decoder/moe_layers/mlp/MoeBlock_0/gate/bias`（堆叠张量，shape 应该是 `[5, 128]`，看每行的范数）
- [ ] **K2**：预期：前 3 个 ≈ 0（从未更新），后者每行非零。如果**确实如此**，§2 root cause 锁定。
- [ ] **K3**：作为对照，再拉一个 unscan 模式的 ckpt（如果有），检查 `moe_layers_{0..22}/mlp/MoeBlock_0/gate/bias` 是否全部非零。

### Step 3 — 运行时验证（需要 Pod）

- [ ] **R1**：在 ling3-tiny scan 训练里加临时打点，dump `intermediate_outputs["intermediates"]["decoder"]` 的 keys。预期看到 `moe_layers_0`、`moe_layers_1`、`moe_layers_2`、`moe_layers` 共 4 项；前 3 个的 `moe_expert_counts` 有内容但被 `_collect_moe_intermediate_sum` 丢弃。
- [ ] **R2**：把当前 scan 训练运行 50~100 step 的 router 监控（`router_bias_mean`、`router_probs_std`）和 unscan 训练做对比，看 scan 模式是否在 Phase 1b 层（layer 1, 2, 3）出现专家分布显著倾斜的迹象。
- [ ] **R3**：写一个 numerical equivalence 测试：固定 params（reused from a single init），同时跑 scan 和 unscan 一步前向，比对 logits 与 lb_loss、z_loss 的 sum。预期：**logits 应一致**（在 dtype 误差内），**lb_loss / z_loss / bias_update sum 在当前 buggy 代码下不一致**。

---

## 5. Fix 草案

### 5.1 最小修复（保守，仅修对称性）

`_collect_moe_intermediate_sum` 和 `_update_deepseek_bias` 在 scan 分支额外补上对 unscan prefix 的处理。复用 `_apply_ling3_scan_layers` 内的 `unscan_prefix` 计算逻辑：

```python
# 伪代码
def _ling3_unscan_moe_prefix_count(config):
    interval = config.inhomogeneous_layer_cycle_interval
    if config.first_num_dense_layers > 0:
        unscan_prefix = ((config.first_num_dense_layers + interval - 1) // interval) * interval
    else:
        unscan_prefix = 0
    return max(unscan_prefix - config.first_num_dense_layers, 0)
```

`_collect_moe_intermediate_sum` ling3 scan 分支：
```python
if config.scan_layers:
    # scan 区
    scan_values = maxtext_utils.get_nested_value(
        intermediate_outputs, ("intermediates", "decoder", "moe_layers", key), 0.0)
    # unscan prefix
    prefix = _ling3_unscan_moe_prefix_count(config)
    prefix_values = [
        _get_sow_scalar(("intermediates", "decoder", f"moe_layers_{i}", key))
        for i in range(prefix)
    ]
    # ... sum 在一起
```

`_update_deepseek_bias` 同理：scan 分支额外 loop `prefix` 个 `moe_layers_{i}` 路径再调一次 `_try_update_bias`。

### 5.2 测试

- [ ] **T1**：新增 `tests/unit/ling3_train_aux_collection_test.py`，构造一个 mock `intermediate_outputs`（含 `moe_layers_0/1/2/...` 与 `moe_layers` 的 sown 值），断言 scan 模式与 unscan 模式 collect 出的 sum 相等。
- [ ] **T2**：新增 `tests/unit/ling3_bias_update_test.py`，断言 scan 模式调一次 `_update_deepseek_bias` 后，`decoder/moe_layers_{0,1,2}/.../bias` 也被改写。
- [ ] **T3**：（理想但成本高）端到端 numerical equivalence：固定 params，scan vs unscan 一步前向，logits 与 aux 输出在 dtype 容差内一致。

### 5.3 防御性建议（可选）

- 把 `_apply_ling3_scan_layers` 用到的 `unscan_prefix` / `num_moe_prefix` 计算抽出来作为 module-level helper（`maxtext.layers.decoders.compute_ling3_scan_layout(config)` 之类），让 trainer 端、checkpoint conversion 端、test 端都引用同一个，避免再次发散。

---

## 6. 待用户确认

1. ling2 里是不是也有同样的问题（要看 ling2 的 `first_num_dense_layers` / `inhomogeneous_layer_cycle_interval` 组合）。如果 ling2 没用 scan 在生产里，就先不管。
2. ling3 scan ckpt（`gs://data_ant/pretrain/ling3/maxtext_ckpt/ling3-tiny-scan-2/`）是否已经训练了足够多 step 让 §2.4 的累积效应可见——如果只跑了几十 step，bias 范数差别可能还不明显。
3. 是否需要回填修一个旧 ckpt（把 `moe_layers_0/1/2` 的 bias 用 §5 fix 后的逻辑离线 update 出来），还是直接重跑。

---

## 7. 下一步行动（2026-05-12 制定）

### 7.1 修复方向（要改的就这三处）

`src/maxtext/trainers/pre_train/train.py` 加一个 LING3 专用 scan 分支，三处对称修改：

1. **`_collect_moe_intermediate_sum`（L131-148）**
   - 先收 Phase 1b prefix：循环 `_ling3_unscan_moe_prefix_count(config)` 个 `intermediates/decoder/moe_layers_{i}/{key}`；
   - 再收 Phase 2 sub-layers：循环 `inhomogeneous_layer_cycle_interval` (=4) 个 `intermediates/decoder/moe_layers/layers_{j}/{key}`，每个是 `(scan_length,)` 张量；
   - 由于 list 元素 shape 不齐（标量 vs 向量），求和走 `jnp.sum(jnp.asarray(v))` 元素累加，不能用 `jnp.array(values)`。

2. **`moe_expert_counts` 收集（L477-499）**
   - 同样的两段；Phase 1b 每层 `(num_experts,)`，Phase 2 每个 sub-layer `(scan_length, num_experts)`。
   - 后续 `_apply_moe_bias_updates` 需要把这两种形状区分开传下去。

3. **`_update_deepseek_bias`（L222-232）** —— 涉及两个易错点（counts/update 轴方向、zero-mean 轴方向）

   scan 分支拆成两段：

   - **Phase 1b**：循环 prefix 调 `_try_update_bias`，target_path = `params/decoder/moe_layers_{i}/.../bias`，counts shape `(num_experts,)`，bias shape `(num_experts,)`，沿用现有逻辑（1D 情况下 `axis=-1` 就是 expert 轴，`zero_mean_update` 也对）。

   - **Phase 2**：循环 4 个 sub-layer 调 `_try_update_bias`，target_path = `params/decoder/moe_layers/layers_{j}/.../bias`。形状如下：
     - counts：`(scan_length, num_experts) = (5, 128)`，**experts 在最后轴**。
     - bias：`(num_experts, scan_length) = (128, 5)`（参数树实测形状，由 `param_scan_axis=1` 把 scan 轴插到末尾决定），**experts 在第 0 轴**。

   正确流程：
   ```
   counts (5, 128) ──expert_counts_to_bias_update──▶ update (5, 128)   # 沿 axis=-1 求 expert 平均，正确
   update.T = (128, 5)                              ──add──▶ bias (128, 5)   # transpose 后形状对齐
   ```

   **错误做法（不要做）**：在调 `expert_counts_to_bias_update` 之前先把 counts 转成 `(128, 5)`。这样函数的 `axis=-1` 就会沿 scan 轴求平均（"每个 expert 在 5 个 scan step 上的平均负载"），语义就错了。

4. **`update_state_param` zero-mean 轴（`utils/maxtext_utils.py:1172`）—— 顺手修一个潜在 bug**

   当前写死 `jnp.mean(updated, axis=-1, keepdims=True)`。
   - 1D bias `(num_experts,)`：`axis=-1 = experts`，正确。
   - 2D bias `(num_experts, scan_length)`：`axis=-1 = scan_length`，**沿 scan 轴求均值**——把每个 expert 在 5 个 scan 步上的 bias 拉平。语义错；正确做法是 `axis=0`（沿 expert 轴），让每层 router 的 bias 期望为 0。

   **这个 bug 不是 ling3 独有的**：DeepSeek/LING2 scan 路径下 bias 也是 2D `(num_experts, scan_length)`，它们的 `routed_bias_zero_mean_update=true` 一直在沿 scan 轴跑。只是 DeepSeek 没人正面验证过 zero-mean 数值，长期没人吭声。

   **修法**：给 `update_state_param` 加一个 `zero_mean_axis: int = -1` 参数，向后兼容。
   - 1D 路径（DeepSeek/LING2 unscan、ling3 unscan、ling3 Phase 1b prefix）：用默认 `-1`。
   - 2D 路径（DeepSeek/LING2 scan、ling3 Phase 2 sub-layer）：传 `zero_mean_axis=0`。

Prefix 公式（与 `decoders.py:_apply_ling3_scan_layers` 一致）：
```python
def _ling3_unscan_moe_prefix_count(config):
  interval = config.inhomogeneous_layer_cycle_interval
  if config.first_num_dense_layers > 0:
    unscan_prefix = ((config.first_num_dense_layers + interval - 1) // interval) * interval
  else:
    unscan_prefix = 0
  return max(unscan_prefix - config.first_num_dense_layers, 0)
```

### 7.2 测试

- **T1**（必须）：`tests/unit/ling3_train_aux_collection_test.py`，构造 mock `intermediate_outputs`（含 `moe_layers_{0,1,2}/{key}` + `moe_layers/layers_{0..3}/{key}`），断言 scan 与 unscan collect 出的 sum 在容差内一致。同样测 `moe_expert_counts`。
- **T2**（必须）：`tests/unit/ling3_bias_update_test.py`，scan 模式调一次 `_update_deepseek_bias`，包含以下断言：
  - **存在性**：4 个 Phase 2 sub-layer + 3 个 Phase 1b prefix 的 bias 都被改写（`||new - old||_2 > 0`）。
  - **形状**：`update.shape == counts.shape == (scan_length, num_experts)`，`update.T.shape == bias.shape == (num_experts, scan_length)`——避免任何后续重构走错 transpose。
  - **zero-mean 数值**：调用前构造一个非零均值的 init bias，调用后 `new_bias.mean(axis=0)` 在每个 scan 步内 ≈ 0（容差 1e-6），同时显式验证 `new_bias.mean(axis=-1)` **不**为 0（避免误把 axis 改回 -1 时蒙混过关）。
  - **方向性**：构造一个 hot expert（counts 远大于均值），断言对应 bias 收到 **负** delta。
- **T3**（理想）：固定 init params，scan vs unscan 一步前向，logits + lb_loss + z_loss + bias 在 dtype 容差内一致。成本高但能彻底锁死回归。
- **CI**：把 `_ling3_unscan_moe_prefix_count` 抽成 module-level helper（参 §5.3），让 trainer / decoders / ckpt-conversion / tests 共用。

### 7.3 已有 ckpt 的处理

`gs://data_ant/pretrain/ling3/maxtext_ckpt/ling3-tiny-scan-2/`（已训若干步）的 23 个 backbone bias 都还是 init 值，router 已经 drift。两种选择：

- **A. 重跑**：fix 合并后从原始 ckpt（或 base）重新 pretrain，丢弃当前已训步数。最干净。
- **B. 续训**：fix 合并后直接续训，让 routed_bias 从 init 开始追赶。**不推荐**——前期 drift 的 router 权重需要长时间矫正，且 drift 大小未实测，风险不可控。

建议 A。等 fix + T1/T2 通过后启动重训。

### 7.4 不动手项（非本次 scope）

- `Ling3ScannableBlock.__call__` 不传 `global_layer_idx`（§3.1）：当前 KDA 不读，**无数值偏差**。留 comment 标记，等 KDA 加 per-layer 行为时再补。
- RNG 顺序差异（§3.2）：仅影响"同 master seed 跨模式"对比；从同一 init params 出发的训练不受影响。

### 7.5 操作顺序

1. 改 `utils/maxtext_utils.py:update_state_param`：加 `zero_mean_axis` 参数。
2. 改 `train.py` 三处（collect / counts / bias_update）+ 抽 `_ling3_unscan_moe_prefix_count` helper；按 §7.7 隔离 LING3 与 DeepSeek/LING2 的 scan 路径；按 §7.8 把已知必须存在的路径改成 raise。
3. 写 T1/T2 + §7.8 的 fail-loud 测试，本地 + Pod 上各跑一次。
4. 跑 lint：`bash tools/dev/unit_test_and_lint.sh`。
5. 提 PR；关联本文档作为 root-cause 说明。
6. PR merge 后通知用户启动 §7.3-A 重训。

### 7.6 顺手记录的次生发现

- `update_state_param` 的 zero-mean 轴问题对 DeepSeek/LING2 scan 也同样有影响（§7.1.4）。本次顺手修复，但短期内对 ling3 修复以外的下游影响未量化（DeepSeek 训练曲线没人对照过有无 bug 的差异）。在 PR 描述里显式列出，提醒下游模型 owner。

### 7.7 与 DeepSeek / LING2 的隔离策略

当前 `_collect_moe_intermediate_sum` / `_update_deepseek_bias` 的 scan 分支用的是 flat layout `decoder/moe_layers/{key}`，恰好匹配 DeepSeek/LING2 的真实结构（它们用单一 `moe_layer` 直接 scan）。LING3 因为 `Ling3ScannableBlock` 包了 4 个子层，结构是 nested `decoder/moe_layers/layers_X/{key}`，三段路径全错（详见 §0.3-0.4）。

**做法：在 scan 分支再按 `decoder_block` 分一次叉**：

| decoder_block | scan 分支走的路径 | 本次改动 |
|---|---|---|
| `DEEPSEEK` / `LING2` | flat layout（沿用现 L115 / L482） | **零改动** |
| `LING3` | nested layout + Phase 1b prefix（新分支） | 三处都要新增 |

核心约束——**LING3 的修改必须收敛到 LING3-only 的代码块里**，绝不能动到 DEEPSEEK/LING2 共用的代码路径，避免 DeepSeek 训练曲线被意外打扰。

唯一跨模型的改动是 `update_state_param` 的 `zero_mean_axis` 参数（§7.1.4）：
- 加默认 `-1` 保证 API 向后兼容；
- LING3 Phase 2 显式传 `axis=0`；
- DEEPSEEK/LING2 scan 路径**也应**传 `axis=0`（它们当前就是错的），但这一改动会改变 DeepSeek/LING2 scan 的训练数值。**默认勾选**，但在 PR 描述里独立列一条，提醒 DeepSeek owner，给一个 opt-out 的窗口。

### 7.8 把静默失败改成显式抛错

本 bug 能藏到 ckpt 训完才被发现，根因是这一连串 fallback 都是 silent default：

| 位置 | 当前行为 | 改法 |
|---|---|---|
| `_collect_moe_intermediate_sum`：`get_nested_value(..., default=0.0)` | 路径不存在 → 0.0 静默累加 | 进入"已知 decoder_block + scan 组合"分支后，缺失就 `raise RuntimeError(f"Expected sown {key} at {path} but missing — model layout drift?")` |
| `moe_expert_counts` 收集（L482/495）：`get_nested_value(..., default=None)` + `if any(...)` | 路径全空 → `moe_expert_counts=None` → 跳过 bias update | `routed_bias=True` 且 `decoder_block` 受支持时，必须收到非空 counts，否则 raise |
| `_try_update_bias`（L162-164）：`has_nested_key` 不命中 → `max_logging.log` + return | 静默 log，继续训 | 已知必须存在的路径（DeepSeek/LING2 scan 的 `moe_layers/.../bias`、unscan 的 `moe_layers_{i}/.../bias`、LING3 scan 的 Phase 1b prefix 与 Phase 2 sub-layer）→ raise；**只有 MTP 路径**保留 log（`mtp_num_layers=0` 时合法不存在） |
| `_update_deepseek_bias` scan 分支 | 路径错也只是 `_try_update_bias` log 一下 | 由上一条覆盖 |

**实现要点**：

- `_try_update_bias` 加个 `required: bool = True` 参数。已知必须存在的调用点不传或传 True，缺失即 raise；MTP 那条调用点传 `required=False`，保留 log + return 行为。
- 抛错信息要带上：当前 `decoder_block`、`scan_layers`、被找的完整 path、（如果可能）顶层可见的 keys。**目的是任何一次新模型接入或结构调整忘了 trainer 端对应路径，都会立刻在第一个 step 炸出来，而不是 100 步后训练曲线偏离。**
- 注意 jit/trace 时这些 raise 是 Python-level 的 control flow，不会被 trace 进 graph，所以路径检查发生在 `loss_fn` Python 体里就够，没有 jit-time 副作用。

**对应测试（必须）**：

- `test_collect_raises_on_missing_path`：构造一个**故意缺 `moe_layers/layers_2/moe_lb_loss`** 的 scan intermediates，断言 `_collect_moe_intermediate_sum` 抛 RuntimeError。
- `test_bias_update_raises_on_missing_param`：构造一个故意缺 `moe_layers/layers_1/.../bias` 的 params，断言 `_try_update_bias` 抛错。
- `test_mtp_path_missing_is_silent`：MTP 路径缺失只产生 log，不 raise。

这三条测试是这次 fix 的"反向保险"——本次能把当前 bug 改对，但更重要的是**未来同类 layout drift 不会再被静默吞掉**。
