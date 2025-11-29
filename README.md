## SimVQGAN for Nanopore Current Reconstruction

SimVQGAN 是一个基于 JAX/Flax 的 1D 向量量化生成对抗网络，用于对 Nanopore POD5 原始电流信号进行时域重建。项目聚焦于：

- **自监督压缩与重建**：SimVQ 编码器/解码器 + PatchGAN 判别器，共 80× 下采样、约 2 s（10000 sample @ 5 kHz）窗口。
- **面向 Colab/A100 的流水线**：线程化 POD5 流式读取、设备预取、XLA cache、TF32。
- **训练后验证**：借助 `scripts/dorado_validate.py` 将真实/生成信号交给 ONT Dorado basecaller，得到简单 identity 指标。

---

### 目录速览

| 路径 | 说明 |
| --- | --- |
| `train.py` | 入口脚本：配置 runtime 后转调 `scripts/train.py`。 |
| `scripts/train.py` | CLI 训练管线（配置解析、数据集构建、模型组装、日志/断点）。 |
| `scripts/dorado_validate.py` | 训练后验证（重建 POD5 + Dorado basecalling + identity 报告）。 |
| `configs/` | JSON 配置（训练、验证）。默认：`train_config.colab.json`、`validate_dorado.colab.json`。 |
| `codec/data/` | POD5 正规化、窗口切分、线程/设备预取；核心类 `NanoporeSignalDataset`。 |
| `codec/models/` | 编码器、解码器、量化器、PatchGAN 判别器、整合模型。 |
| `codec/train/` | 损失、TrainState、梯度计算、完整训练循环。 |
| `codec/utils/` | POD5 文件发现、随机数、WandB、checkpoint 同步等工具。 |
| `codec/runtime.py` | 设置 JAX/XLA 环境（TF32、CUDA、编译缓存等）。 |

---

### 数据流与预处理

1. **文件发现**：`scripts/train.py` 读取配置 → `codec/utils/pod5_files.py` 枚举 POD5 → 每个 epoch 顺序遍历所有文件。
2. **流式读取**：`NanoporeSignalDataset` 在独立线程中顺序读取 POD5，每个 read 通过 `codec/data/pod5_processing.py` 将 int16 ADC → pA，并执行鲁棒 median/MAD 归一化。
3. **窗口化**：`segment_sec=2.0` 与 `segment_samples=10000`（固定每个窗口 10000 sample）共同约束 chunk 大小，短于 10000 的 read 会被直接跳过，末尾不足的片段也会被丢弃。训练阶段仅消费归一化波形；验证/生成阶段额外保留 normalization stats 用于还原。
4. **预取**：`Prefetcher` 将 CPU 线程产出的 batch 推送到主进程，`make_device_prefetcher`（可选）再搬运到 GPU，以减少输入瓶颈。

---

### 模型结构（默认）

- **编码器 `SimVQEncoder1D`**：`base_channels=32`、channel multipliers `(1,1,2,2,4)`，down-strides `(4,4,5,1)`（总 80×），残差块每级 2 个。
- **量化器 `SimVQ1D`**：`codebook_size=4096`、`latent_dim=128`，无 EMA，支持 perplexity/usage 统计。
- **解码器 `SimVQDecoder1D`**：镜像上采样 schedule `(128,64,64,32,32)`、up-strides `(1,5,4,4)`，PixelShuffle 风格。
- **判别器 `PatchDiscriminator1D`**：通道 `(32,64,128,256)`，每层卷积+残差，hinge GAN 损失。
- **Loss 组合**：时间域 L1（`time_l1`）、commitment（`beta`）、GAN（`gan`）、feature matching（`feature`），权重可在配置中调整。

---

### 快速上手

> 以下命令均在仓库根目录执行；默认配置位于 `configs/train_config.colab.json`。

1. **安装依赖**（示例）
   ```bash
   pip install -r requirements.txt  # 若无 requirements，请根据 JAX/Flax 版本手动安装
   ```

2. **启动训练**
   ```bash
   python train.py --config configs/train_config.colab.json
   ```
   - 默认 batch size=256、base LR=5e-4、epochs=2、单 GPU（如遇 OOM 可用 `--batch-size` 降低到 128/64）。
   - 日志写入 `<ckpt_dir>/train.log`，WandB 可通过配置中的 `logging.wandb.enabled` 或 `--wandb` 开启。

3. **自定义输出 & 保存频率**
   ```bash
   python scripts/train.py \
     --config configs/train_config.colab.json \
     --ckpt-dir checkpoints/run1 \
     --save-every 2000
   ```

4. **冒烟测试**
   ```bash
  python scripts/train.py --config configs/train_config.colab.json --epochs 1 --log-every 1
   ```

---

### 配置说明

#### `configs/train_config.colab.json`

| 键 | 作用 | 备注 |
| --- | --- | --- |
| `train.epochs` | 训练完整遍历次数（默认 2） | 结合真实 POD5 文件流式迭代；`save_every` 仍按 step 计数。 |
| `train.batch_size` | 每步样本数（默认 256） | 若 Colab/T4 OOM，可在 CLI 传 `--batch-size 128` 或更低。 |
| `train.learning_rate` | AdamW LR 基准（默认 5e-4） | 常数学习率，默认采用 AdamW。 |
| `train.loss_weights` | L1 / commit / GAN / feature 权重 | 对应 `codec/train/losses.py`。 |
| `train.disc_start` | 开始引入 GAN/feature loss 的 step | 0 表示从第一步就启用判别器。 |
| `train.disc_steps` | 每个 batch 的判别器更新次数（默认 1） | 可在配置中显式设为 2（示例配置已这样做）以防止判别器过早饱和。 |
| `model.*` | 编解码通道/步幅、`latent_dim`、`codebook_size` | 需保持 enc/dec schedule 对齐。 |
| `data.segment_sec` | 窗口长度（秒） | 2.0 → 10000 sample @ 5 kHz；与 `segment_samples` 联动。 |
| `data.segment_samples` | 固定每个窗口的样本数 | 10000 时会跳过所有 <10000 read，并丢弃末尾不足的片段。 |
| `checkpoint.dir` | 断点输出目录 | 可配合 `drive_backup_dir` 做镜像。 |
| `logging.wandb.*` | WandB 开关、项目/Run 名 | `api_key` 建议改用环境变量 `WANDB_API_KEY`。 |


#### `configs/validate_dorado.colab.json`

| 键 | 作用 |
| --- | --- |
| `pod5` | 需要验证的真实 POD5 文件。 |
| `ckpt_final` | 训练完的 checkpoint 目录（`ckpt_final/checkpoint_*`）。 |
| `window` & `model` | 复制训练时的 segment / 结构配置，确保验证模型一致。 |
| `dorado` | Dorado 可执行、模型、设备等信息。 |

使用示例：
```bash
python scripts/dorado_validate.py \
  --config configs/validate_dorado.colab.json \
  --pod5 /path/to/sample.pod5 \
  --ckpt-final /path/to/checkpoints/checkpoint_<step> \
  --out-dir /path/to/dorado_eval \
  --dorado-bin /path/to/dorado \
  --dorado-model dna_r10.4.1_e8.2_400bps_sup@v5.2.0
```
输出：裁剪后的真实 POD5、生成 POD5、可选 FASTQ、`dorado_report.json`（包含 mean/median identity）。

---

### 常见工作流

1. **替换数据集**：
   - 复制训练配置，修改 `data.root`、`subdirs` 等路径相关字段。
   - 若采样率不同，更新 `data.sample_rate` 并确保 POD5 元数据一致。
   - 多个 flowcell 目录时，将 `root` 设为共同父目录，再把 `FC01/pod5` 这类子路径放入 `subdirs`，可一次性遍历全部文件。

2. **尝试新模型**：
   - 在配置中调整 `model.enc_channels` / `dec_channels` 等；保持下采样/上采样匹配。
   - 同步修改 `scripts/dorado_validate.py` 使用的验证配置。

3. **调高吞吐**：
   - 增加 `data.loader_workers`、`loader_prefetch_chunks`。
   - 设置 `XLA_CACHE_DIR` 环境变量（`codec/runtime.enable_jax_compilation_cache` 会使用）。

4. **仅继续训练**：
   - 指定 `checkpoint.resume_from` 或在 CLI 中 `--ckpt-dir` 指向已有目录。
   - 使用 `codec/train/train_more` 时可手动加载存档后继续。

5. **验证/发布**：
   - 训练结束后运行 `scripts/dorado_validate.py`，无需依赖在线验证；可在报告中附上 Dorado identity。

---

### 开发建议

1. **编码风格**：Python 3.11+，PEP8/Black 风格，4 空格缩进，模块/函数/变量使用 snake_case。
2. **调试**：`test/` 目录被 gitignore，可自由写 pytest；冒烟测试可通过 `--epochs 1 --log-every 1` 搭配小型数据子集完成。
3. **提交信息**：使用简短祈使句（如 “add pod5 cache guard”），一次提交涵盖相关修改。
4. **安全**：不要提交 POD5、大型 checkpoint 或密钥；通过环境变量提供 `WANDB_API_KEY`、`CUDA_VISIBLE_DEVICES` 等。
5. **资源**：若只想 CPU 运行，设置 `CUDA_VISIBLE_DEVICES=""`；如需自定义 JAX cache 路径，设置 `XLA_CACHE_DIR`。
6. **吞吐**：若 batch size 已触顶，可将 `train.grad_accum_steps` 设为 2（或更高）配合 `--grad-accum` CLI 做梯度累积，同时保持 `data.loader_prefetch_chunks`、`train.host_prefetch_size`、`train.device_prefetch_size` > 0 确保线程/设备预取持续启用。

---

### 故障排查与提示

| 问题 | 可能原因 | 对策 |
| --- | --- | --- |
| 数据集迭代阻塞 | POD5 损坏 / 样本率不匹配 | 控制台会打印 `[warn]` 并跳过；检查 `data.sample_rate`、移除损坏文件。 |
| OOM | batch size 过大或 预取设置导致显存紧张 | 减小 `train.batch_size`、调整 `loader_prefetch_chunks`，或禁用设备预取。 |
| Dorado 验证报错 | `dorado` 不在 PATH、模型路径错误 | 显示的命令会包含复制路径，确认配置中的 `dorado.bin`、`dorado.model`。 |
| WandB 无法初始化 | API Key 缺失 / 网络限制 | 关闭 `logging.wandb.enabled` 或设置 `WANDB_MODE=offline`。 |

---

如需进一步说明（如定制量化器、扩展损失、接入多 GPU），欢迎在对应模块 (`codec/models/quantize.py`, `codec/train/losses.py` 等) 上继续扩展。

Happy hacking!
