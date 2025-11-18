## 项目概览

  - 本仓库实现基于 JAX/Flax 的 1D SimVQGAN（向量量化自动编码器 + PatchGAN 判别器），目标是对 Nanopore POD5 原始电流信号进行时域重建与对抗式训练。

## 核心目录与职责

  - **train.py**：入口脚本，设置 JAX 运行环境（TF32 / CUDA 选择）后转调 `scripts/train.py`。路径：`train.py`.
  - **scripts/train.py**：解析配置、加载 POD5 数据、构建模型与判别器、训练/验证、记录日志与断点。路径：`scripts/train.py`.
  - **configs/**：训练配置 JSON（默认 `train_config.colab.json`），包含数据路径、模型与优化超参。路径：`configs/train_config.colab.json`.
  - **codec/ 包**：
      - **data/**：POD5 读取、鲁棒归一化、分块与线程/设备预取。路径示例：`codec/data/pod5_dataset.py`.
      - **models/**：编码器/解码器、SimVQ 量化器、PatchGAN 判别器。主模型：`codec/models/model.py`.
      - **train/**：训练状态、损失、单步梯度计算、主训练循环。路径：`codec/train/loop.py`, `codec/train/step.py`, `codec/train/losses.py`.
      - **jaxlayers/**：卷积等 JAX/Flax 层封装。路径：`codec/jaxlayers/layers.py`.
      - **utils/**：POD5 文件发现、随机种子、WandB 包装、断点同步。路径：`codec/utils/*.py`.
      - **runtime.py**：JAX/XLA 环境配置与编译缓存。路径：`codec/runtime.py`.
  - **scripts/setup\_local\_repo.py**：在 Colab SSD 镜像仓库以加速 I/O。

## 数据与预处理流程

1.  **scripts/train.py** 解析配置 -\> 解析数据根目录、子目录与采样率。
2.  **codec/utils/pod5\_files.py** 查找 POD5 文件；`NanoporeSignalDataset.from_paths` 校验存在性。
3.  每个 read：读取 POD5 校准 (offset/scale) 将 int16 ADC 转为 pA，再执行鲁棒归一化 (median/MAD) 并按窗口长度分段 (window\_ms 推导 chunk\_size)。
4.  通过 `Prefetcher` 线程预取到主机队列，可选 `make_device_prefetcher` 将 batch 放入设备并按多 GPU 分片。

## 模型与训练主线

  - 编码器 (`SimVQEncoder1D`) 级联残差块+降采样；量化前 1×1 卷积；`SimVQ1D` 量化输出代码索引与承诺损失；解码器上采样重建波形 (`SimVQDecoder1D` + `PatchGAN` 判别器)。
  - 损失 (`compute_generator_losses`)：L1 重建 + 承诺 + GAN + feature matching，权重来自配置。
  - 训练步 (`compute_grads`)：生成器前向→判别器真/假→生成器和判别器各自梯度；记录 perplexity、code usage 等统计。
  - 学习率：余弦 + warmup（`_make_lr_schedule`，位于 `codec/train/loop.py`）。
  - 断点：`flax.training.checkpoints` 保存；可镜像到 Google Drive (`drive_backup_dir`)。

## 运行与命令示例

  - **默认训练**（自动选用 `configs/train_config.colab.json` 或候选路径）：
    ```bash
    python train.py --config configs/train_config.colab.json
    ```
  - **指定输出与验证间隔**：
    ```bash
    python scripts/train.py --config configs/train_config.colab.json --ckpt-dir checkpoints/run1 --val-every 2000
    ```
  - **快速冒烟测试**（少步数）：
    ```bash
    python scripts/train.py --config configs/train_config.colab.json --steps 10 --val-every 10
    ```
  - **Colab 镜像以提速 I/O**：
    ```bash
    python scripts/setup_local_repo.py --target /content/VQGAN
    ```

## 配置要点（configs/train\_config.colab.json）

  - **数据**：`data.root` 指向 POD5 目录，`segment_sec=4.8s`，`sample_rate=5000Hz`，`files_per_epoch` 控制采样子集。
  - **模型**：`base_channels=128`，`codebook_size=4096`，`beta=0.25`，编码/解码通道与步幅在 `enc_down_strides` / `dec_up_strides`。
  - **训练**：`steps`、`batch_size`、`learning_rate`、判别器权重 `disc_factor`、保存/验证频率。
  - **日志**：WandB 开关与项目名。请通过环境变量设置 `WANDB_API_KEY`，不要写入源码。

## 开发与扩展步骤（建议顺序）

1.  **新数据路径**：修改/复制 `configs/*.json`，调整 `data.root`、`files_per_epoch` 以适配磁盘/显存。
2.  **模型试验**：在配置中调 `codebook_size`、`latent_dim`、enc/dec 通道表；保持下采样与上采样对齐。
3.  **增加指标**：在 `codec/train/step.py` 中扩展日志键，或在 `codec/train/loop.py` 的 `_simvq_style_logs` 映射输出。
4.  **性能调优**：开启 `XLA_CACHE_DIR`（已在 `runtime.enable_jax_compilation_cache` 支持），合理设置 `loader_workers`、`loader_prefetch_chunks`。
5.  **云端运行**：先执行 `scripts/setup_local_repo.py` 镜像到本地 SSD，再运行训练以减少 I/O 瓶颈。
6.  **测试与调试**：无现成单测；可在 `test/`（已忽略）下用 pytest 针对 `NanoporeSignalDataset` 和损失函数写小样本测试；训练冒烟用极小 `steps` 验证流程不崩。
7.  **提交规范**：短祈使句提交，例如“add pod5 cache guard”；PR 说明需包含改动摘要、运行过的命令/日志、配置差异和是否影响 checkpoints 或路径。

## 注意事项

  - **GPU 环境自动探测**：若不想用 GPU，导出 `CUDA_VISIBLE_DEVICES=""` 再运行。
  - **不要提交大型 POD5 或密钥**；配置中的 API key 字段仅作占位，应依赖环境变量。
