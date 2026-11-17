# 部署结果

> 2026-11-10：环境已迁移为 `/home/seed/miniconda3/envs/datagen`，使用 `conda activate datagen`；旧环境目录已删除。详见 `ENVIRONMENT_MIGRATION.md`。

日期：2026-11-04。主目录 `/home/seed/wangruiyao/datagen`。

## 已完成

| 项目 | 实际状态 |
|---|---|
| 新环境 | `/home/seed/miniconda3/envs/datagen`，Python 3.10.22 |
| GPU | RTX 5090，计算能力 12.0，Torch 包含 sm_120 |
| CUDA | Torch cu128；独立 nvcc 12.8.93，不改系统 CUDA 12.6 |
| Torch | 2.7.1+cu128 |
| TorchVision | 0.22.1+cu128 |
| TorchAudio | 2.7.1+cu128 |
| xFormers | 0.0.31；CUTLASS 小规模前向通过，默认调度失败，见下文 |
| COLMAP | pycolmap 3.12.6，CPU 提特征/匹配/建图/去畸变 |
| SAM2 | 官方源码独立放在 repos/sam2，SAM2.1 small 权重已下载 |
| Omnidata | 官方推理源码已部署；两个正式权重已补齐并通过单图 GPU 推理（2026-11-17） |
| 依赖一致性 | pip check：No broken requirements found |
| 工具 | extract / sfm / prepare / split / segment / priors / preview / export |

## 实际验证边界

- 核心版本、sm120、GPU 矩阵运算、TorchVision GPU NMS、PyTorch SDPA：通过。
- xFormers 0.0.31 默认 attention 在此 RTX 5090 上进入 Hopper FlashAttention 路径后报 `invalid argument`，子进程退出；显式 `MemoryEfficientAttentionCutlassOp` 通过。**不能把“装好了”理解为所有 xFormers kernel 都支持此卡。** 本流水线使用 PyTorch SDPA，不依赖 xFormers 默认 attention；没有升级或降级用户指定版本。
- 合成视频：抽出 12 帧；COLMAP 恢复 12 个相机，去畸变/裁剪/相机 NPZ 与重投影预览生成成功。
- SAM2.1 small：使用真实官方权重在 GPU 上跟踪上述全部 12 帧成功。
- Omnidata 深度/法线网络：无网络下载、随机初始化的 GPU 架构前向通过，输出分别 `(1,384,384)`、`(1,3,384,384)`。2026-11-17 已用正式 checkpoint 追加完成单图 GPU 推理，输出形状/类型/数值有限性通过；真实视频先验质量仍需在实际素材上检查。随机架构测试输出没有作为实际数据集先验使用。
- 4 项针对抽帧源帧号、投影/裁剪/归一化、无重复采样、双格式导出的测试通过。
- 实际 DP-Recon 和 ObjectSDF++ 的原始 dataset loader 均成功读取独立目录中的合成导出数据：10 图、1 个前景实例。测试未改写两个项目，且禁止向源码写入 pyc。
- 没有提供真实视频，尚未完成真实素材全流程验收，也未启动两个基线训练。

## 下载和隔离说明

普通依赖使用可访问的清华镜像；Torch 家族及 xFormers 使用官方 cu128 源。外网下载 NVIDIA 大包很慢，因此只读复制现有 dprecon 中**精确匹配版本**的 NVIDIA 运行库和 Triton 到新环境，使用普通文件复制，没有软/硬链接，不修改原环境。记录在 `config/runtime-copy-provenance.json`。

Omnidata 的 Git 按需拉取也较慢，已通过本地 HTTPS 下载固定提交 `f69ff3aedf983cb34c490a0afd0d29fdd83f6a1c` 的 8 个官方推理源文件后 scp 到新目录，记录在 `config/omnidata-source.json`。无需安装 Omnidata 的旧版完整 requirements，更不要因此替换 Torch 或安装其旧 Lightning 训练栈。

SAM2 可选连通区域 CUDA 扩展未编译，关闭该后处理；核心预测网络正常。SAM3 不在本次部署范围，避免扰动 Python 3.10 和固定依赖。无需额外安装 SAM3 才能使用当前流程。

没有进行模型哈希检查、全服务器扫描或系统软件安装。pip/conda 自带的包完整性处理属于安装本身。

## 交接文件

- `README.md`：完整命令和数据协议。
- `MANUAL_TODO.md`：需用户处理的事项、模型链接、放置位置。
- `config/annotations.example.json`：提示点/框、实例说明和落地标记模板。
- `config/runtime-verification.json`：实际 GPU 验证记录。
- `config/installed-packages.txt`：最终 pip 包清单。
- `logs/`：安装、模型下载、COLMAP/SAM2/两项目 loader 测试记录。
- `work/synthetic-smoke/`：合成素材及流程演示，不能当作研究实验数据。
- `datasets/`：留给真实数据导出，不向基线项目自动写文件或链接。
