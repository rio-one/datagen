# datagen 环境迁移记录

迁移日期：2026-11-10。

- 当前环境：`/home/seed/miniconda3/envs/datagen`，Conda 名称 `datagen`。
- 已通过离线克隆迁移全部 Conda/pip 包；74 项 Python 分发包版本与原环境一致。
- 已更新启动器、env.sh、bootstrap.sh、缓存隔离配置和使用文档。
- Python 3.10 / Torch 2.7.1 cu128 / TorchVision 0.22.1 / TorchAudio 2.7.1 / xFormers 0.0.31 保持不变。
- 新路径激活、pip 依赖检查、CUDA 12.8 编译器、RTX 5090 sm120 和小规模 GPU 运算验证通过。
- 原环境 `/home/seed/wangruiyao/datagen/envs/datagen` 已用 Conda 删除，目录不存在。
- 工具源码、模型、缓存和数据仍位于 `/home/seed/wangruiyao/datagen`。
- 其他环境、系统 CUDA、全局 shell/Conda 配置未修改；Conda 正常更新自身的环境注册表。

```bash
conda activate datagen
cd ~/wangruiyao/datagen
source scripts/env.sh
./datagen --help
```

历史安装日志中的原路径仅用于追溯，不是当前运行路径。
详情见 `config/environment-migration.json`、`config/migration-verification.json`。
