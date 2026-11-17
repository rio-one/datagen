# 视频 → DP-Recon / ObjectSDF++ 数据生成工具

> 2026-11-17：正式 Omnidata 权重已补齐并通过单图 GPU 推理。按最新完整手册 [USER_GUIDE_ZH.md](USER_GUIDE_ZH.md) 操作。

> 2026-11-10：环境已迁移为 `/home/seed/miniconda3/envs/datagen`，使用 `conda activate datagen`；旧环境目录已删除。详见 `ENVIRONMENT_MIGRATION.md`。

部署目录：`/home/seed/wangruiyao/datagen`（即 `~/wangruiyao/datagen`）。

这是一套独立工具，不修改两个基线的代码、数据或环境。先运行 `./datagen --help`。实际安装状态见 `DEPLOYMENT_STATUS.md`，需手工处理的事项见 `MANUAL_TODO.md`。

## 环境与隔离

- 新 Python 环境：`/home/seed/miniconda3/envs/datagen`，Python 3.10。
- 固定 Torch 2.7.1 / TorchVision 0.22.1 / TorchAudio 2.7.1，官方 cu128；xFormers 0.0.31。
- 独立 CUDA 12.8 编译器在此环境，目标架构 `sm_120`，不使用系统 CUDA 12.6 来编译。
- pip/conda/Hugging Face/Torch 缓存都在本目录 `.cache/`。
- 部署时为避免外网反复下载数 GB CUDA 运行库，按精确版本只读复用了 dprecon 中的 NVIDIA wheel 运行库和 Triton；均为独立文件复制，不是软链接/硬链接。新环境不依赖原目录继续存在。详见 `config/runtime-copy-provenance.json`。
- 不使用 sudo、apt，不修改 `.bashrc`、系统 CUDA、共享环境和基线代码。源码仅放在 `repos/`。
- 通过 `./datagen` 启动即可，不必激活环境。需要 shell 环境时：

```bash
cd ~/wangruiyao/datagen
source scripts/env.sh
# 此时 python/nvcc 均来自本目录。仅影响当前 shell。
```

常规使用：`conda activate datagen` 后运行 `./datagen ...` 即可，启动器会自动加载隔离设置；不必手动 source。只有绕过启动器直接调用 Python 时，才可能需要手动加载 scripts/env.sh。不要把这个脚本添加到全局启动文件。

`constraints.txt` 限制核心版本；通过启动脚本运行 pip 时会自动生效。不要在 datagen 内运行其他项目的完整 requirements，也不要去修改原来的 dprecon/objsdf 环境。SAM2 的可选 CUDA 后处理扩展未编译，调用时关闭该后处理；网络本身通过 PyTorch 在 GPU 上运行。

## 第一次使用

拍摄静态房间，固定焦距、慢速移动、有视差，避免运动模糊和大面积无纹理画面。建议第一段视频目标 5–10 个物体，候选 200 帧，训练/测试各 10 帧。仅旋转相机、几乎没有位移的视频容易无法恢复几何。视频建议恒定帧率；脚本保存名义帧号/fps 时间，手机可变帧率素材先转码为恒定帧率。

```bash
cd ~/wangruiyao/datagen

# work 名称请按视频新建；不覆盖已有阶段。
./datagen extract --video /绝对路径/room.mp4 --work work/room01 --fps 2 --max-frames 200

# 相机恢复：CPU COLMAP，默认 4 线程，不占用其他 GPU。
./datagen sfm --work work/room01 --threads 4

# 统一裁剪/内参/归一化，生成相机重投影预览。
./datagen prepare --work work/room01 --size 384
./datagen split --work work/room01 --train-views 10 --test-views 10
./datagen preview --work work/room01 --count 12
```

查看 `work/room01/prepared/reprojection_*.jpg`：绿点应落在相应场景特征上。`prepared/sparse_original_coordinates.ply` 可用于观察 COLMAP 原坐标系。`prepare` 未指定 `--up` 时，根据摄像机竖直方向估计世界 up，仅适合大致直立拍摄，必须人工确认。需要时在新工作目录中重用已完成的 frames/sfm 阶段，再运行 `prepare --up X Y Z`；该向量在原始 COLMAP 世界坐标中，目标为 +Z。场景归一化来自稀疏点云的稳健包围范围，不代表已知米制尺度。

### 标注并跟踪物体

复制模板到自己工作目录，并按 `prepared/rgb/` 图片修改：

```bash
cp config/annotations.example.json work/room01/annotations.json
# 用编辑器修改 work/room01/annotations.json，然后：
CUDA_VISIBLE_DEVICES=0 ./datagen segment --work work/room01 --annotations work/room01/annotations.json
./datagen preview --work work/room01 --count 20
```

- 示例点/框只是展示格式，不能直接用于自己的视频。
- `frame` 为 **prepared 序列编号**，不是原视频帧号；坐标是最终 384×384 图上的 `[x,y]`。
- `points` 的 `labels`：1 为物体内部，0 为外部；框是 `[左,上,右,下]`。
- 每个物体稳定编号 1..N，背景 0，同类不同物体也分开。各物体可有多帧提示，尤其遮挡前后/重新出现的帧。
- 先固定 `split.json`，提示点/框只选其中 train 帧；有 split 时 segment 会拒绝在非训练帧添加提示。对测试掩码的人工核验只用于评价，不能将测试纠错再传播回训练标签。
- 对重叠预测选择正 logit 最大的实例作为像素标签。请检查遮挡、细腿、反光、消失/重现时的 ID，SAM2 结果不是人工真值。
- JSON 中填背景及各物体的英文几何/纹理描述；只根据训练视角写描述。`floor_supported` 是该物体是否直接落地，桌上花瓶填 false。
- 修改提示重新跟踪时，先把 `work/room01/labels` 重命名保留为其他名字，再重新运行 segment；不自动覆盖已有标签。
- 也支持手工/其他工具导出的 `work/room01/labels/000000.png` 等单通道整数标签，必须和 prepared/rgb 对齐且保持 ID。

### 生成深度、法线，再导出

```bash
CUDA_VISIBLE_DEVICES=0 ./datagen priors --work work/room01

./datagen export --work work/room01 \
  --annotations work/room01/annotations.json \
  --output datasets/custom/scan1 --data-dir custom --scan-id 1 \
  --train-views 10 --test-views 10
```

使用 Omnidata v2 深度与法线，先保证两边共享完全一致的输入。不是声称复刻 DP-Recon 的全部原论文先验。输入直接使用已处理的 384×384 RGB，不再重复裁剪；深度保存 H×W float32，法线 CHW float32、[0,1] 编码，保持 ObjectSDF++ 官方预处理约定。`mask.npy` 是先验有效区域，实例掩码另存。priors 会跳过已存在的单项输出，可补完中断的阶段。

默认导出优先使用 `work/room01/split.json`，没有时才按参数选择互不重叠、沿时间分布的训练/测试帧；也可用 `--split /路径/split.json` 指定 prepared 帧编号：`{"train":[1,5,...],"test":[0,4,...]}`。修改 split 应在实例标注之前完成。最终 manifest 保留源视频帧和时间戳，避免两个方法因重编号使用不同源帧。

当前 DP-Recon 后续渲染固定需要 100 个相机。默认 export 要求至少 100 个注册帧，从真实恢复的相机中选 100 个，并包含全部测试相机；它们不是额外的 100 张训练图，也不是虚假的重复矩阵。训练 RGB/掩码只导出训练列表。相机恢复使用候选视频全部帧，包含测试视角，只用于求位姿；论文需披露这一协议，两种方法一致。

若 COLMAP 只注册了不足 100 帧，可用 `--render-count 0` 导出**基础训练格式**；此时完整 DP-Recon 的固定 100 相机逻辑尚需后续适配，脚本不会伪造相机。DP-Recon 辅助渲染当前使用统一内参，默认导出会检查这一必要条件。测试 RGB/掩码不应参与提示词选择、训练监督或模型调参。

## 输出如何给两个模型使用

两个模型共用 `datasets/custom/scan1`：

- 同时有 `instance_mask/*.png`、`segs/*_segs.png` 和各自映射表。
- 相机 NPZ 保存 `world_mat=K[R|t]` 和归一化 `scale_mat`；不是直接保存 c2w。
- 设置 `center_crop_type=no_crop`、`img_res=[384,384]`、`num_views=-1`（用导出的全部训练帧；原加载器 num_views 只允许 -1/3/6/9）。
- ObjectSDF++ 的 `model.implicit_network.d_out=N+1`，`use_mask=True`；它从 `../data/custom/scan1` 读取数据，后续可在独立运行目录安排 data 链接。此次不向基线项目写链接。
- DP-Recon 的 `data_root_dir` 指向 `.../datagen/datasets`，`data_dir=custom`，`scan_id=1`。
- 每个数据集附带 `CONFIG_NOTES.md`，说明实例数、路径和完整流程的限制。此工具不自动启动任何基线训练。

## 资源与验证

默认 GPU 操作只使用可见的第一张卡，推荐显式 `CUDA_VISIBLE_DEVICES=<空闲卡号>`；不用八卡并行抢占服务器。COLMAP 为 CPU 版 Python 接口 `pycolmap`，没有系统级 COLMAP 安装。ffmpeg 如需转码使用服务器现有只读命令。

部署验证包括一次版本/架构检查、小矩阵 GPU 运算、TorchVision GPU NMS、xFormers 小规模前向，及合成数据上的抽帧/相机变换/双格式导出测试。合成测试输出不是真实重建结果。没有素材时不能声称真实视频全链路已经验收。不会进行模型哈希、大目录递归扫描或循环重复验证。

真实视频没有完整三维 GT；可评估独立视角图像/人工核验轮廓，不能仅据此宣称完整几何 CD/F-score 胜出。
