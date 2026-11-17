# datagen 使用手册：视频生成 DP-Recon / ObjectSDF++ 数据集

更新日期：2026-11-17。以下命令均在你的 Linux 服务器执行。

## 1. 现在可以做什么

两个 Omnidata 权重已位于正确目录，并已用一张合成场景测试图完成正式权重加载和 GPU 推理。无需再次下载或重新配置环境。

```text
工具目录：/home/seed/wangruiyao/datagen
环境目录：/home/seed/miniconda3/envs/datagen
环境名称：datagen

models/
  sam2.1_hiera_small.pt
  omnidata_dpt_depth_v2.ckpt
  omnidata_dpt_normal_v2.ckpt
```

环境保持 Python 3.10、Torch 2.7.1、TorchVision 0.22.1、TorchAudio 2.7.1、xFormers 0.0.31、CUDA 12.8。旧 checkpoint 中的 Lightning 回调元数据已通过 datagen 内的专用兼容读取器处理，不需要安装 Lightning 训练框架。

你需要准备的是：一段静态场景视频，以及要重建哪些物体。脚本依次生成相机、实例掩码、深度、法线，并导出两项目共用的数据。实例提示和质量检查需要你参与；它不是完全自动的全场景物体发现工具。

第一次建议：一个房间、5–10 个物体、约 200 个候选帧、384×384，训练和测试各 10 帧。

## 2. 启动与目录约定

连接服务器后：

```bash
conda activate datagen
cd /home/seed/wangruiyao/datagen
```

**使用 `./datagen` 时，不用手动 `source scripts/env.sh`。** 启动器会自动设置独立缓存、源码路径和 CUDA 路径。不要重新运行 bootstrap 或下载脚本，也不用重新安装依赖。

手册统一使用以下示例，实际使用时替换视频路径；新视频换一个新的 work 名称：

```text
视频：/home/seed/wangruiyao/videos/room01.mp4   ← 示例，不代表文件已经存在
工作：/home/seed/wangruiyao/datagen/work/room01
导出：/home/seed/wangruiyao/datagen/datasets/custom/scan1
```

工作目录保存中间结果；导出目录保存训练数据。不要直接把输出放进 DP-Recon 或 ObjectSDF++ 的原始数据目录。

## 3. 准备视频

- 相机缓慢移动并有平移视差，覆盖目标物体多个方向；只站在原地转动手机通常不够。
- 保持焦距不变，不切换摄像头，不变焦；避免电子防抖造成明显动态裁切。
- 场景静止，尽量没有走动的人、移动的椅子；避免模糊、过曝和长时间对着纯白墙。
- 物体应在最终中心方形裁剪区域内，不能拍在画面最外侧后又被裁掉。
- 第一版尽量让目标物体在前段或第一张处理图中可见，方便给出稳定提示。

默认抽帧脚本采用“视频帧号 ÷ 名义帧率”记录时间。若手机素材是可变帧率且需要准确的时间对应，先用现有 ffmpeg 转为恒定帧率，例如：

```bash
# 可选：仅在确有需要时转码；输出用新文件名，不覆盖原视频。
ffmpeg -n -i /home/seed/wangruiyao/videos/room01.mp4 \
  -vf fps=30 -c:v libx264 -crf 18 -preset fast -an \
  /home/seed/wangruiyao/videos/room01_cfr.mp4
```

这一步不会修复严重运动模糊、相机变焦或场景运动。转码后，后续 `--video` 改为新文件。

## 4. 第一步：抽帧

```bash
./datagen extract \
  --video /home/seed/wangruiyao/videos/room01.mp4 \
  --work work/room01 \
  --fps 2 --max-frames 200
```

输出：

```text
work/room01/
  extraction.json       # 源视频帧号、时间、清晰度
  frames/000000.jpg     # COLMAP 使用的候选图
  frames/000001.jpg
  ...
```

`--fps 2` 表示约每秒取两帧，`--max-frames 200` 是上限，不会把短视频自动补成 200 帧。达到上限会停止读取，所以这组设置约覆盖前 100 秒；长视频需要先确定使用区间。

- 30 秒素材要取约 200 张，可从 `--fps 7 --max-frames 200` 起步。
- 60 秒素材可用 `--fps 4 --max-frames 200`。
- 截取第 10–70 秒可加 `--start 10 --end 70`。
- `--min-sharpness` 是可选的拉普拉斯清晰度阈值，数值依赖画面和分辨率；第一轮保持默认，不建议直接套用很高阈值。

目标是保留足够视角重叠，同时覆盖所需路径。**候选图多于 100 张不等于最终成功注册的相机一定多于 100 个。**

## 5. 第二步：COLMAP 恢复相机

```bash
./datagen sfm --work work/room01 --threads 4
```

当前使用 CPU 版 pycolmap，完成特征提取、相邻帧匹配、稀疏建图及去畸变。默认相邻匹配重叠窗口为 10，可通过 `--overlap` 调整。此步骤不使用 GPU。

成功后会显示类似：

```text
Registered 180 frames. Next: prepare.
```

具体注册数也保存在 `work/room01/sfm/result.json`。脚本选择注册图像最多的稀疏模型；没有成功注册的候选图不会进入后续 prepared 序列。

判断标准：

| 结果 | 下一步 |
|---|---|
| 至少 100 个注册相机 | 可尝试默认完整 DP-Recon 相机格式 |
| 不足 100 个，但满足训练/测试数量 | 可继续基础训练格式，导出时用 `--render-count 0` |
| 注册很少或完全失败 | 先解决素材/匹配问题，不要继续长时间先验生成 |

CPU 运行时间与图像分辨率、纹理和候选数量有关，不是固定时长。若有必要保留日志，可用：

```bash
# 这是首次执行 sfm 的替代写法，不要紧接着上面的命令重复执行。
./datagen sfm --work work/room01 --threads 4 > work/room01/sfm.log 2>&1
```

## 6. 第三步：统一图像、内参和场景坐标

```bash
./datagen prepare --work work/room01 --size 384
```

此步执行：去畸变图中心方形裁剪、缩放到 384×384、内参同步变换、世界 +Z 对齐估计、场景统一归一化、相机 NPZ 导出。

输出：

```text
work/room01/prepared/
  rgb/000000.jpg
  rgb/000001.jpg
  cameras.npz
  manifest.json
  reprojection_000000.jpg
  ...
  sparse_original_coordinates.ply
```

请检查两件事：

1. `rgb/`：目标物体是否清晰、是否被方形裁剪切掉。
2. `reprojection_*.jpg`：绿点与场景图像特征是否基本重合；明显错位时先停在这里排查。

未指定 `--up` 时，程序从摄像机竖直方向估计世界 up，不等同于真实地面法向。大致直立拍摄的室内视频更合适。若场景坐标明显倾斜，可在确认 COLMAP 原坐标系的 up 后指定 `--up X Y Z`，在新的 prepared 阶段重新处理；不要猜一个向量直接使用。原始稀疏 PLY 可用于辅助观察原坐标系。

**后续所有提示点、掩码都基于 prepared/rgb，而非原视频或 frames。** prepared 编号会因 COLMAP 丢帧而变化，不能直接沿用源视频帧号。

## 7. 第四步：先固定训练/测试划分

```bash
./datagen split --work work/room01 --train-views 10 --test-views 10
cat work/room01/split.json
```

输出是两个不重叠的 prepared 帧编号列表。自动划分沿时间分布，尽量保留第 0 帧作为训练提示锚点；划分仍需要查看目标物体是否在训练视角中可见。

- 10 个训练视角和 10 个测试视角都不是连续的前 10 帧。
- 如果想用 20 个训练视角，应在标注前调整 split。
- 需要自定义划分时，编辑 `split.json` 中的整数列表，保持范围有效、两组不重叠且无重复。
- **一旦存在 split.json，export 优先使用它。** 之后仅在 export 命令中写 `--train-views 20` 不会覆盖已有的 10 视角划分。
- 所有方法使用同一份划分；测试帧不能用于选模型参数、决定提示词或反复挑选最优结果。

## 8. 第五步：标注目标物体

先复制模板：

```bash
cp -n config/annotations.example.json work/room01/annotations.json
```

用你常用的编辑器修改 `work/room01/annotations.json`。当前没有网页点选界面，不会自动打开标注窗口；需要在能显示像素坐标的图片工具中查看 prepared/rgb 图片，记录点或框，再填 JSON。

可以先生成带 64 像素网格的辅助预览：

```bash
./datagen preview --work work/room01 --count 20
```

结果在 `work/room01/preview/`，原图不变。该预览会沿全序列选帧，并不保证覆盖你想标注的训练帧；具体标注应按 split 中的 train 编号打开对应 `prepared/rgb/XXXXXX.jpg`。

### 标注例子

下面是一个物体的合法 JSON 结构。**坐标和描述只是示例，必须替换成自己图片上的实际内容。** 增加物体时，继续追加 id=2、3… 的对象。

```json
{
  "background": {
    "geometry_prompt": "An empty room with walls, a floor and a ceiling.",
    "texture_prompt": "A room with white walls and a wooden floor."
  },
  "objects": [
    {
      "id": 1,
      "name": "chair_1",
      "geometry_prompt": "A chair with a backrest and four legs.",
      "texture_prompt": "A brown wooden chair.",
      "floor_supported": true,
      "prompts": [
        {
          "frame": 0,
          "points": [[160, 220], [20, 20]],
          "labels": [1, 0]
        }
      ]
    }
  ]
}
```

字段说明：

| 字段 | 如何填写 |
|---|---|
| `id` | 按数组顺序连续使用 1…N，N≤254；背景由程序使用 0 |
| `name` | 唯一名称，例如 chair_1、chair_2，同类的不同物体分开 |
| `geometry_prompt` | 英文形状/结构描述，根据训练视角填写 |
| `texture_prompt` | 英文颜色/材质描述，根据训练视角填写 |
| `floor_supported` | 该物体直接落地为 true，桌上花瓶为 false |
| `frame` | prepared 序列编号，必须属于 split.json 的 train 列表 |
| `points` | 图上 `[x,y]`；左上角为原点，x 向右、y 向下；384 图范围 0…383 |
| `labels` | 与 points 一一对应；1 为该物体内部点，0 为该物体外部点 |

也可用框替代 points/labels：

```json
{"frame": 0, "box": [100, 80, 260, 320]}
```

box 顺序为 `[左,上,右,下]`，右大于左、下大于上。框只包目标物体，尽量不包其他物体。一个提示使用点或框中的一种最容易排查。

第一版每个物体给一个清楚的训练帧提示即可；轮廓不稳、长时间遮挡、重新出现时，可在 prompts 中增加其他训练帧提示。当前实现从第 0 帧向后传播，不自动执行一次反向跟踪；如果物体只在后半段有提示，更早帧的结果需要特别核查。

## 9. 第六步：SAM2 跟踪和人工检查

先选一张空闲 GPU，下面以物理 GPU 1 为例，不代表它在你执行时一定空闲：

```bash
nvidia-smi

CUDA_VISIBLE_DEVICES=1 ./datagen segment \
  --work work/room01 \
  --annotations work/room01/annotations.json

./datagen preview --work work/room01 --count 20
```

`CUDA_VISIBLE_DEVICES=1` 后，程序内部仍使用默认 `cuda:0`，它对应被选中的物理卡 1；不用再加 `--device cuda:1`。

输出 `work/room01/labels/000000.png` 等。它们是单通道整数实例标签，直接查看可能几乎全黑，因为像素值是 0、1、2…；请看 preview 中的彩色叠加图。

检查：

- 同一个物体跨帧保持同一颜色/ID。
- 相邻的两把椅子没有交换编号，桌腿没有被错误分给背景或其他物体。
- 物体离开画面时没有大面积残留掩码，重新出现时没有换 ID。
- 只监督可见区域，不把被遮挡的整件物体涂到其他表面上。

需要修正时先改 annotations，再保留原标签后重跑，例如：

```bash
# 使用尚不存在的备份名；只移动本场景的标签结果。
mv -n work/room01/labels work/room01/labels_before_fix_01
CUDA_VISIBLE_DEVICES=1 ./datagen segment \
  --work work/room01 --annotations work/room01/annotations.json
./datagen preview --work work/room01 --count 20
```

人工提示只放在训练帧。对测试标签可另行核验供评价使用，不能将测试帧上的纠错重新传播回训练标签。SAM2 会读取候选视频帧进行跟踪，所得掩码属于伪标签；论文应披露这一预处理协议，两方法使用同一套标签。

## 10. 第七步：生成深度、法线和有效区域

```bash
CUDA_VISIBLE_DEVICES=1 ./datagen priors --work work/room01
```

不需要再给权重路径，默认使用已下载的 models 文件。脚本逐图推理，不会占满八张卡，也不会训练模型。

输出：

```text
work/room01/priors/
  000000_depth.npy     # float32，H×W
  000000_normal.npy    # float32，3×H×W，[0,1] 编码
  000000_mask.npy      # float32，H×W，几何先验有效区域
  ...
  provenance.json
```

注意：

- 本接口要求 prepare 使用 **384×384**，不会额外裁剪或悄悄缩放。
- 深度是相对单目先验，不是米制测量；不要自行反相或做逐图 min-max 归一化。
- `mask.npy` 不是实例分割；实例标签在 labels/。
- 不要把法线改成 H×W×3 再存储，两个模型读取的是 CHW。
- 本步骤会给全部 prepared 帧生成先验；导出训练数据时仅复制训练帧先验。网络测试监督仍应使用独立视角协议。
- 中断后可以原命令重跑，已存在的单项先验会跳过。若某个文件在异常中被写坏，先保留并移走该文件再重跑；存在即跳过不等于文件必然完整。

这一步可在实例标注前单独运行，因为它只依赖 prepared/rgb；为了先发现素材和标注问题，首轮建议按手册顺序进行。

## 11. 第八步：导出两项目共用数据

至少 100 个注册相机时：

```bash
./datagen export \
  --work work/room01 \
  --annotations work/room01/annotations.json \
  --output datasets/custom/scan1 \
  --data-dir custom --scan-id 1
```

导出使用已有 split.json。它会检查：训练/测试不重复、所需标签和先验形状正确、所有目标物体在所选训练帧中出现、相机格式有效。

若注册相机不足 100 个，可导出基础训练格式：

```bash
./datagen export \
  --work work/room01 \
  --annotations work/room01/annotations.json \
  --output datasets/custom/scan1 \
  --data-dir custom --scan-id 1 \
  --render-count 0
```

这两条是**二选一**，不是连续执行。`--render-count 0` 不影响基础训练数据，但不能直接满足当前 DP-Recon 后续固定读取 100 相机的完整流程，届时还需要适配。

默认的 100 相机来自已恢复的真实相机，其中包含测试相机；不是 100 张训练图片，不是复制重复矩阵。当前 DP 辅助渲染假设共享内参，脚本会验证这一点。

同一个场景导出第二个版本时，不覆盖原输出，换一个父目录并保持末尾 `custom/scan1`，例如 `datasets/revision02/custom/scan1`。第二个场景可用 `datasets/custom/scan2`，同时改 `--scan-id 2`。

输出示意：

```text
datasets/custom/scan1/
  000000_rgb.png
  000000_depth.npy
  000000_normal.npy
  000000_mask.npy
  ...
  cameras.npz
  instance_id.json
  instance_mapping.txt
  instance_mask/000000.png
  segs/000000_segs.png
  custom_scan1_prompt_geometry.json
  custom_scan1_prompt_texture.json
  custom_scan1_location_modify.json
  eval-novel-views/
    cameras.npz
    eval-10views.json
    rgb/
    instance_mask/
  datagen_manifest.json
  annotations.used.json
  CONFIG_NOTES.md
```

训练图会重新从 000000 编号，测试图按辅助相机编号保存，不必连续。源视频帧→prepared 编号→导出编号的对应信息在 manifest 中，不要仅凭不同目录下的同名文件判断是否同一视角。

## 12. 给 DP-Recon / ObjectSDF++ 使用

生成完数据后先读导出目录内的 `CONFIG_NOTES.md`。datagen 不会自动启动模型训练、修改项目配置或向原项目添加链接。

| 设置 | DP-Recon | ObjectSDF++ |
|---|---|---|
| `data_dir` | `custom` | `custom` |
| `scan_id` | `1` | `1` |
| `img_res` | `[384,384]` | `[384,384]` |
| `center_crop_type` | `no_crop` | `no_crop` |
| `num_views` | `-1`，使用导出的全部训练图 | 同左 |
| `use_mask` | 可设 true 使用生成的有效区域 | true |
| 数据根目录 | `data_root_dir=/home/seed/wangruiyao/datagen/datasets` | 原加载器按运行目录读取 `../data/custom/scan1` |
| 实例输出通道 | 按实际 N 个物体配置，训练器也有实例数处理 | `model.implicit_network.d_out=N+1`，包含背景 |

原加载器的 num_views 参数只接受 -1/3/6/9，**不要因为生成了 10 张训练图就把 num_views 写成 10**。10 张已经由 datagen 的 split/export 控制，训练时设 -1 即可。

ObjectSDF++ 路径需要在后续训练部署时安排独立运行目录或数据链接；本手册不自动改原始项目。若使用修订输出根目录，例如 `datasets/revision02/custom/scan1`，DP 的 data_root_dir 也相应改为 `.../datasets/revision02`。

两边应使用同一组帧、相机、标签和先验。自采数据仍可能触发完整 DP-Recon 中其他场景假设，数据格式兼容不等于原论文全部流程无需配置。

当前导出不生成真实三维 GT、测量尺度或官方 Replica 轨迹文件。普通 RGB 视频无法单独支撑完整几何的真实 CD/F-score 评价；建议原有带三维 GT 的数据集负责几何定量，自采视频负责真实场景泛化、独立视角图像/人工核验掩码的评价。SAM2 测试标签未经人工核验时只能作为代理标签，不能直接称人工真值。

## 13. 中断、重跑和常见错误

| 提示或问题 | 处理方法 |
|---|---|
| `Stage already exists` | extract/sfm/prepare/segment 不覆盖阶段目录；确认是否已完成。要重做则保留旧目录并换名，或新建 work |
| `Split already exists` | 不必重复 split；如确要改划分，标注前编辑 split.json |
| `Output already exists` | 另选新输出根目录，不覆盖已有数据集 |
| `Missing ... checkpoint` | 检查当前所用路径是否确为本手册目录；默认三份权重均已就位 |
| `No module named pytorch_lightning` | 当前 datagen 兼容读取器已修复；使用 `./datagen priors`，不要调用旧备份脚本或安装旧 Lightning 栈 |
| `Registered` 数量太少 | 增加视角重叠、改善纹理/模糊，必要时改用 exhaustive 匹配；纯旋转、动态物体不能靠改参数完全补救 |
| `Full DP-Recon export requires >=100 ...` | 增加成功注册相机，或选 `--render-count 0` 仅导出基础训练格式 |
| `Some target objects never appear ... TRAINING` | 检查掩码是否跟踪成功、裁剪是否切掉物体、训练划分是否覆盖物体；不要直接删除检查来绕过 |
| `Use training frames for SAM prompts` | annotations 的 frame 不属于 split.json 的 train 列表，改到训练帧 |
| `Unexpected prior shape` | 确认 prepare 使用 384，不混用不同 work 的先验 |
| CUDA 显存不足 | 换空闲卡，避免和其他训练同卡；不要为此改已固定的 Torch/CUDA 版本 |
| 掩码 PNG 看起来全黑 | 实例 ID 是小整数，查看 preview 彩色叠加图 |
| 断开 SSH 后任务中断 | 可用服务器已有的 tmux 保持会话；无需安装新系统软件才能运行工具 |

如果首次 SfM 完全失败，且还没有生成下游结果，可保留失败目录后用全匹配重试：

```bash
# 备份名必须尚不存在；全匹配计算量高于默认相邻匹配。
mv -n work/room01/sfm work/room01/sfm_failed_01
./datagen sfm --work work/room01 --threads 4 --matching exhaustive
```

如果已生成 prepared/labels/priors，改变抽帧、SfM 或裁剪后，后续结果的编号和几何可能全部失效。此时另建 work 更稳妥，不能把旧掩码和先验直接套到新图片上。只修改实例提示时，不必重跑相机或 Omnidata，只重跑 segment、检查并重新 export。

## 14. 最简执行顺序

以下用于回顾顺序，不是无人工介入的一键命令：

```bash
conda activate datagen
cd /home/seed/wangruiyao/datagen

./datagen extract --video /home/seed/wangruiyao/videos/room01.mp4 --work work/room01 --fps 2 --max-frames 200
./datagen sfm --work work/room01 --threads 4
./datagen prepare --work work/room01 --size 384
./datagen split --work work/room01 --train-views 10 --test-views 10
cp -n config/annotations.example.json work/room01/annotations.json

# 此处停下：查看 prepared/rgb 和 split，编辑 annotations，替换所有示例提示。

CUDA_VISIBLE_DEVICES=1 ./datagen segment --work work/room01 --annotations work/room01/annotations.json
./datagen preview --work work/room01 --count 20

# 此处检查掩码；有问题先修正，再继续。

CUDA_VISIBLE_DEVICES=1 ./datagen priors --work work/room01
./datagen export --work work/room01 --annotations work/room01/annotations.json --output datasets/custom/scan1 --data-dir custom --scan-id 1
```

每一阶段只在上一步成功后执行。GPU 编号、视频路径和场景名称按实际情况更换。
