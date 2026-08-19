<p align="center">
<img src="./deoxys_github_logo.png">
</p>

<p align="center">
<a href="https://github.com/UT-Austin-RPL/deoxys_control/actions">
<img alt="Tests Passing" src="https://github.com/anuraghazra/github-readme-stats/workflows/Test/badge.svg" />
</a>
<a href="https://github.com/UT-Austin-RPL/deoxys_control/graphs/contributors">
<img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/UT-Austin-RPL/deoxys_control" />
</a>
<a href="https://github.com/UT-Austin-RPL/deoxys_control/issues">
<img alt="Issues" src="https://img.shields.io/github/issues/UT-Austin-RPL/deoxys_control?color=0088ff" />
</a>


[**[Documentation]**](https://zhuyifengzju.github.io/deoxys_docs/html/index.html) &ensp; 

Deoxys is a modular, real-time controller library for Franka Emika Panda arm, aiming to facilitate a wide range of robot learning research. Deoxys comes with a user-friendly python interface and real-time controller implementation in C++. If you are a [robosuite](https://github.com/ARISE-Initiative/robosuite) user, Deoxys APIs provide seamless transfer 
from you simulation codebase to real robot experiments!




https://user-images.githubusercontent.com/21077484/206338997-8dbaa128-dc63-4911-84ca-64d80a05673f.mp4



## Cite our codebase

If you use this codebase for your research projects, please cite our codebase based on the following project:

```
@article{zhu2022viola,
  title={VIOLA: Imitation Learning for Vision-Based Manipulation with Object Proposal Priors},
  author={Zhu, Yifeng and Joshi, Abhishek and Stone, Peter and Zhu, Yuke},
  journal={arXiv preprint arXiv:2210.11339},
  doi={10.48550/arXiv.2210.11339},
  year={2022}
}
```


# Installation of codebase

Overall, the installation has three parts:
1. Install dependencies by running `InstallPackage`
2. Compile desktop-side codebase (Python)
3. Compile NUC-side codebase (C++)

Here are the details. For more information, please refer to the [Codebase Installation Page](https://ut-austin-rpl.github.io/deoxys-docs/html/installation/codebase_installation.html).

Clone this repo to the robot workspace directory on Desktop computer (e.g. `/home/USERNAME/robot-control-ws`)

``` shell
cd deoxys_control/deoxys
```

## Install dependencies

Run the `InstallPackage` file to install necessary packages.
``` shell
./InstallPackage
```


## Deoxys - Desktop

Make sure that you are in your python virtual environment before
	building this.
``` shell
make -j build_deoxys=1
```

And install all the python dependencies (feel free to add pull requests if anything is missing) from `deoxys_control/requirements.txt`, by doing:
```shell
pip install -U -r requirements.txt
```

## Franka Interface - Intel NUC

Franka Interface is the part which is supposed to run on NUC. Run this 
command in directory `deoxys_control/deoxys/` on Intel NUC. 

``` shell
make -j build_franka=1
```

## A laundry list of pointers:
   - [How to turn on/off the robot](https://ut-austin-rpl.github.io/deoxys-docs/html/tutorials/running_robots.html)
   - [How to install spacemouse](https://ut-austin-rpl.github.io/deoxys-docs/html/tutorials/using_teleoperation_devices.html)
   - [How to set up the RTOS](https://ut-austin-rpl.github.io/deoxys-docs/html/installation/system_prerequisite.html)
   - [How to record and replay a trajectory](https://ut-austin-rpl.github.io/deoxys-docs/html/tutorials/record_and_replay.html)
   - [How to write a simple motor program](https://ut-austin-rpl.github.io/deoxys-docs/html/tutorials/handcrafting_motor_program.html)

# Control the robot

## Commands on Desktop

Here is a quick guide to run `Deoxys`.

Under `deoxys_control/deoxys`,  run

``` shell
python examples/run_deoxys_with_space_mouse.py 
```

Change 1) spacemouse vendor_id and product_id ([here](https://github.com/UT-Austin-RPL/deoxys_control/blob/eb8d69f7f0838389fca81cac6b250ba05fc97f92/deoxys/examples/run_deoxys_with_space_mouse.py#L19)) 2) robot interface 
config ([here](https://github.com/UT-Austin-RPL/deoxys_control/blob/eb8d69f7f0838389fca81cac6b250ba05fc97f92/deoxys/examples/run_deoxys_with_space_mouse.py#L16)) if necessary.

You might also check and change the PC / NUC names [here](https://github.com/UT-Austin-RPL/deoxys_control/blob/master/deoxys/config/charmander.yml). 

## Commands on Control PC (Intel NUC)

Under `deoxys_control/deoxys`, run two commands. One for real-time control of the arm, one for non
real-time control of the gripper.

``` shell
bin/franka-interface config/charmander.yml
```

``` shell
bin/gripper-interface config/charmander.yml
```

# FurnitureBench 真机适配与环境搭建

本节命令均在 **FrankaControl 的图形桌面终端**中执行。相机标定程序会打开
OpenCV 窗口，因此不要在没有图形转发的普通 SSH 终端中运行。

微调前置相机、安装障碍物和静态环境测试时，机械臂可以保持关机，不需要启动
NUC 上的 `run2.sh`、`run3.sh`，也不要添加 `--prepare-robot`。只有机械臂挡住
前置相机视野、确实需要自动移动机械臂时，才需要另外启动 FCI 和 Deoxys server。

## 首次部署（当前 FrankaControl 已完成，可跳过）

`robust-rearrangement-custom` 和它的 FurnitureBench submodule 应放在
`YueHu_deoxys` 同级目录：

```shell
cd /home/hz/code
git clone --branch main --recurse-submodules \
  git@github.com:amorphophallus/robust-rearrangement-custom.git
git -C /home/hz/code/robust-rearrangement-custom submodule update --init --recursive
git -C /home/hz/code/YueHu_deoxys submodule update --init --recursive
```

FrankaControl 的 `/home/hz/.bashrc` 已配置以下环境变量，新机器部署时需要保持相同
目录结构。`DEOXYS_ROOT` 必须放在 `PYTHONPATH` 最前面，避免旧的 editable install
优先加载其他 Deoxys checkout。

```shell
export DEOXYS_ROOT=/home/hz/code/YueHu_deoxys/deoxys
export ROBUST_REARRANGEMENT_ROOT=/home/hz/code/robust-rearrangement-custom
export FURNITURE_BENCH="$ROBUST_REARRANGEMENT_ROOT/furniture-bench"
export RARL_SOURCE_DIR="$ROBUST_REARRANGEMENT_ROOT"
export DATA_DIR_RAW=/media/hz/e23044d0-8588-4f1e-b760-0912d3b4655d/robust-rearrangement-data
export DATA_DIR_PROCESSED="$DATA_DIR_RAW"
export PYTHONPATH="$DEOXYS_ROOT:$ROBUST_REARRANGEMENT_ROOT:$FURNITURE_BENCH${PYTHONPATH:+:$PYTHONPATH}"
```

首次安装 adapter 依赖：

```shell
source ~/.bashrc
conda activate deoxys
pip install "$FURNITURE_BENCH/wheels/dt_apriltags-3.2.0-py3-none-manylinux2010_x86_64.whl" \
  ipdb gym==0.26.2 huggingface-hub
```

## 每次开始 setup 前：进入环境

每打开一个新终端，先复制执行：

```shell
source ~/.bashrc
conda activate deoxys
cd /home/hz/code/YueHu_deoxys/deoxys
```

## 第 0 步：检查两台 RealSense

相机通过硬件序列号区分：

- front：`327122071654`，提供 RGB-D 和 AprilTag 零件定位。
- wrist：`001622071252`，当前 adapter 提供 RGB-D。

把相机接到 FrankaControl 的 USB 3 接口，然后执行：

```shell
rs-enumerate-devices -s
```

输出中应同时出现 `327122071654` 和 `001622071252`。当前 setup 流程会分别启动
front 和 wrist，兼容两台相机仍共享 `480M` USB 2.0 Hub 的情况。执行下面的
front-only 命令时可以临时拔掉 wrist；执行 wrist-only 命令时也可以临时拔掉
front。如果显示 `No device detected`，重新插拔相机后再次检查，不要继续标定。

## 第 1 步：微调 front camera

确认 base AprilTag 已平整固定、方向正确，然后复制执行：

```shell
python -m deoxys.examples.furniture_bench_setup_deoxys calibrate \
  --target setup_front
```

当前 FrankaControl 的标定默认 profile 为仅 RGB 的 `1280x720@15`：它与官方
参考图的尺寸和比例一致，并且能在当前 USB 2.0 链路上运行。标定不使用 depth。
程序会拒绝把 16:9 参考图拉伸到 4:3 profile，因此不要添加
`--width 640 --height 480`。无需手工指定 `--width`、`--height` 或 `--fps`。

调整时先让实时画面与透明参考图中的桌面边缘、机械臂底座和底座上的两个孔尽量
重合，然后先调位置、再小幅调旋转。这一步的官方视角只用于帮助安装 FurnitureBench
场景，不再作为最终 front camera 的硬性验收条件。初始参考条件：

- `x/y/z pos` 均为绿色，每轴误差不超过 `0.004 m`。
- `x/y/z rot` 均为绿色，每轴误差不超过 `0.8 deg`。
- 数字变绿的同时，实时画面轮廓也必须与透明参考图对齐。

当前人工调整结果可参考 2026-08-19 的标定截图；这是校准界面显示值，不是需要
写死到程序里的相机外参：

```text
x/y/z pos [m]   = [-0.0038, -0.1384, -0.0102]
x/y/z rot [deg] = [-12.0203, -0.4771, -0.2179]
```

该视角优先保证标准初始摆放下零件 AprilTag 可见、`valid` 全为 `1`；重新安装支架
或移动相机后仍须重新执行标定和第 4 步测试，不能直接照抄上述数值。

完成后按 `q` 或 `Esc` 退出。在完成第 2、3 步之前不要移动 base AprilTag 或
front camera；否则透明参考图不再对应真实视角。

## 第 2 步：安装障碍物

保持 front camera 和 base AprilTag 不动，复制执行：

```shell
python -m deoxys.examples.furniture_bench_setup_deoxys calibrate \
  --target obstacle
```

移动和旋转障碍物，使它与透明参考图中的障碍物完全重合。确认位置后用双面橡胶胶带
固定，并轻推检查障碍物不会滑动。完成后按 `q` 或 `Esc` 退出。

## 第 3 步：one-leg 环境最终静态校验

保持相机、base AprilTag 和障碍物不动，复制执行：

```shell
python -m deoxys.examples.furniture_bench_setup_deoxys calibrate \
  --target one_leg
```

确认六个误差数字全部变绿，并检查桌面边缘、机械臂底座、base AprilTag 和障碍物
都与参考图对齐。绿色背景应尽量平整、少褶皱，相机镜头应保持干净。完成后按
`q` 或 `Esc` 退出。

第 1～3 步完成后，场景中的 base AprilTag、障碍物和零件初始位置已经确定，后续
只能调整相机，不能再为了识别 Tag 而移动零件。最终相机视角允许偏离 benchmark，
取舍原则是优先保证完整、稳定的 `parts_poses`。

## 第 4 步：front-only AprilTag 与 valid 静态测试

这一步仍然不需要启动机械臂、FCI 或 NUC server，只启动 front 的
`1280x720@15` RGB，不启动 wrist 和 depth，因此兼容当前 `480M` 链路：

```shell
python -m deoxys.examples.furniture_bench_setup_deoxys test-front
```

把 one-leg 的 tabletop 和可动腿放在 FurnitureBench 指定位置，不要为了识别 Tag
而移动零件。程序在原始 `1280x720` 图像上检测 base tag 和零件 tag。如果指定位置
下不能得到完整位姿，应小幅升高 front camera 并增加俯视角，同时确保机械臂、完整
工作区和所有关键零件仍在画面中央。允许牺牲一部分与 benchmark 参考图的对齐。

移动相机后必须按 `q` 或 `Esc` 退出并重新执行 `test-front`。程序只在启动阶段估计
并锁定 `camera_to_april`，不能在同一进程中移动相机后继续使用旧外参。最终验收应在
相机已经固定、零件保持标准初始位置的全新进程中进行，目标是：

```text
found=[1, 0, 0, 0, 1, 0] valid=[1, 1, 1, 1, 1, 1] base=10/10 PASS
```

`found[0]` 和 `found[4]` 分别对应当前帧中的 tabletop 和可动腿，应稳定为 `1`。
`valid` 是锁存状态，曾经识别成功后会保持为 `1`；所以只有当前帧的 `found[0]`、
`found[4]` 也同时为 `1` 时程序才显示 `PASS`。`base=10/10` 表示程序已对 10 个
高分辨率 base-tag 外参结果求平均，避免把第一帧的角点噪声锁存为整段数据的相机
外参。硬性验收目标是 `valid=[1, 1, 1, 1, 1, 1]`；不要只看已经锁存的 `valid`，
还应观察至少 10 秒，确认 `found[0]` 和 `found[4]` 大多数帧同时为 `1`。验收后锁紧
支架并标记位置，数采和测试期间不得再移动相机。

## 第 5 步：wrist-only RGB-D 与帧率测试

只启动 wrist 的 `640x480@30` RGB-D，不运行 AprilTag，也不启动 front：

```shell
python -m deoxys.examples.furniture_bench_setup_deoxys test-wrist
```

窗口左侧是腕部 RGB，右侧是对齐到 RGB 视角的 depth。终端持续输出实测 FPS；稳定
运行后应大于 `10 FPS`，目标接近 `30 FPS`。确认画面方向、工作距离和深度图正常后，
按 `q` 或 `Esc` 退出。

以上两个测试只读取单台相机，不连接 Deoxys/NUC，也不会发送机械臂 action。

## SpaceMouse 真机数采

### 数采前检查

每次调整或重新插拔 front camera 后，必须先重新执行第 4 步，并在全新进程中达到
`valid` 全 1、`base=10/10`。数采脚本启动时会重新估计外参，因此不要复用调整相机
之前已经启动的数采进程。相机位置一旦改变，正在录制或尚未保存的 episode 必须
丢弃。

启动机械臂、夹爪 FCI 和 Deoxys server 后，在 FrankaControl 的图形桌面终端执行：

```shell
source ~/.bashrc
conda activate deoxys
cd /home/hz/code/YueHu_deoxys/deoxys
df -h "$DATA_DIR_RAW"
rs-enumerate-devices | grep -E '327122071654|001622071252'
```

### USB 3.x / 5000M 正式配置

新线到达且 `lsusb -t` 中两台相机均显示 `5000M` 后，直接使用默认配置：

```shell
python -m deoxys.examples.run_deoxys_with_space_mouse_V3_record \
  --interface-cfg config/charmander.yml \
  --controller-type OSC_POSE \
  --vendor-id 9583 \
  --product-id 50746 \
  --draw-part-poses \
  --prompt-depth-anything \
  --prompt-depth-model vitl \
  --prompt-depth-cameras both \
  --prompt-depth-max-size 448 \
  --prompt-depth-colormap viridis
```

默认配置是 front `1280x720@30` RGB-D、wrist `640x480@30` RGB-D、数据记录
`10 Hz`。这套配置不能在当前两台相机共享的 `480M` Hub 上运行。

### 当前 480M / USB 2.1 降级配置

在新线到达前，可以使用下面这套已经在 FrankaControl 两台相机同时连接、AprilTag
tracker 开启时实测通过的组合。front 约 `10 FPS`，wrist 约 `15 FPS`，满足当前
`10 Hz` 数采的最低要求：

```shell
python -m deoxys.examples.run_deoxys_with_space_mouse_V3_record \
  --interface-cfg config/charmander.yml \
  --controller-type OSC_POSE \
  --vendor-id 9583 \
  --product-id 50746 \
  --front-color-width 1280 \
  --front-color-height 720 \
  --front-color-fps 10 \
  --front-depth-width 848 \
  --front-depth-height 480 \
  --front-depth-fps 10 \
  --wrist-color-width 424 \
  --wrist-color-height 240 \
  --wrist-color-fps 15 \
  --wrist-depth-width 480 \
  --wrist-depth-height 270 \
  --wrist-depth-fps 15 \
  --record-fps 10 \
  --draw-part-poses \
  --prompt-depth-anything \
  --prompt-depth-model vitl \
  --prompt-depth-cameras both \
  --prompt-depth-max-size 448 \
  --prompt-depth-colormap viridis
```

这套降级配置保留 front `1280x720`，因此不会牺牲 AprilTag 检测所需的 front RGB
像素；主要降低 front depth 和 wrist RGB-D 的带宽。两台相机仍共用一条 USB 2.1
链路，余量很小，只适合临时采集。启动后如果出现帧超时、FPS 下降或 USB 错误，应
停止本次采集并等待 `5000M` 线，不要继续保存不完整 episode。USB2 配置的 wrist
原图为 16:9 的 `424x240`，写入 pickle 时会中央裁出 `320x240`；与正式 USB3 的
`640x480 -> 320x240` 相比会损失左右视野，因此采集前必须确认腕部关键操作区域仍
完整可见。

### 按键与保存结果

脚本默认实时显示写入 pickle 前的 wrist/front RGB 拼接画面；推荐命令均添加
`--draw-part-poses`，在 front 画面上绘制 `P0 tabletop` 和 `P4 movable_leg` 的
三维坐标轴。坐标轴使用与 front camera setup 相同的 `camera_to_april` 求逆、
Rodrigues 和 `cv2.drawFrameAxes` 投影流程。绿色 `FOUND` 表示当前帧成功检测，黄色
`STALE` 表示暂时使用上一次检测位姿及其 age。配置初始化的 `P1/P2/P3` 和固定障碍物
`P5` 不会伪装成实时检测结果。

按键：`b` 开始、`e` 结束、`s` 保存为成功、`f` 保存为失败、`d` 丢弃、`r` 关节
复位、`p` 实时开关 part-pose 绘制、`q` 退出。OpenCV 预览窗口获得焦点时这些按键
同样有效。只有 front 已经得到 tabletop 和可动腿的位姿时才允许开始录制。短暂的
AprilTag 遮挡会保留最后一次位姿，同时用 `parts_founds`、`parts_pose_valid` 和
`parts_pose_age_ms` 标记是否为当前帧检测以及位姿新鲜度。

预览和坐标轴只用于屏幕显示，不会写入 pickle RGB 或保存的 MP4。没有图形桌面或
通过普通 SSH 启动时，添加 `--no-camera-preview`；该参数只关闭窗口，不影响相机
采集和 `parts_poses` 计算。

原始 episode 保存到：

```text
$DATA_DIR_RAW/raw/osc/real/one_leg/teleop/low/{success|failure}/
```

每个 pickle 包含 `N+1` 个 observation、`N` 个 8 维 delta action 和 `N` 个 reward。
`color_image1`/`depth_image1` 是 wrist，`color_image2`/`depth_image2` 是 front。
RGB 为 `240x320 uint8`，对齐到 RGB 的 depth 为正米制 `240x320 float16`。
`parts_poses` 是 FurnitureBench AprilTag 坐标系中的 5 个 one-leg 零件加障碍物，共
42 个数值。action 格式为 `[dx, dy, dz, dqx, dqy, dqz, dqw, gripper]`：平移单位
为米，四元数顺序为 `xyzw`，旋转是末端局部坐标系右乘 delta。

正式数采默认按 USB 3.x/`5000M` 配置：front RGB-D 为 `1280x720@30`，wrist
RGB-D 为 `640x480@30`。图像处理没有 `1280x720 -> 640x480 -> 320x240` 这样的
两级缩放：

- front AprilTag 始终直接使用原始 `1280x720` RGB。
- front 写入 pickle 前先从原图中央裁出 `960x720`，即左右各去掉 160 像素，再用
  `INTER_AREA` 等比例缩小为 `320x240`；不会把 16:9 拉伸成 4:3。
- front depth 先由 RealSense 对齐到 `1280x720` RGB 视角，再执行完全相同的裁剪，
  最后用 `INTER_NEAREST` 缩小为 `320x240`，避免生成不存在的深度插值值。
- wrist 原图已经是 `640x480` 的 4:3，只用 `INTER_AREA` 等比例缩小为
  `320x240`；wrist depth 使用相同几何变换和最近邻缩放。

`camera_info` 会同时保存每台相机的原始 color/depth profile、原始内参、裁剪窗口、
缩放比例以及变换后的 `320x240` 内参。每个 observation 还保存本次启动重新估计的
`camera_to_april`。因此移动相机后不需要改变 pickle schema，但必须重启脚本获得新
外参。在线 AprilTag 使用原始内参；后续从 pickle 图像重建点云或做离线几何计算时，
应使用 `record_intrinsics`。

pickle 和左右相机拼接 MP4 会由后台线程先写临时文件，再原子重命名。退出程序前应
等待保存完成，不要在终端刚显示保存按键后立即关机。

## Prompt Depth Anything 深度增强

本仓库把官方 Prompt Depth Anything 固定为 `third_party/PromptDA` submodule。
新机器第一次使用前复制执行：

```shell
source ~/.bashrc
conda activate deoxys
cd /home/hz/code/YueHu_deoxys
git submodule update --init --recursive
pip install huggingface-hub
export HF_ENDPOINT=https://hf-mirror.com
```

推荐 ViT-L、`max-size 448` 和 `320x240` 保存分辨率，因为双相机实测约
`16.79 observation/s`，能够覆盖当前 `10 Hz` 数采频率且已经确认视觉效果。

### 方案一：在线预览并保存增强深度

SpaceMouse record 新增的 PromptDA 参数会实时显示 wrist/front 的 RGB、原始 depth
和增强 depth，同时把增强后的 `320x240 float16` 米制 depth 保存到 pickle 的
`depth_image1/2`；原始 RealSense depth 保存在 `depth_image1/2_realsense`。

```shell
source ~/.bashrc
conda activate deoxys
cd /home/hz/code/YueHu_deoxys/deoxys
export HF_ENDPOINT=https://hf-mirror.com

python -m deoxys.examples.run_deoxys_with_space_mouse_V3_record \
  --interface-cfg config/charmander.yml \
  --controller-type OSC_POSE \
  --vendor-id 9583 \
  --product-id 50746 \
  --record-image-width 320 \
  --record-image-height 240 \
  --record-fps 10 \
  --draw-part-poses \
  --prompt-depth-anything \
  --prompt-depth-model vitl \
  --prompt-depth-cameras both \
  --prompt-depth-max-size 448 \
  --prompt-depth-colormap viridis
```

### 方案二：离线处理已有 pickle

对以前保存的原始 RealSense pickle 使用下面的命令；脚本不会修改输入文件，而是生成
新的 ViT-L 增强 pickle、指标 JSON 和双相机对比 MP4。

```shell
python -m deoxys.examples.process_pickle_prompt_depth \
  "$DATA_DIR_RAW/raw/osc/real/one_leg/teleop/low/success/示例.pkl" \
  --model vitl \
  --max-size 448 \
  --cameras both \
  --comparison-video
```

输出文件名为 `示例_promptda_vitl.pkl`、`示例_promptda_vitl.metrics.json` 和
`示例_promptda_vitl_comparison.mp4`；新 pickle 的字段、分辨率和单位与在线方案一致。
