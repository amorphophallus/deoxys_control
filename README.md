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
export RR_STORAGE_ROOT=/media/hz/e23044d0-8588-4f1e-b760-0912d3b4655d/robust-rearrangement-data
export DATA_DIR_RAW="$RR_STORAGE_ROOT"
export DATA_DIR_PROCESSED="$RR_STORAGE_ROOT"
export RR_CHECKPOINT_ROOT="$RR_STORAGE_ROOT/checkpoints"
export PYTHONPATH="$DEOXYS_ROOT:$ROBUST_REARRANGEMENT_ROOT:$FURNITURE_BENCH${PYTHONPATH:+:$PYTHONPATH}"
```

FrankaControl 的大体积数据和 checkpoint 优先放在 `RR_STORAGE_ROOT` 所在的 15 TB
本地盘；`/home/hz` 只保留代码、环境和小型日志。当前目录约定为：原始数据放在
`$DATA_DIR_RAW/raw/`，处理后数据放在 `$DATA_DIR_PROCESSED/processed/`，checkpoint
放在 `$RR_CHECKPOINT_ROOT/<campaign>/`。开始任务前用 `df -h "$RR_STORAGE_ROOT"`
确认该盘已挂载；如果命令显示根分区 `/dev/nvme1n1p2`，不要继续写入大文件。

首次安装 adapter 依赖：

```shell
source ~/.bashrc
conda activate deoxys
pip install "$FURNITURE_BENCH/wheels/dt_apriltags-3.2.0-py3-none-manylinux2010_x86_64.whl" \
  ipdb gym==0.26.2 huggingface-hub
```

## 每次开始 setup 前：进入环境

下面所有 FrankaControl 命令都从 `YueHu_deoxys` 仓库根目录执行。每打开一个新终端，
先复制执行：

```shell
source ~/.bashrc
conda activate deoxys
cd /home/hz/code/YueHu_deoxys
```

## 第 0 步：检查两台 RealSense

相机通过硬件序列号区分：

- front：`327122071654`，提供 RGB-D 和 AprilTag 零件定位。
- wrist：`001622071252`，当前 adapter 提供 RGB-D。

把相机接到 FrankaControl 的 USB 3 接口，然后复制执行：

```shell
lsusb -t
rs-enumerate-devices 2>&1 | grep -E \
  'Name|Serial Number|Physical Port|Usb Type Descriptor|Could not create device'
```

当前正式接线是 front 使用 USB 3.x 好线、wrist 使用 USB 2.1 旧线。必须同时满足
以下条件才算通过：

- `lsusb -t` 中 front 的 RealSense `Video` 接口显示 `5000M`；wrist 允许显示
  `480M`。
- `rs-enumerate-devices` 同时列出 front `327122071654` 和 wrist
  `001622071252`；front 的 `Usb Type Descriptor` 应为 `3.x`，wrist 允许为
  `2.1`。
- 输出中没有 `Could not create device`、`xioctl` 或 UVC control timeout。

`lsusb -t` 只证明物理 USB 链路的协商速度，不能单独证明 librealsense 可以正常
打开相机。如果第二条命令缺少任一序列号，仍然判定为失败；重新插紧相机端
USB-C、检查线缆和 Hub 后再次执行，不要继续标定或数采。下面的 front-only /
wrist-only 命令可用于单相机排查。

## 第 1 步：微调 front camera

确认 base AprilTag 已平整固定、方向正确，然后复制执行：

```shell
python -m deoxys.examples.furniture_bench_setup_deoxys calibrate \
  --interface-cfg deoxys/config/charmander.yml \
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
  --interface-cfg deoxys/config/charmander.yml \
  --target obstacle
```

移动和旋转障碍物，使它与透明参考图中的障碍物完全重合。确认位置后用双面橡胶胶带
固定，并轻推检查障碍物不会滑动。完成后按 `q` 或 `Esc` 退出。

## 第 3 步：one-leg 环境最终静态校验

保持相机、base AprilTag 和障碍物不动，复制执行：

```shell
python -m deoxys.examples.furniture_bench_setup_deoxys calibrate \
  --interface-cfg deoxys/config/charmander.yml \
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

#### 第 0 步：打开 Franka Desk 并启动 NUC server

FrankaControl 和机械臂控制柜保持连接到 `172.16.0.0/24` 有线控制网。在
FrankaControl 的浏览器打开 Franka Desk：

```text
https://172.16.0.2
```

首次访问可能出现自签名证书警告，确认地址为 `172.16.0.2` 后继续访问。登录 Desk，
解除急停并解锁关节，然后激活 FCI。Desk 只能从能到达机器人控制网的机器访问，
Tailscale/ZeroTier 地址不能代替这里的 `172.16.0.2`。

常规数采在 NUC 的两个独立终端分别启动 arm 和 gripper 自动重启脚本：

```shell
# NUC 终端 1：arm server
cd /home/mingyu/code/deoxys_control/deoxys
bash run2.sh
```

```shell
# NUC 终端 2：gripper server
cd /home/mingyu/code/deoxys_control/deoxys
bash run3.sh
```

`run2.sh` 调用 `auto_scripts/auto_arm.sh`，`run3.sh` 调用
`auto_scripts/auto_gripper.sh`。`run1.sh` 是不带自动重启的单次 arm server，只在排查
arm 退出原因时临时使用；不要同时运行 `run1.sh` 和 `run2.sh`。正常数采固定使用
`run2.sh + run3.sh`。可在 NUC 的第三个终端检查：

```shell
pgrep -af 'franka-interface|gripper-interface'
```

确认两个进程都存在且没有反复退出重启后，再到 FrankaControl 启动下面的 SpaceMouse
数采程序。

每次调整或重新插拔 front camera 后，必须先重新执行第 4 步，并在全新进程中达到
`valid` 全 1、`base=10/10`。数采脚本启动时会重新估计外参，因此不要复用调整相机
之前已经启动的数采进程。相机位置一旦改变，正在录制或尚未保存的 episode 必须
丢弃。

启动机械臂、夹爪 FCI 和 Deoxys server 后，在 FrankaControl 的图形桌面终端执行：

```shell
source ~/.bashrc
conda activate deoxys
cd /home/hz/code/YueHu_deoxys
df -h "$DATA_DIR_RAW"
rs-enumerate-devices | grep -E '327122071654|001622071252'
lsusb -d 256f:
```

### SpaceMouse 有线/无线配置与一次性权限安装

无线连接时 SpaceMouse Universal Receiver 应显示为 `256f:c652`（十进制 product
ID `50770`）；有线连接使用 `256f:c63a`（十进制 product ID `50746`）。首次使用
任一种连接，或设备存在但程序报 `OSError: open failed` 时，执行一次：

```shell
cd /home/hz/code/YueHu_deoxys
bash deoxys/installation/create_spacemouse.sh
```

该命令会要求输入 FrankaControl 本机的 sudo 密码，因为它需要向系统目录
`/etc/udev/rules.d` 写入规则并刷新 USB/hidraw 设备节点。sudo 仅在安装权限规则时
使用一次；不要用 sudo 启动数采程序，否则输出数据可能变成 root 所有。脚本同时
覆盖有线 `c63a` 和无线 `c652`。执行后用当前连接对应的一条命令验证：

```shell
# 无线
python -c "from deoxys.utils.io_devices import SpaceMouse; d=SpaceMouse(vendor_id=9583, product_id=50770); print('wireless SpaceMouse: OK'); d.close()"

# 有线
python -c "from deoxys.utils.io_devices import SpaceMouse; d=SpaceMouse(vendor_id=9583, product_id=50746); print('wired SpaceMouse: OK'); d.close()"
```

若仍报 `OSError: open failed`，重新插拔 SpaceMouse/接收器后再执行验证命令。

数采脚本默认使用更稳定的有线连接。切换到无线接收器时加
`--spacemouse-connection wireless`；`--product-id` 仅用于覆盖未知型号的 ID。

### 当前正式混合配置（front USB 3.x + wrist USB 2.1）

确认 front 显示 `5000M`、wrist 显示 `480M` 且两台相机都能被
`rs-enumerate-devices` 枚举后，直接使用默认配置：

```shell
source ~/.bashrc
conda activate rr-real
export DEOXYS_REPO=/home/hz/code/YueHu_deoxys
cd "$DEOXYS_REPO"

python -m deoxys.examples.run_deoxys_with_space_mouse_V3_record \
  --interface-cfg "$DEOXYS_REPO/deoxys/config/charmander.yml" \
  --latency-profile /home/hz/code/robust-rearrangement-custom/src/real/latency_profile.measured_20260908.json \
  --controller-type OSC_POSE \
  --annotation-source scripted \
  --output-suffix one-leg-umi-scripted-rgbd-202609 \
  --vendor-id 9583 \
  --spacemouse-connection wired \
  --draw-part-poses \
  --real-skill-annotation \
  --prompt-depth-anything \
  --prompt-depth-model vitl \
  --prompt-depth-cameras both \
  --prompt-depth-max-size 448 \
  --prompt-depth-colormap viridis
```

默认配置是 front `1280x720@30` RGB-D、wrist RGB `424x240@30` + depth
`480x270@30`、数据记录 `10 Hz`。该组合已在 FrankaControl 当前接线下双相机
并发实测通过。

`--output-suffix` 是本次 campaign 的独立目录名，每次新 campaign 必须更换；文件
写入 `.../teleop/low/<output-suffix>/{success|failure}/`，禁止与历史 pickle 混放。
`--annotation-source scripted` 是必填且唯一允许的 policy target provenance。
要得到完整训练标注，还需要 `--real-skill-annotation` 和两台相机的离线 PromptDA；
少了任一项，脚本会在启动时警告。现在仍允许保存 pickle，但会标成 `incomplete`，
并在同名 `.txt` 记录原因。实时 dashboard 显示与最终 pickle 标注是两套独立计算，
不能用屏幕上的标注代替离线标注。

正式 one-leg 数采统一使用上面的 `rr-real` 环境。2026-09-01 20:32 的第一次启动
命令在仓库根目录传入了不存在的 `config/charmander.yml`；正确文件是上面使用的绝对
路径。脚本现在会在启动相机前验证该文件，并把启动阶段写到控制台及
`logs/debug.log`；未捕获异常还会连同 `stage=dual_realsense_initialization`、
`prompt_depth_initialization`、`spacemouse_initialization`、
`franka_interface_initialization`、`robot_state_wait` 或 `controller_warmup` 写入
`logs/error.log`，用于区分视觉、输入设备和机器人状态错误。

当前录制控制已恢复为原来的 SpaceMouse 直接控制：每次循环读取**当前**摇杆状态，
生成 `OSC_POSE` 的 delta action，调用一次同步的 `robot_interface.control()`；
`FrankaInterface` 使用默认 `20 Hz` 控制周期，沿用原有平移/旋转缩放及限速设置。
不积分成绝对 pose、不排队等待发送旧动作，也不使用 `--teleop-fps` 改变控制频率。
`control()` 阻塞时本轮会等待，下轮重新读取当前摇杆输入；不会把卡顿期间的历史
摇杆动作依次补发。

相机抓帧和 episode 原始帧缓存分别在后台运行，观测拼装、实时 FSM 与 dashboard
预览在独立的异步 worker 中运行，不进入上述控制调用。按 `b` 后先预热相机 3 秒，
这期间不采样或发送 SpaceMouse 动作；按 `e` 后再留 1 秒相机/状态 post-roll。
如果连续 1 秒没有新的 front+wrist RGB-D 配对帧，dashboard 会显示红色
`CAMERA STALLED` 横幅，控制台也会每秒报警；相机未恢复前不会开始新的 episode。
录制中出现该警告时应立即按 `e` 停止并检查 RealSense/USB。该次冻结会写入同名
`.txt` 与 pickle 的 `save_quality`，并将 episode 保存到 `incomplete/`；离线处理还会
检查相机覆盖是否提前结束，避免把被静默裁短的数据误判为完整数据。
实时 FSM 在 `one_leg`、`round_table`、`lamp` 加 `--real-skill-annotation` 且打开预览时启用；dashboard
会显示 `SKILL / skill_state` 和 front/wrist 可见的二维目标点。`FSM: OFF`、
`WAITING FOR POSES` 或 `ERROR` 表示当前没有可用实时标注，不能误认作无目标点。

每次成功发送都保存当轮原始 delta、审计用的绝对 wrist target 和真实 command
timestamp，并用 `command timestamp + calibrated latency` 估算设备生效时间。
按 `e` 后才建立固定 10 Hz 的 `t_k`：arm 和 gripper action 分别选择
`effect_time <= t_k` 的最新已发送命令，不会使用未来 action。两台相机按曝光时间选
最近帧，掉帧时允许复用上一帧；robot pose/joint 按修正后的状态时间插值，gripper width
线性插值。相机 residual 超过 `50 ms` 或双相机 skew 超过 `40 ms` 只写 warning 和质量
报告；单台相机在目标附近 `200 ms` 内完全没有帧才判定为真实 coverage gap。对应参数是
`--camera-match-max-residual-ms`、`--camera-pair-max-skew-ms` 和
`--camera-hard-gap-ms`。相机硬缺帧、robot state 超过 `20 ms` 或 gripper state 超过
`60 ms` 时，对齐继续使用最近可用相机帧或插值后的本体状态，记录超阈值的 timestep 和最大残差；
PromptDA 与离线标注仍继续运行。保存时该条进入 `incomplete/{success|failure}/`，
并在同名 `.txt` 和 pickle 的 `alignment_report`/`save_quality` 中留下质量记录，
不会把时间质量不足的数据默认为可直接训练的完整样本。若原始流为空等导致无法构造
observation，仍保存原始流供之后恢复，但不能凭空生成增强深度。

pickle 继续使用训练 pipeline 已识别的
`deoxys_furniturebench_raw_v6_offline_buffered` schema，其中
`action_target_timestamps_ns` 是唯一主时间轴，`action_timestamps_ns` 只是相同值的
兼容别名。训练兼容字段 `actions` 仍是 8 维 delta；`actions_absolute`、
`raw_arm_commands_absolute`、`raw_gripper_commands` 和 `raw_spacemouse_samples` 保留
原始 delta、由当时 EE pose 推算的绝对目标及发送审计信息；绝对目标不是在线控制输入。
默认读取上面显式指定的实测 profile：arm action `120 ms`、gripper action
`642 ms`、robot/gripper observation 各 `0.067 ms`；命令行中
`--latency-profile` 会覆盖单独的 latency 参数。相机和状态保留各自的 source/receive
时间、两个 action 通道的采样/发送时间及最终 residual 报告。按 `e` 后处理可能需要等待，
终端打印 PromptDA/标注进度；显示 `materialized` 后才使用 `s`/`f` 保存。

### Round-table 数采（默认配置，可直接复制）

先完成 `--target round_table` 的相机与零件初始位置 setup，再运行下面的命令。
这里只增加 `--task-name round_table`，相机 profile、记录频率和 PromptDA 都沿用
当前默认值：

此命令同时启用实时 FSM 和按 `e` 后的离线标注。push 只有达到 FurnitureBench
几何目标才转入 leg pick，不以松爪代替；各装配步骤也必须满足几何判定。

```shell
source ~/.bashrc
conda activate deoxys
cd /home/hz/code/YueHu_deoxys

python -m deoxys.examples.run_deoxys_with_space_mouse_V3_record \
  --interface-cfg deoxys/config/charmander.yml \
  --latency-profile /home/hz/code/robust-rearrangement-custom/src/real/latency_profile.measured_20260908.json \
  --controller-type OSC_POSE \
  --annotation-source scripted \
  --output-suffix round-table-fsm-202609 \
  --vendor-id 9583 \
  --spacemouse-connection wired \
  --task-name round_table \
  --draw-part-poses \
  --real-skill-annotation \
  --prompt-depth-anything \
  --prompt-depth-model vitl \
  --prompt-depth-cameras both \
  --prompt-depth-max-size 448 \
  --prompt-depth-colormap viridis
```

front 预览会绘制 `P0 round_table_top`、`P1 round_table_leg` 和
`P2 round_table_base`。首次按 `b` 前建议确认 `valid=111`。完整数据保存在
`<output-suffix>/{success|failure}/`；标注或其他质量检查失败时仍可保存到：

```text
$DATA_DIR_RAW/raw/osc/real/round_table/teleop/low/<output-suffix>/incomplete/{success|failure}/
```

### Lamp 数采（默认配置，可直接复制）

先完成 `--target lamp` 的相机与零件初始位置 setup，再运行下面的命令。相机
profile、记录频率和 PromptDA 沿用当前默认值：

此命令同时启用实时 FSM 和按 `e` 后的离线标注。base push 只有达到几何目标
才进入 bulb pick；bulb 和 hood 的装配完成也要求对应的几何判定。

```shell
source ~/.bashrc
conda activate deoxys
cd /home/hz/code/YueHu_deoxys

python -m deoxys.examples.run_deoxys_with_space_mouse_V3_record \
  --interface-cfg deoxys/config/charmander.yml \
  --latency-profile /home/hz/code/robust-rearrangement-custom/src/real/latency_profile.measured_20260908.json \
  --controller-type OSC_POSE \
  --annotation-source scripted \
  --output-suffix lamp-fsm-202609 \
  --vendor-id 9583 \
  --spacemouse-connection wired \
  --task-name lamp \
  --draw-part-poses \
  --real-skill-annotation \
  --prompt-depth-anything \
  --prompt-depth-model vitl \
  --prompt-depth-cameras both \
  --prompt-depth-max-size 448 \
  --prompt-depth-colormap viridis
```

front 预览会绘制 `P0 lamp_base`、`P1 lamp_bulb` 和 `P2 lamp_hood`。首次按
`b` 前建议确认 `valid=111`。完整数据保存在
`<output-suffix>/{success|failure}/`；标注或其他质量检查失败时仍可保存到：

```text
$DATA_DIR_RAW/raw/osc/real/lamp/teleop/low/<output-suffix>/incomplete/{success|failure}/
```

### 完整参数命令（与默认配置相同）

需要显式固定每个相机 profile 时，复制下面的完整命令：

```shell
python -m deoxys.examples.run_deoxys_with_space_mouse_V3_record \
  --interface-cfg deoxys/config/charmander.yml \
  --latency-profile /home/hz/code/robust-rearrangement-custom/src/real/latency_profile.measured_20260908.json \
  --controller-type OSC_POSE \
  --annotation-source scripted \
  --output-suffix one-leg-umi-scripted-rgbd-202609 \
  --vendor-id 9583 \
  --spacemouse-connection wired \
  --front-color-width 1280 \
  --front-color-height 720 \
  --front-color-fps 30 \
  --front-depth-width 1280 \
  --front-depth-height 720 \
  --front-depth-fps 30 \
  --wrist-color-width 424 \
  --wrist-color-height 240 \
  --wrist-color-fps 30 \
  --wrist-depth-width 480 \
  --wrist-depth-height 270 \
  --wrist-depth-fps 30 \
  --record-fps 10 \
  --draw-part-poses \
  --real-skill-annotation \
  --prompt-depth-anything \
  --prompt-depth-model vitl \
  --prompt-depth-cameras both \
  --prompt-depth-max-size 448 \
  --prompt-depth-colormap viridis
```

该配置保留 front `1280x720`，确保 AprilTag 检测所需的 front RGB 像素；wrist
通过较低分辨率在 USB 2.1 上提供 30 Hz 采样余量。wrist 原图为 16:9 的
`424x240`，写入 pickle 时会中央裁出 `320x240`，因此采集前必须确认腕部关键操作
区域仍完整可见。

### 按键与保存结果

脚本默认实时显示写入 pickle 前的 wrist/front RGB 拼接画面；推荐命令均添加
`--draw-part-poses`。`one_leg` 在 front 画面上绘制 `P0 tabletop` 和
`P4 movable_leg`；`round_table` 绘制 `P0 top`、`P1 leg` 和 `P2 base`；`lamp`
绘制 `P0 base`、`P1 bulb` 和 `P2 hood` 的三维坐标轴。坐标轴使用与 front camera
setup 相同的 `camera_to_april` 求逆、
Rodrigues 和 `cv2.drawFrameAxes` 投影流程。绿色 `FOUND` 表示当前帧成功检测，黄色
`STALE` 表示暂时使用上一次检测位姿及其 age。配置初始化的 `P1/P2/P3` 和固定障碍物
`P5` 只存在于 `one_leg`，不会伪装成实时检测结果。

按键：`b` 开始、`e` 结束、`s` 保存为成功、`f` 保存为失败、`d` 丢弃、`r` 关节
复位、`p` 实时开关 part-pose 绘制、`q` 退出。OpenCV 预览窗口获得焦点时这些按键
同样有效。建议 `one_leg` 的 tabletop 与可动腿、`round_table` 和 `lamp` 各自三个
零件尽量全部有效；即使缺失也允许开始录制和保存，但会在质量日志中标为问题。
短暂的 AprilTag 遮挡会保留
最后一次位姿，同时用 `parts_founds`、`parts_pose_valid` 和
`parts_pose_age_ms` 标记是否为当前帧检测以及位姿新鲜度。

预览和坐标轴只用于屏幕显示，不会写入 pickle RGB 或保存的 MP4。没有图形桌面或
通过普通 SSH 启动时，添加 `--no-camera-preview`；该参数关闭窗口与在线预览 FSM，
不影响相机采集、`parts_poses` 计算或按 `e` 后的离线标注。

三个任务命令中的 `--real-skill-annotation` 会调用同级仓库
`robust-rearrangement-custom/src/eval/real_skill_annotation_util.py` 的几何标注接口。
dashboard 的 front/wrist RGB 画面会分别标出可投影的紫色目标点；白色的
`FSM: skill / skill_state` 与两路目标像素坐标显示在 wrist 画面的
`state=...` 下一行及其下方，不加底色。无法投影的点显示为 `--`；未启用、
还没有可用位姿或标注异常会显示 `FSM: OFF/WAITING/ERROR`，不会静默留白。
这些图形只画在预览副本上，不会污染保存的原始 RGB。预览标注不进入 pickle；按 `e` 完成时间匹配
后，脚本新建 `mode=offline` session，严格按最终 observation 顺序重算全部标注。
RR 真机 annotator 支持 `one_leg`、`round_table` 和 `lamp`；实时预览与最终 pickle
分别运行独立的 FSM session。

启用 `--real-skill-annotation` 后，按 `b` 开始时 dashboard 会重置并继续运行一套
独立的实时 FSM，持续显示当前 `skill/skill_state` 和 guidance point，方便确认遥操作阶段。它只消费当前预览帧，既不
写入 raw buffer，也不参与保存；按 `e` 后仍以对齐到 action 时间线的观测重新执行离线
标注，离线结果是 pickle 中唯一的 ground truth。

离线标注成功后，每个 observation 会保存 `skill`、`skill_state`、`assembly_step`、
`guidance_point`、`guidance_pose`、`guidance_point_2d`、`grasp_annotation_2d` 和
`real_annotation_debug`；pickle 根目录写入 `annotation_source=scripted`，并在
metadata 记录实现为 `real_skill_annotation_util`、`mode=offline`。最终标注或几何验证
失败时，或对齐、PromptDA、最终契约检查失败时，仍可按 `s`/`f` 保存；pickle 会
显式标为 `incomplete`，不会把预览结果或 VLM prediction 冒充 ground truth。
PromptDA 模型初始化失败时也会继续采原始 RGB-D，并在质量日志中记录初始化异常。
特别是未加 `--real-skill-annotation` 时，即使对齐和 PromptDA 完成，也只会得到
`unannotated` pickle。`round_table`/`lamp` 的真机推断中，障碍物使用 setup 时固定的
标定位姿，来源记录在 `metadata.real_skill_annotation.obstacle_pose_source`。
`metadata.real_skill_annotation.complete` 表示标注运行完成，`task_fsm_complete`
单独表示全部装配步骤达到几何完成条件；两者不能混为一谈。

原始 episode 保存到：

```text
$DATA_DIR_RAW/raw/osc/real/<task-name>/teleop/low/<output-suffix>/{success|failure}/
```

新 schema 每个 pickle 包含同为 `N` 个的 target-time observation、8 维 delta action
和 reward；旧 schema 仍可能是 `N+1` observation / `N` action。
每个 pickle 同目录另有同名 `.txt`，列出保存状态、每项错误、原始流数量和时序质量报告。
完整数据仍是 `deoxys_furniturebench_raw_v6_offline_buffered`；有任何质量错误时
schema 加 `_incomplete` 后缀并写入 `save_quality`。不完整文件单独放在
`<output-suffix>/incomplete/{success|failure}/`，不会混入标准的
`<output-suffix>/{success|failure}/`；使用递归 `--input-dir` 时仍须主动排除或重新处理。
若连时序对齐都失败，pickle 仍保存 SpaceMouse/发送命令和原始相机、机器人、夹爪流，
但 `observations/actions` 可能为空，不能直接用于训练。
录制中若 `robot_interface.control()` 报错，脚本会停止继续发送动作并结束当前采集；
按 `s`/`f` 仍可保存为不完整 pickle，然后按 `q` 退出并排查机械臂故障。
无法启动相机/机器人而尚未按 `b`、磁盘不可写或空间不足时，不能保证生成文件。
`color_image1`/`depth_image1` 是 wrist，`color_image2`/`depth_image2` 是 front。
RGB 为 `240x320 uint8`，对齐到 RGB 的 depth 为正米制 `240x320 float16`。
`parts_poses` 使用 FurnitureBench AprilTag 坐标系：`one_leg` 是 5 个零件加障碍物，
共 42 个数值；`round_table` 按 top、leg、base 顺序，`lamp` 按 base、bulb、hood
顺序保存 3 个 7 维 pose，共 21 个数值。action 格式为
`[dx, dy, dz, dqx, dqy, dqz, dqw, gripper]`：平移单位为米，
四元数顺序为 `xyzw`，旋转是末端局部坐标系右乘 delta。

当前默认配置为 front RGB-D `1280x720@30`（USB 3.x），wrist RGB
`424x240@30`、depth `480x270@30`（USB 2.1）。图像处理没有
`1280x720 -> 640x480 -> 320x240` 这样的
两级缩放：

- front AprilTag 始终直接使用原始 `1280x720` RGB。
- front 写入 pickle 前先从原图中央裁出 `960x720`，即左右各去掉 160 像素，再用
  `INTER_AREA` 等比例缩小为 `320x240`；不会把 16:9 拉伸成 4:3。
- front depth 先由 RealSense 对齐到 `1280x720` RGB 视角，再执行完全相同的裁剪，
  最后用 `INTER_NEAREST` 缩小为 `320x240`，避免生成不存在的深度插值值。
- wrist RGB 原图 `424x240` 在中间裁出 `320x240`，不缩放也不拉伸；对齐到
  RGB 后的 wrist depth 使用相同裁剪窗口。

`camera_info` 会同时保存每台相机的原始 color/depth profile、原始内参、裁剪窗口、
缩放比例以及变换后的 `320x240` 内参。每个 observation 还保存本次启动重新估计的
`camera_to_april`。因此移动相机后不需要改变 pickle schema，但必须重启脚本获得新
外参。在线 AprilTag 使用原始内参；后续从 pickle 图像重建点云或做离线几何计算时，
应使用 `record_intrinsics`。

pickle 和左右相机拼接 MP4 会由后台线程先写临时文件，再原子重命名。退出程序前应
等待保存完成，不要在终端刚显示保存按键后立即关机。每次 pickle 成功写盘后，终端
会输出当前 task/randomness 目录下的原始文件总数，例如
`files: success=12 fail=3`；离线生成的 annotated pickle 和临时文件不会计入。

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

推荐 ViT-L、`max-size 448` 和 `320x240` 保存分辨率；双相机历史实测约
`16.79 observation/s`，但正式数采不再依赖它满足实时控制 deadline。

### 方案一：episode 结束后离线增强并保存

SpaceMouse record 的 PromptDA 参数只在按 `e`、完成 target-time 筛选重排后运行，
不会占用录制控制循环。它会处理最终选中的每一帧，把增强后的 `320x240 float16`
米制 depth 保存到 `depth_image1/2`；原始 RealSense depth 原样保存在
`depth_image1/2_realsense`。RGB 始终保持相机原始像素，colored guidance point 只在
后续 pickle-to-LMDB 转换中渲染。

```shell
source ~/.bashrc
conda activate deoxys
cd /home/hz/code/YueHu_deoxys
export HF_ENDPOINT=https://hf-mirror.com

python -m deoxys.examples.run_deoxys_with_space_mouse_V3_record \
  --interface-cfg deoxys/config/charmander.yml \
  --latency-profile /home/hz/code/robust-rearrangement-custom/src/real/latency_profile.measured_20260908.json \
  --controller-type OSC_POSE \
  --annotation-source scripted \
  --output-suffix one-leg-umi-scripted-rgbd-202609 \
  --vendor-id 9583 \
  --spacemouse-connection wired \
  --record-image-width 320 \
  --record-image-height 240 \
  --record-fps 10 \
  --draw-part-poses \
  --real-skill-annotation \
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
`示例_promptda_vitl_comparison.mp4`；新 pickle 的字段、分辨率和单位与数采结束后离线
增强方案一致。

## RR 240×320 full-frame 真机 Eval（ModelScope 0912）

以下命令在 FrankaControl 图形桌面终端运行。策略以 front RealSense source time 为
`T_obs`，action chunk 的目标时间固定为 `T_obs + k * action_period`。推理在后台 worker
运行，主线程独占 `FrankaInterface`，并按各自 deadline 独立调度 arm queue 和 gripper
queue。通过公共 admission cutoff 且安全检查通过的 action 会原子地进入两条 queue；
单个通道事件过期只丢弃该事件，不再清空其后的 action。已进入 immutable timeline 的
timestep 不允许被后续 query 覆盖，后续 query 只能向 timeline 尾部追加。本轮使用
campaign `rr_real_sim_modelscope_0912`：包含 `real40`、`real40_sim400` 和
`real10_sim400` 三组，每组都部署了 3000 和 5000 epoch。`real40` 与
`real40_sim400` 使用最新 `real40-reannotation-v13-timestamp-20260912` 时序数据；
`real10_sim400` 暂时使用已完成的 `timeline10hz` 版本，不得用尚未训完的
ws2 `actor_chkpt_last.pt` 冒充 3000/5000。所有 checkpoint 都是 240×320 full-frame
RGB-D，配置中的
`data.image_spatial_transform=none`；eval 不再 crop 或 resize。

`evaluate_policy` 会直接读取 checkpoint 配置决定标注方式。本轮 checkpoint 的
`annotate_guidance_point_colored=true`，因此每次 query 都会先实时运行 real annotation
util，再把 colored guidance point 画入 front RGB 后送给策略；wrist RGB 按训练契约保持
不画点。`--show-input-dashboard` 会在按 `b` 后第一次成功 query 时打开一个 OpenCV 页面，
随后逐 query 更新。页面同时显示经过 checkpoint 图像变换后、实际送入策略的 240×320
front/wrist RGB、两路 PromptDA 深度、本体状态、skill/skill_state、guidance point 及两相机投影、零件检测有效性、延迟，
以及预测 action chunk 中张开/闭合夹爪的数量。wrist 的 guidance UV 只作为诊断文字显示，
不会改变 wrist 策略输入。操作 `r/b/e/q` 时仍需让终端获得键盘焦点。
页面和终端会分别报告整段 chunk 的闭合预测数、下一次 query 前近期执行窗口中的闭合
预测数、两条 queue 的长度、immutable coverage、保留的 occupied timestep 和
warm-start 映射数。已经进入 timeline 的动作不会被下一段 receding-horizon query
覆盖；gripper 每个 timestep 都会消费，但只有 sign 改变时才发送物理命令。

在 FrankaControl 上必须保持启动顺序为“两路 RealSense 管线 → CUDA policy →
PromptDA/Deoxys”。这不是性能优化：该机器上如果先初始化 CUDA 再启动
librealsense，会在退出阶段触发 glibc heap corruption；`evaluate_policy` 已固定该顺序，
不要把 checkpoint 加载移回相机启动之前。dry-run warmup 如果报告
`PromptDA ready=True` 但 `robot states=0, gripper states=0`，说明视觉链路正常而 NUC
状态发布器未就绪。先在 Franka Desk 确认 FCI，再由现场操作员按既有流程在 NUC 启动
`/home/mingyu/code/deoxys_control/deoxys/run2.sh` 和 `run3.sh`；不要用放宽超时或直接
加 `--execute` 绕过检查。

先准备环境变量和公共安全参数。下面的公共参数默认不含 `--execute`，先完成 dry-run；
确认相机时间域、PromptDA、checkpoint、240×320 输入和状态插值正常后，再运行单独列出的
真机执行命令。workspace 数值是当前 one-leg 配置，工作台或机器人基座位置改变后必须
重新测量。

程序完成 warmup 后停在 `IDLE`，不会自动开始：`r` 使用与正式数采相同的 joint
reset 目标，`b` 重置在线标注状态并 begin，`e` 丢弃未执行 action、保持当前位姿、
重置标注状态并回到 `IDLE`，`q` 退出。reset 只允许在 `IDLE` 执行。首次真机建议
`--execution-frequency 5`，即 200 ms/action；这会整体拉伸 UMI target-time 时间轴，
并非简单地跳过 10 Hz action。

### Workspace 与末端高度安全边界（当前 one-leg）

2026-09-01 现场测量确认 tabletop 必须允许比原边界低约 3 cm，因此当前 one-leg
参数先从 `workspace z_min=0.03/min_ee_z=0.04` 调整为 `0.00/0.01`；随后根据
2026-09-01 pick 实测中被拒绝的 `z=0.0065–0.0097 m`，将 `min_ee_z` 再向下
放宽 0.5 cm 到 `0.005 m`。该测量只适用于当前工作台和基座位置。

- `--workspace-min 0.30 -0.35 0.00`：机器人基座坐标系中的 XYZ 下界。

- `--workspace-max 0.75 0.35 0.60`：机器人基座坐标系中的 XYZ 上界。

- `--min-ee-z 0.005`：末端执行器 Z 的独立硬下界。

- `--max-translation-step-m 0.085`：单条命令的平移硬上限，即 8.5 cm。2026-09-09
  最新 `real40_sim400-5000` run 在成功执行 127 步后，于 `place/leg-top-place` 出现
  19 个 translation reject，范围为 `0.0567–0.0807 m`、中位数为 `0.0792 m`；目标主要
  沿 `+x≈5.0 cm`、`+z≈6.3 cm` 移动，仍在 workspace 内。本值比实测最大值保留约 5%
  余量。

- `--max-translation-speed-m-s 0.425`：平移速度硬上限。当前 5 Hz 下对应
  `0.425 m/s × 0.2 s = 0.085 m/action`，与 translation step 上限一致。不能只提高
  translation step 而保留旧的 `0.25 m/s`，否则有效上限仍是 `0.05 m/action`。

- `--max-rotation-step-rad 0.40`：单条命令的旋转硬上限，约 22.9°。2026-09-09 的
  `real40_sim400-5000` run 中 19 个 rotation reject 位于 `0.3390–0.3791 rad`，本值比
  实测最大值保留约 5% 余量。

- `--max-rotation-speed-rad-s 2.0`：旋转速度硬上限。当前 5 Hz 下对应
  `2.0 rad/s × 0.2 s = 0.40 rad/action`，与 rotation step 上限一致。不能只提高
  rotation step 而保留旧的 `1.5 rad/s`，否则有效上限仍是 `0.30 rad/action`。

workspace 参数和 `--execution-frequency 5` 已同时写入程序默认值及下面的正式运行数组。
上述 translation/rotation 值是本轮 one-leg 5 Hz eval 的显式覆盖值，没有修改程序全局
默认值；更换任务或控制频率后必须重新检查动作分布。
已有终端中的旧 `RR_EVAL_ARGS` 不会因 README 更新而自动改变；每次开始一组实验都要
重新执行完整的准备代码块。日志第一行中的 `workspace_min`、`workspace_max`、
`min_ee_z`、`max_translation_step_m`、`max_translation_speed_m_s`、
`max_rotation_step_rad`、`max_rotation_speed_rad_s` 和 `execution_frequency_hz` 是本次
进程实际采用的权威值。

### Eval 前准备：一次标定完整 timing profile

更换控制 PC、NUC、网络路径、任一 RealSense、控制器配置或 gripper 后，在启动
eval 前重新运行同一个标定入口。默认 `--component all` 一次采集六项 latency，另存
RealSense、robot/gripper protobuf uptime 到主机时钟的 affine mapping、offset、drift、
residual，以及前腕相机 timestamp skew。

相机项要求两台 RealSense 均成功启用 `global_time_enabled`，且 frame timestamp domain
为 `global_time` 或 `system_time`；否则脚本 fail closed，不生成错误 profile。robot 和
gripper protobuf 只提供设备 uptime、没有共享 epoch，因此绝对 observation latency
采用与 UMI 相同的半 RTT 近似，uptime 拟合只用于检查 drift/jitter，不能冒充单程延迟。

action 标定会让末端沿选定轴往返 10 mm，并让空夹爪开合；运行前必须清空工作区和
夹爪，确认当前末端位于 `--workspace-min/max` 内，并准备好急停。不要在测量期间触碰
机器人。

arm 标定与已验证的 `src.real.evaluate_policy` 使用同一条发送路径：OSC_POSE absolute
pose（`is_delta=false`）、`LINEAR_POSE`、`time_fraction=2.0`，并以 5 Hz 重发目标
absolute pose。不要把它改回 `osc-pose-controller.yml` 原始的 delta/0.3 配置；短促 delta
脉冲与实际 eval 的控制时序不同，也可能让 NUC 的 libfranka 控制循环异常退出。
等待运动 onset 时也会像 policy eval 的 idle hold 一样持续重发同一个 target；stdout 会
输出实际模块路径、resolved controller 配置、每次 target send、robot state age 和相对
baseline 的位移。若仍失败，请保留 FrankaControl 与 NUC 两端从启动到退出的完整输出。
gripper 的单条 ZMQ 命令可能因 NUC subscriber 的 `try_lock` 竞态被吞掉；标定默认会按
state onset 确认并最多重发 3 次，每次尝试单独记录发送时间，不把失败等待计入 latency。

```shell
source ~/.bashrc
conda activate rr-real
export RR_ROOT=/home/hz/code/robust-rearrangement-custom
export DEOXYS_ROOT=/home/hz/code/YueHu_deoxys
CAL_TAG=$(date +%Y%m%dT%H%M%S)
export LATENCY_PROFILE_DIR="$RR_ROOT/logs/latency"
LATENCY_PROFILE_OUTPUT="$LATENCY_PROFILE_DIR/latency_profile-$CAL_TAG.json"
cd "$DEOXYS_ROOT/deoxys"

python -m deoxys.examples.calibrate_action_latency \
  --interface-cfg "$DEOXYS_ROOT/deoxys/config/charmander.yml" \
  --controller-cfg "$DEOXYS_ROOT/deoxys/config/osc-pose-controller.yml" \
  --component all \
  --trials 6 \
  --arm-axis z \
  --arm-step-m 0.010 \
  --arm-threshold-m 0.0005 \
  --arm-command-frequency-hz 5 \
  --controller-time-fraction 2.0 \
  --gripper-command-attempts 3 \
  --workspace-min 0.30 -0.35 0.00 \
  --workspace-max 0.75 0.35 0.60 \
  --output "$RR_ROOT/logs/latency/full-latency-calibration-$CAL_TAG.json" \
  --base-latency-profile "$RR_ROOT/src/real/latency_profile.estimated_10ms.json" \
  --profile-output "$LATENCY_PROFILE_OUTPUT" \
  --execute
```

同一个命令可连续运行多次。若目标文件已经存在，标定程序不会覆盖，而会让 full JSON
与 profile 成对追加 `-run02`、`-run03` 等后缀；stdout 会打印每次实际写入的完整路径。

2026-09-08 在 FrankaControl 上连续三次标定的结果如下。每次结果写作
`median（p95）`，三次合并值使用全部原始 samples；单位均为 ms。

| 参数与最终推荐值 | 起点事件与时间戳/时钟域 | 终点事件与时间戳/时钟域 | 实际计算 | 第 1 次 | 第 2 次 | 第 3 次 | 三次合并 median（p95） |
|---|---|---|---|---:|---:|---:|---:|
| `front_observation_ms` = **36.810** | 前 RealSense 彩色帧曝光；`color_frame.get_timestamp()`，由 `global_time_enabled` 映射到 FrankaControl system/global time | `read()` 完成取帧、depth-to-color alignment、depth 转换和 RGB 拷贝；FrankaControl `time.time_ns()` | `t_ready_FC - t_exposure_FC` | 37.322（44.227） | 36.339（39.654） | 39.711（43.582） | 36.810（43.623） |
| `wrist_observation_ms` = **40.304** | 腕 RealSense 彩色帧曝光；同上，映射到 FrankaControl system/global time | 腕相机 `read()` 完成全部处理；FrankaControl `time.time_ns()` | `t_ready_FC - t_exposure_FC` | 33.148（46.263） | 41.044（74.667） | 32.988（45.760） | 40.304（74.490） |
| `robot_observation_ms` = **0.067** | 概念起点是 NUC 发出 robot-state ZMQ；protobuf 只有 NUC/机器人 device uptime，没有与 FrankaControl 对齐的 wall time | FrankaControl 收到并解析 robot state 后的 `time.time_ns()` | 不跨主机直接相减；用 FrankaControl 发起的 ICMP RTT / 2 近似 | 0.067（0.089） | 0.059（0.075） | 0.068（0.090） | 0.0665（0.089） |
| `gripper_observation_ms` = **0.067** | 概念起点是 NUC 发出 gripper-state ZMQ；同样没有共享 wall time | FrankaControl 收到并解析 gripper state 后的 `time.time_ns()` | 同样使用 NUC ICMP RTT / 2 近似 | 0.067（0.089） | 0.059（0.075） | 0.068（0.090） | 0.0665（0.089） |
| `robot_action_ms` = **120.000** | arm ZMQ publish 返回后；FrankaControl `time.time_ns()` | FrankaControl 收到的 robot state 中首次连续两帧 EE 位移 ≥0.5 mm，取第一帧的本机 receive `time.time_ns()` | `(t_onset_receive_FC - t_publish_FC) - robot_observation_ms` | 122.656（144.627） | 117.415（127.299） | 119.784（142.916） | 119.047（148.154） |
| `gripper_action_ms` = **642.000** | gripper ZMQ publish 返回后；FrankaControl `time.time_ns()` | FrankaControl 收到的 gripper state 中首次连续两帧 width 变化 ≥2 mm，取第一帧的本机 receive `time.time_ns()` | `(t_onset_receive_FC - t_publish_FC) - gripper_observation_ms` | 668.781（1244.734） | 621.534（982.369） | 720.153（1109.517） | 641.725（1237.695） |
| `action_stale_guard_ms` = **10.000** | 非物理 latency，无跨时钟端点 | stale-prefix admission 的额外调度余量 | 从 base profile 继承，本次不标定 | 10 | 10 | 10 | 10 |

NUC 与 FrankaControl 的 wall clock 不保证对齐，代码也不使用二者的 wall-time 差值。
robot/gripper observation 的 RTT/2 不要求时钟同步，但只是网络单程延迟近似，不能单独
识别 NUC publisher 的排队、序列化和线程调度时间。action latency 的发送和接收端点都由
FrankaControl 打时间戳，因此不受两台主机 clock offset 影响。完整 JSON
保留原始样本、median、p95、clock fit 和方法限制；eval profile 写入六项推荐 latency，
只从 base profile 继承 `action_stale_guard_ms` 等非物理延迟配置，并在
`calibrated_fields` / `inherited_fields` 中明确记录。默认用 median 做时间对齐，p95 用于
判断抖动和设置 guard；若 p95 与 median 相差很大，应先排查 USB、网络和状态 publisher。

当前 RR evaluator 直接以共享时钟上的相机曝光 timestamp 为 `t_obs`，因此不会再从
`t_obs` 减去 `front/wrist_observation_ms`；这两个字段记录 camera pipeline 的可用性延迟，
用于审计 observation age、action horizon 和 drop 数量。robot/gripper state 没有共享
epoch，`robot/gripper_observation_ms` 才会作为 receive-time correction 实际参与插值。

arm/gripper 的 open/close、正/反方向原始 trial 均保留。夹爪还会单独输出 open/close
统计；若两者相差超过一个控制周期，当前单一 `gripper_action_ms` 只是折中值，应先扩展
runtime schema 后再按方向调度。新标定完成后可用目录模式检查当天最新的 profile；
目录模式会解析 `measured_at`，仅在本地当天的文件中选择时间最新者，若当天没有有效
profile 则直接报错，不会退回昨天的参数：

```shell
cd "$RR_ROOT"
python - <<'PY'
from pathlib import Path
from src.real.time_alignment import LatencyProfile

profile = LatencyProfile.resolve_path(Path("logs/latency"))
print(profile)
print(LatencyProfile.load(profile))
PY
```

### 固定 checkpoint 位置

6 个固定 epoch checkpoint 都在 RR 仓库下，不在 15 TB 盘的旧 campaign 目录：

```text
/home/hz/code/robust-rearrangement-custom/checkpoints/rr_real_sim_modelscope_0912/real40/rr_modelscope0912_real40_b256_ws2_seed2026091213_modelscopev13_0916/rr_modelscope0912_real40_b256_ws2_seed2026091213_modelscopev13_0916/actor_chkpt_latest_3000.pt
/home/hz/code/robust-rearrangement-custom/checkpoints/rr_real_sim_modelscope_0912/real40/rr_modelscope0912_real40_b256_ws2_seed2026091213_modelscopev13_0916/rr_modelscope0912_real40_b256_ws2_seed2026091213_modelscopev13_0916/actor_chkpt_latest_5000.pt
/home/hz/code/robust-rearrangement-custom/checkpoints/rr_real_sim_modelscope_0912/real40_sim400/rr_modelscope0912_real40_sim400_b256_seed2026091211/rr_modelscope0912_real40_sim400_b256_seed2026091211/actor_chkpt_latest_3000.pt
/home/hz/code/robust-rearrangement-custom/checkpoints/rr_real_sim_modelscope_0912/real40_sim400/rr_modelscope0912_real40_sim400_b256_seed2026091211/rr_modelscope0912_real40_sim400_b256_seed2026091211/actor_chkpt_latest_5000.pt
/home/hz/code/robust-rearrangement-custom/checkpoints/rr_real_sim_modelscope_0912/real10_sim400/rr_modelscope0912_real10_sim400_b256_ws1_seed2026091212_timeline10hz/rr_modelscope0912_real10_sim400_b256_ws1_seed2026091212_timeline10hz_2026-09-14_16-08-18.727013/actor_chkpt_latest_3000.pt
/home/hz/code/robust-rearrangement-custom/checkpoints/rr_real_sim_modelscope_0912/real10_sim400/rr_modelscope0912_real10_sim400_b256_ws1_seed2026091212_timeline10hz/rr_modelscope0912_real10_sim400_b256_ws1_seed2026091212_timeline10hz_2026-09-14_16-08-18.727013/actor_chkpt_latest_5000.pt
```

### 选择条件和 epoch

每次新开终端都复制下面整个代码块。只需要修改开头的 `RR_RUN` 和
`RR_EPOCH`：`RR_RUN` 可取 `real40`、`real40_sim400`、`real10_sim400`，
`RR_EPOCH` 可取 `3000` 或 `5000`。例如 `RR_RUN=real40_sim400`、
`RR_EPOCH=3000` 就会选中 real40+sim400 的 3000 epoch checkpoint。

```shell
source ~/.bashrc
conda activate rr-real

export RR_ROOT=/home/hz/code/robust-rearrangement-custom
export DEOXYS_ROOT=/home/hz/code/YueHu_deoxys
export RR_PYTHON=/home/hz/miniconda3/envs/rr-real/bin/python
export CKPT_ROOT="$RR_ROOT/checkpoints/rr_real_sim_modelscope_0912"
export LATENCY_PROFILE="$RR_ROOT/src/real/latency_profile.measured_20260908.json"

# 只修改这两个变量。
RR_RUN=real40_sim400
RR_EPOCH=3000

case "$RR_RUN" in
  real40)
    RR_RUN_DIR="real40/rr_modelscope0912_real40_b256_ws2_seed2026091213_modelscopev13_0916/rr_modelscope0912_real40_b256_ws2_seed2026091213_modelscopev13_0916"
    ;;
  real40_sim400)
    RR_RUN_DIR="real40_sim400/rr_modelscope0912_real40_sim400_b256_seed2026091211/rr_modelscope0912_real40_sim400_b256_seed2026091211"
    ;;
  real10_sim400)
    RR_RUN_DIR="real10_sim400/rr_modelscope0912_real10_sim400_b256_ws1_seed2026091212_timeline10hz/rr_modelscope0912_real10_sim400_b256_ws1_seed2026091212_timeline10hz_2026-09-14_16-08-18.727013"
    ;;
  *)
    printf '错误：RR_RUN 必须是 real40、real40_sim400 或 real10_sim400，当前为 %s\n' "$RR_RUN" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

case "$RR_EPOCH" in
  3000|5000) ;;
  *)
    printf '错误：RR_EPOCH 必须是 3000 或 5000，当前为 %s\n' "$RR_EPOCH" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

export RR_CHECKPOINT="$CKPT_ROOT/$RR_RUN_DIR/actor_chkpt_latest_${RR_EPOCH}.pt"
export RR_LOG_DIR="$RR_ROOT/logs/real_policy_eval"

cd "$RR_ROOT"
test -x "$RR_PYTHON"
test -f "$DEOXYS_ROOT/deoxys/config/charmander.yml"
test -f "$LATENCY_PROFILE"
test -f "$RR_CHECKPOINT"
mkdir -p "$RR_LOG_DIR"
printf '条件：%s\nepoch：%s\ncheckpoint：%s\n' "$RR_RUN" "$RR_EPOCH" "$RR_CHECKPOINT"

unset RR_EVAL_ARGS
RR_EVAL_ARGS=(
  --interface-cfg "$DEOXYS_ROOT/deoxys/config/charmander.yml"
  --latency-profile "$LATENCY_PROFILE"
  --execution-frequency 5
  --query-interval-steps 2
  --max-action-lateness-ms 10
  --max-wall-time-s 270
  --workspace-min 0.30 -0.35 0.00
  --workspace-max 0.75 0.35 0.60
  --min-ee-z 0.0
  --max-translation-step-m 0.085
  --max-translation-speed-m-s 0.425
  --max-rotation-step-rad 0.40
  --max-rotation-speed-rad-s 2.0
  --prompt-depth-model vitl
  --prompt-depth-device cuda
  --show-input-dashboard
)
```

`--query-interval-steps` 完全由 CLI 控制，程序默认值没有硬编码为 `2`。本轮 5 Hz
真机命令推荐使用 `2`；需要做调度对比时，可以在重新创建 `RR_EVAL_ARGS` 时改成其他
正整数。

`--show-input-dashboard` 继续逐 query 显示 RGB-D、标注、本体状态和 action 页面。
`--save-input-video` 从按下 `b` 后的第一条成功 query 开始记录，到 `e` 为止；每帧是
front RGB、wrist RGB、front PromptDA depth、wrist PromptDA depth 四宫格。MP4 保存在
对应 JSONL 旁边，文件名后缀为 `-rollout-XXX-rgbd-grid.mp4`。
每条成功 query 的 JSONL 记录也会保存完整 `parts_poses`、检测/valid 状态和 annotation
debug，便于在后续 run 中直接恢复 pick 时的 EE-to-leg 夹持位姿并检查手动物体移动。

若任一 `test` 报错，不要开始真机执行。程序启动后还要检查打印的 checkpoint 路径包含
`rr_real_sim_modelscope_0912`，并确认 checkpoint 配置显示
`observation_type=rgbd`、`image_spatial_transform=none`。不要沿用旧终端里的
`CKPT_ROOT`、`RR_CHECKPOINT` 或 `RR_EVAL_ARGS`。

下载或复制 checkpoint 后可一次核对全部 6 个 SHA-256：

```shell
sha256sum \
  "$CKPT_ROOT/real40/rr_modelscope0912_real40_b256_ws2_seed2026091213_modelscopev13_0916/rr_modelscope0912_real40_b256_ws2_seed2026091213_modelscopev13_0916/actor_chkpt_latest_3000.pt" \
  "$CKPT_ROOT/real40/rr_modelscope0912_real40_b256_ws2_seed2026091213_modelscopev13_0916/rr_modelscope0912_real40_b256_ws2_seed2026091213_modelscopev13_0916/actor_chkpt_latest_5000.pt" \
  "$CKPT_ROOT/real40_sim400/rr_modelscope0912_real40_sim400_b256_seed2026091211/rr_modelscope0912_real40_sim400_b256_seed2026091211/actor_chkpt_latest_3000.pt" \
  "$CKPT_ROOT/real40_sim400/rr_modelscope0912_real40_sim400_b256_seed2026091211/rr_modelscope0912_real40_sim400_b256_seed2026091211/actor_chkpt_latest_5000.pt" \
  "$CKPT_ROOT/real10_sim400/rr_modelscope0912_real10_sim400_b256_ws1_seed2026091212_timeline10hz/rr_modelscope0912_real10_sim400_b256_ws1_seed2026091212_timeline10hz_2026-09-14_16-08-18.727013/actor_chkpt_latest_3000.pt" \
  "$CKPT_ROOT/real10_sim400/rr_modelscope0912_real10_sim400_b256_ws1_seed2026091212_timeline10hz/rr_modelscope0912_real10_sim400_b256_ws1_seed2026091212_timeline10hz_2026-09-14_16-08-18.727013/actor_chkpt_latest_5000.pt"
```

期望依次为：

```text
d7917a59916ba6728c08e8fc86441ca8a88ed1968e5fafa7076c4515676babd2  real40/3000
b7ac91616006421a49aa7d04c6c62c96d2d4c9afe13d02ecfe911bf37ce8f450  real40/5000
25bf7b6f321027a7f772978c6df49d9aab3b3028319466f6f4050cd5971d6ab8  real40_sim400/3000
d26a0645ba21ea3cb80f0f0d861484e10d655080001f593c649a0d61a4c8f674  real40_sim400/5000
dcb3a50c2c17bc9c32169ce96fbbe2f43aec97fc64c57ddda3707c8030fbd028  real10_sim400/3000
814183afc4365f4dfd119e96ed9809125db5157b115fb9aa418a98f251929923  real10_sim400/5000
```

### 先 dry-run

准备代码块默认没有 `--execute`，复制运行以下命令不会向机器人发送策略动作：

```shell
"$RR_PYTHON" -m src.real.evaluate_policy \
  --checkpoint "$RR_CHECKPOINT" \
  --log-path "$RR_LOG_DIR/${RR_RUN}-${RR_EPOCH}-dryrun-$(date +%Y%m%dT%H%M%S).jsonl" \
  "${RR_EVAL_ARGS[@]}"
```

### 确认后执行真机 eval

确认现场安全、FCI、NUC 状态发布器、相机、PromptDA、240×320 dashboard 和状态插值均正常后，
再复制下面的命令。只有这里显式添加 `--execute`：

```shell
"$RR_PYTHON" -m src.real.evaluate_policy \
  --checkpoint "$RR_CHECKPOINT" \
  --log-path "$RR_LOG_DIR/${RR_RUN}-${RR_EPOCH}-execute-$(date +%Y%m%dT%H%M%S).jsonl" \
  "${RR_EVAL_ARGS[@]}" \
  --execute
```

`latency_profile.estimated_10ms.json` 明确标记为 estimated。日志中的
`deadline_residual_ms` 用于检查调度线程是否按 deadline 醒来；更换网络路径、控制频率
或 gripper 配置后应重新估计 action latency，并把 profile 的 `latency_source`、
`basis`、`measured_at` 一起更新。不得通过增大 lateness 容忍度来执行已经过期的动作。
