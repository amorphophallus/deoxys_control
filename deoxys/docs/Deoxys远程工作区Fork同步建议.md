# Deoxys 远程工作区 Fork 与同步建议

更新日期：2026-08-05

## 1. 当前仓库关系

FrankaControl 实际运行目录：

```text
/home/hz/code/YueHu_deoxys/deoxys
```

这个目录本身及其父目录不是 Git 仓库，约 4.8 GB。它的核心代码与下面的 Git 工作树基本一致：

```text
/home/hz/code/MingyuLiu___deoxys/deoxys_control
origin: git@github.com:HaoyiZhu/deoxys_control.git
branch: main
commit: ae625b8c54f1a67265fb64ec552e85ffdd3ab16c
```

NUC：

```text
/home/mingyu/code/deoxys_control
upstream: UT-Austin-RPL/deoxys_control
branch: main
commit: 97396fd...
```

建议以干净的新 fork/clone 为基线，从现有工作区逐项挑选代码，不要把 4.8 GB 目录整体上传。

## 2. 建议同步到 fork 的内容

### 第一优先级：当前硬件必需

| 文件 | 原因 | 建议 |
|---|---|---|
| `deoxys/utils/io_devices/spacemouse.py` | 支持 product id 50746、新旧 HID 报告和组合按键 | 同步并单独提交 |
| `installation/create_spacemouse.sh` | 增加 50746 的 udev 规则 | 同步并单独提交 |
| `examples/run_deoxys_with_space_mouse.py` | SpaceMouse 控制和优雅退出修订 | 同步；与 V3 功能择一整理 |
| `examples/run_deoxys_with_space_mouse_V3.py` | 当前真机控制入口 | 同步 |
| `examples/run_deoxys_with_space_mouse_V3_record.py` | 新的控制+RGB-D+raw pickle 录制入口 | 同步，完成真机验证后提交 |
| `deoxys/franka_interface/franka_interface.py` | 夹爪 force 从 30 N 改为 15 N | 若这是硬件安全策略则同步，并在 commit 中明确 |

### 第二优先级：实验/回放工具

- `examples/osc_control_replay_robot_eval.py`
- `examples/osc_control_replay_robot_eval_ee_pose.py`
- `examples/osc_control_replay_robot_eval_ee_pose.py` 的必要配套文件
- `config/joint-impedance-min-jerk-controller.yml`
- 确实仍在使用的 calibration、replay、visualization 脚本

这些应逐个检查硬编码路径和数据依赖，按功能拆 commit，不要把整个 `examples/` 未跟踪集合一次加入。

### 配置和运维脚本

`config/charmander.yml` 包含实验室固定 IP/端口，建议同步为明确命名的 site config，例如：

```text
config/charmander-lab.yml
```

不要无说明地覆盖上游默认配置。NUC 的 `run1.sh`、`run2.sh`、`run3.sh` 有用，但名称不表达用途，建议整理为：

```text
scripts/run_franka_once.sh
scripts/run_franka_auto_restart.sh
scripts/run_gripper_auto_restart.sh
```

NUC 当前真正的工作区更改只有 `charmander.yml` 和这三个未跟踪脚本。

## 3. 不建议直接同步的内容

| 内容 | 处理意见 |
|---|---|
| `InstallPackage` 的 HTTPS→SSH clone URL | 不同步；SSH URL 降低通用性，除非团队明确要求 |
| `beta_scripts/osc_traj_following.py` | 不按现状同步；含 `/home/hz/...` 硬编码路径，先改为 CLI/config |
| `vqvla_scripts/` 整目录 | 不放入核心 deoxys fork；体量和职责不同，建议独立 repo/submodule |
| `zzdata1897` symlink | 不同步；指向另一个本地 RoboTwin 数据目录 |
| 大量 `deoxys/demo_data/test/...` tracked deletion | 不与代码功能提交混合；先恢复/另做数据清理策略 |
| `protobuf/`、`spdlog/`、`libfranka/`、`yaml-cpp/`、`zmqpp/` 的构建产物 | 只保留项目原有依赖管理，不上传本地 build/cache |
| `.vscode/`、`__pycache__/`、日志和本地校准输出 | 加入 `.gitignore`，不上传 |

## 4. 可清理或移出代码工作区的内容

未执行任何删除。下列是候选项，删除前仍应做一次路径和差异确认。

### 可以直接清理的明显临时项

- `aaaa.py`：0 字节
- `syncRecord.txt`：0 字节
- `logs/debug.log`：约 129 MB 的运行日志
- `__pycache__/`、`.pyc`、临时 cache
- `examples/osc_control_replay_robot_eval_ee_pose.py.bak_20260512_163322`：确认与正式文件差异后删备份
- `examples.zip`：确认内容已存在后删冗余压缩包

### 建议归档或移到独立工具仓库

- `bbbb.py`：硬编码路径的 MoviePy 一次性脚本
- `zzread.py`：HDF5/图像转换 scratch script
- `zzutils.py`：约 93 KB 的视觉/仿真数据工具，不属于 deoxys 核心
- `episode0.hdf5`、CSV/NPZ 轨迹、截图、回放结果
- `vqvla_scripts/insert/insert_our.ckpt`：约 416 MB checkpoint

### 必须移出 Git、但不要贸然删除的数据

- `data/`：约 1.9 GB
- 相机视频、raw pickle、HDF5 数据集
- calibration output、模型 checkpoint

这些应放到独立数据盘/对象存储，并在仓库中只保留 README、schema 和小型测试 fixture。

## 5. 推荐的同步步骤

1. 在 GitHub fork 官方仓库并做一个全新 clone。
2. 从当前工作树只复制第一优先级文件，逐项查看 diff。
3. 先提交 SpaceMouse 设备兼容性，再提交 V3 控制脚本，再提交录制脚本。
4. 将 lab 网络配置和 NUC 运维脚本作为单独的 `chore` commit。
5. 增补 `.gitignore`，确认没有数据、日志、checkpoint、SSH 密钥或机器路径。
6. 用 `git status --short`、`git diff --cached --stat` 和 staged diff 做最后检查。
7. 用户确认 commit message 后再 commit；push 前确认 fork remote 和目标分支。

推荐拆分的 commit messages：

```text
feat: add spacemouse compact device support

feat: add spacemouse osc pose control workflow

feat: add rgb-d raw episode recording

chore: add lab franka network and launch configuration
```

如果必须合成一个提交，建议 comment：

```text
feat: add spacemouse rgb-d teleoperation workflow

- support the compact spacemouse hid reports
- add osc pose control and joint reset handling
- record dual-camera actions and robot states as raw episodes
- add lab-specific franka launch configuration
```

不要添加 `Co-Authored-By`、`Signed-off-by` 或其他共同署名行。
