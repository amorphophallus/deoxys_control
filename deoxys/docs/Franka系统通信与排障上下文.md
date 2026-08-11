# Franka 系统通信、排障与遥操作录制上下文

更新日期：2026-08-05

这份文档用于下一次排障时直接提供上下文。它不保存 SSH 密码、私钥或其他凭据。

## 1. 机器和网络拓扑

| 节点 | 用途 | 控制网地址 | ZeroTier 地址 |
|---|---|---:|---:|
| FrankaControl | Python client、SpaceMouse、相机和数据录制 | `172.16.0.3/24` | `10.147.93.90` |
| NUC | Deoxys C++ server、实时控制进程 | `172.16.0.1/24` | `10.147.93.216` |
| Franka 机械臂 | FCI/机器人控制器 | `172.16.0.2` | 无 |

正常控制数据必须走 `172.16.0.0/24` 物理有线网络，不应走 Wi-Fi 或 ZeroTier：

```text
SpaceMouse / Python client
        FrankaControl 172.16.0.3
                 │ ZMQ/TCP
                 │ action -> 5543；state <- 5544
                 │ gripper action -> 5557；gripper state <- 5558
                 ▼
              NUC 172.16.0.1
                 │ libfranka / FCI，1 kHz 实时环
                 ▼
          Franka robot 172.16.0.2
```

ZeroTier 只用于 SSH 维护。当前 SSH 别名为：

- `ssh FrankaControl` → `hz@10.147.93.90`
- `ssh NUC` → `mingyu@10.147.93.216`

ZeroTier 曾出现节点显示 `ONLINE` 但双方 ping 不通，以及 NUC、FrankaControl 只能间歇到达的情况。这不等同于控制网故障。必要时可从 NUC 跳到 FrankaControl 的物理地址：

```bash
ssh -J NUC hz@172.16.0.3
```

## 2. Deoxys client/server 的核心逻辑

Client 启动命令原来是：

```bash
python examples/run_deoxys_with_space_mouse_V3.py \
  --interface-cfg config/charmander.yml \
  --controller-type OSC_POSE \
  --vendor-id 9583 \
  --product-id 50746
```

Client 的大致链路：

1. `SpaceMouse` 后台读取 HID 报告，转换为 6-DoF 位姿增量和夹爪命令。
2. `input2action()` 生成 OSC action。
3. `FrankaInterface.control()` 用 ZMQ 把 action 发到 NUC。
4. `FrankaInterface` 同时订阅 NUC 发布的机器人 state，放入本地 state buffer。
5. V3 的右键用于结束；`r` 键执行关节复位。

NUC 的 C++ server：

- `bin/franka-interface` 接收 arm action，并通过 libfranka/FCI 控制 `172.16.0.2`。
- `bin/gripper-interface` 负责夹爪。
- `run2.sh` 调用 `auto_scripts/auto_arm.sh config/charmander.yml`。
- `run3.sh` 调用 `auto_scripts/auto_gripper.sh config/charmander.yml`。
- `auto_arm.sh`/`auto_gripper.sh` 会在进程退出后自动重启，所以故障表现是“能动一下、server 崩溃、重启后又能动一下”。

`config/charmander.yml` 里的地址和端口是显式写入的，不会自动跟随 DHCP 地址变化。当前应保持：

```yaml
PC:
  IP: 172.16.0.3
NUC:
  IP: 172.16.0.1
ROBOT:
  IP: 172.16.0.2
```

Arm 的 PUB/SUB 端口为 `5544/5543`，gripper 为 `5558/5557`。Client 和 NUC 两端必须使用一致的配置。

## 3. 这次故障的结论

### 3.1 根因链

NUC 的 `eno1` 是 Intel I219-V/e1000e。故障时它从 1000 Mbps 降到了 100 Mbps：

- 2026-08-04 15:51：link up 1000 Mbps
- 2026-08-04 16:18：link down，随后以 100 Mbps 恢复
- 2026-08-05 的复现中：link 再次掉线并以 100 Mbps 恢复；FCI 连接后只处理约十几条 action 就退出

插拔/更换物理链路并恢复到 1000 Mbps 后，控制网络问题消失。以后启动控制前应把下面的结果为 `1000` 当作硬性条件：

```bash
cat /sys/class/net/eno1/speed
ethtool eno1 | grep -E 'Speed|Duplex|Link detected'
```

普通的 100 Hz ping 即使零丢包，也不能证明 1 kHz FCI 实时流量稳定。需要同时检查协商速率、link flap、延迟尖峰和 e1000e 日志。

### 3.2 为什么日志只看到 terminate

可见错误是：

```text
terminate called without an active exception
auto_arm.sh: ... Aborted (core dumped) bin/franka-interface ...
```

它很可能是第二层错误。`franka_control_node.cpp` 中有一个可 join 的 `std::thread control_msg_sub`；当 `robot.control()` 因网络/FCI 异常抛出时，栈展开过程中线程对象先被析构，C++ 会调用 `std::terminate()`。因此真正的 libfranka 网络异常被掩盖了。`StatePublisher` 也存在类似的空析构/线程生命周期风险。

建议后续修复 server：用 RAII 管理 stop flag 和 `join()`，并在控制函数内部捕获、记录原始 `franka::Exception` 后再清理线程。对应上游现象可参考 [deoxys_control issue #20](https://github.com/UT-Austin-RPL/deoxys_control/issues/20)。

### 3.3 已排除或确认的信息

- NUC 使用 `PREEMPT_RT` 内核，`/sys/kernel/realtime=1`。
- 实时权限、performance governor、CPU/内存/磁盘没有发现明显问题。
- Client 和 NUC 的关键源文件及配置内容一致。
- NUC 链接的是 libfranka 0.15。
- FCI 未开启时日志是 `Connection to FCI refused`；开启后才进入控制并暴露网络异常。
- 当时没有可用的 core dump：apport 目录为空，`coredumpctl` 无记录。
- NUC 同时存在 `eno1=172.16.0.1/24` 和 Wi-Fi `wlp58s0=172.16.0.12/24`，两张网卡位于同一子网会带来 ARP flux/非对称路由风险。长期建议让机器人控制网成为独立子网，或避免 Wi-Fi 与有线控制口使用同一网段。

## 4. 下次启动和排障清单

在 NUC：

```bash
ip -br addr
ip route
cat /sys/class/net/eno1/speed
ethtool eno1
dmesg -T | grep -Ei 'e1000e|eno1|link.*(up|down)'
ping -c 5 172.16.0.2
```

在 FrankaControl：

```bash
ip -br addr
ip route get 172.16.0.1
ping -c 5 172.16.0.1
```

判断顺序：

1. 两端控制网 IP 是否仍为 `.3` 和 `.1`。
2. NUC `eno1` 是否为 1000 Mbps/full duplex。
3. 是否有新的 link down/up 或 e1000e reset。
4. NUC 是否能稳定到达 robot `.2`。
5. Client/NUC 使用的 YAML 是否一致。
6. 最后才启动 arm/gripper server、打开 FCI，再启动 client。

## 5. 双 RealSense 和新的录制脚本

已新增但未覆盖 V3 的脚本：

```text
examples/run_deoxys_with_space_mouse_V3_record.py
```

默认相机映射：

- front：序列号 `327122071654`，数据字段 `color_image2`
- wrist：序列号 `001622071252`，数据字段 `color_image1`

建议把两台相机分别接到 FrankaControl 的两个 USB 3.x root branch/独立高速口。避免共用无源 hub，也不要落到 USB 2.0。插好后验证：

```bash
rs-enumerate-devices -s
lsusb -t
```

两台设备应都被列出，`lsusb -t` 应显示 5000M 或更高，而不是 480M。本次检查时两台 RealSense 都未插入，所以序列号、USB 拓扑和流配置还没有做真机验证。

录制快捷键：

- `b`：开始一个 episode
- `e`：停止录制，保留在内存中等待决定
- `s`：保存为 `.pkl`
- `d`：丢弃本次 episode
- `r`：关节复位；录制中禁用
- `q`：退出
- SpaceMouse 右键：录制中等价于停止；空闲时退出

启动示例：

```bash
python examples/run_deoxys_with_space_mouse_V3_record.py \
  --interface-cfg config/charmander.yml \
  --controller-type OSC_POSE \
  --vendor-id 9583 \
  --product-id 50746 \
  --front-camera-serial 327122071654 \
  --wrist-camera-serial 001622071252 \
  --output-root data/raw_spacemouse \
  --task-name move_card \
  --randomness low
```

添加 `--save-depth` 后会保存与彩色图对齐的 Z16 深度：

- `depth_image1`：wrist
- `depth_image2`：front
- `front_depth_scale_m`/`wrist_depth_scale_m`：Z16 单位到米的比例

RealSense API 原生可以提供 RGB、Z16 depth、内参、外参和设备时间戳。原来的 `DualRealSenseVideoRecorder` 已经开启和对齐了 depth stream，但捕获循环只取彩色帧，因此旧数据实际上只有 RGB MP4。

## 6. 录制数据格式

旧的 observe 方案保存：

- front/wrist 两个 RGB MP4
- `camera_frame_timestamps.jsonl`
- `metadata.json`
- `replayed_joint_trajectory.json` 和 `.npz`

它被动记录 state，不记录 client 实际发送的 action。旧的低维 SpaceMouse 脚本另有 HDF5/NPZ 格式，但没有形成统一的双相机 action-state episode。

新脚本保存一个 FurnitureBench/robust-rearrangement 风格的 raw pickle：

```python
{
    "furniture": "move_card",
    "observations": [
        {
            "color_image1": wrist_rgb,
            "color_image2": front_rgb,
            "robot_state": {
                "ee_pos": ...,
                "ee_quat": ...,
                "ee_pos_vel": ...,
                "ee_ori_vel": ...,
                "joint_positions": ...,
                "joint_velocities": ...,
                "joint_torques": ...,
                "gripper_width": ...,
            },
        },
        ...
    ],
    "actions": [...],       # OSC_POSE 7-D action
    "rewards": [...],       # 当前填 0
    "skills": [...],        # 当前填 0
    "metadata": {...},
}
```

这是 schema-compatible 的本项目实现，并非保证对 robust-rearrangement 的处理脚本完全免修改。尤其是机器人状态定义、四元数约定、任务名和可选 depth 字段，应在正式训练前做一个适配器和小样本验证。robust-rearrangement 的 README 说明其 raw demo 是 FurnitureBench `.pkl`，再处理为 Zarr；官方字段说明见 [FurnitureBench dataset format](https://clvrai.github.io/furniture-bench/docs/tutorials/dataset.html) 和 [robust-rearrangement](https://github.com/ankile/robust-rearrangement)。

raw pickle 会把图像暂存在内存，适合短 episode。224×224、双 RGB、20 Hz 约 361 MB/分钟；加双 depth 约 602 MB/分钟，未计 Python/pickle 开销。长时间连续采集更适合保留视频+时间戳，或直接流式写 HDF5/Zarr。

## 7. 尚未完成的真机验证

- 两台 RealSense 目前未连接，尚未验证实际 USB 带宽、深度 profile 和序列号映射。
- 新脚本只完成 `py_compile` 和 `--help` 验证，没有在 FCI/机械臂上发送 action。
- 首次真机测试应让机械臂处于安全姿态、低速、有人可按急停；先用 `--no-cameras` 验证按键状态机，再插相机录 3–5 秒样本并检查 pickle。
