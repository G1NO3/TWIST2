# 本机 PICO → GMR 对照测试（不控制真机）

## 目的和边界

另一台电脑的 TWIST2 + GMR 工作正常，但本机相对控制器接管位移很小。
两者读取的不是同一路位置数据。本测试同时记录：

| 输出 | 数据来源 | 单位 / 坐标系 |
|---|---|---|
| `controller_left/right` | SDK `get_left/right_controller_pose()` | 相对首个有效样本的位移，mm，原始 Unity 轴 |
| `body_left/right` | SDK `get_body_joints_pose()` 的 Left/Right_Wrist | 相对首个有效样本的位移，mm，原始 Body 轴 |
| `body_pelvis_relative_left/right` | 骨架手腕位置减去骨架 pelvis 位置 | 去除 pelvis 平移后的位移，不去除 pelvis 旋转 |
| `gmr_left/right` | GMR 输出 qpos 经 MuJoCo FK 得到的 wrist_yaw_link | 虚拟机器人世界系位移，mm |

终端主要显示三维位移范数，JSONL 保存各轴和原始数据。不要直接比较不同坐标系的各轴正负。
没有 intervention 的 0.3 缩放，没有 X/A clutch，没有生产环境速度限制。
GMR 内部仍使用自身的人体缩放、位置/旋转任务权重和 IK，因此虚拟机器人位移不应期待严格等于人手位移。

入口 `deploy_real/check_pico_gmr.py` 复用 TWIST2 的外部 GMR 类、
`src_human=xrobot`、`tgt_robot=unitree_g1`、默认身高参数 1.6、
`offset_to_ground=True`，以及 XRobotStreamer 的原始坐标转换。
不启动 TWIST2 低层策略、不连接 Redis、不发 DDS、不控制 Inspire 手。
这是 TWIST2 的**输入与重定向部分测试**，不是完整低层闭环跟踪测试。

## 本机准备情况（2026-10-09）

GUI 的 `--preflight` 只导入依赖、加载模型，不初始化 SDK。它可以和正式
GMR 输入进程同时存在，不应触发“Other PICO reader”错误。独占检查仅豁免
本 checkout 中明确实现此模式的脚本；真正的输入进程、旧 teleop 仍会被检查，
状态文件的排他锁也仍然保留。冲突消息现在包含 PID 和脚本名。
修改后重新启动输入进程才会生效；不要通过随意杀进程或关闭互斥检查来处理。

- 独立环境：`/home/jwang3617/pipette/.venv-gmr`，Python 3.10。
- GMR：`../GMR`，提交 `bb1bbe40774794fceb2a7c579a3464a28e68c844`。
- 已补齐 sparse checkout 的 `assets/unitree_g1`，未修改上游 GMR 算法。
- 主要依赖：MuJoCo 3.15.0、mink 1.3.0、DAQP 0.10.3、NumPy 2.2.6、SciPy 1.15.3、CPU Torch 2.6.0。
- 启动脚本通过 PYTHONPATH 复用 `.venv-pico` 中**同一个 SDK 1.0.2 二进制**和底层动态库，未升级 GUI 的 SDK 或依赖。
- 模型加载和人工构造骨架的离线 IK 测试已通过；真实人体动作和桌面窗口仍需现场验证。

## 现场步骤

1. 先使用现有正常停止流程确保真机已安全停止/有人看护，再在 GUI **END SESSION**。
   本测试不负责把机器人移动到安全姿态，也不会杀掉进程。
   不要同时运行 policy bridge、原来的 `teleop.sh` 或其他 PICO SDK 采集程序。
2. 保留 XRoboToolkit PC Service。头显与 PC 连到正确网络，选择本 PC 并开启发送。
   开启 Controller 和 Body Tracking，按另一台电脑的 TWIST2/GMR 流程完成身体校准。
   如果那套流程使用脚环，本次也使用并完成校准；只开启 Controller 不会产生完整骨架。
   关闭头显 App 的 `Switch w/A Button`，避免误按 A 停止发送。
3. 在这台电脑的桌面终端检查环境（不连接 SDK）：

   ```bash
   cd /home/jwang3617/pipette/TWIST2
   bash check_pico_gmr.sh --preflight
   ```

4. 启动 120 秒输入及可视化测试：

   ```bash
   bash check_pico_gmr.sh --seconds 120
   ```

   会打开 G1 的 MuJoCo 窗口。不需要按 Unitree PLAY，也不需要按 PICO X/A。
   `--human-height 1.6` 默认与当前 `teleop.sh` 一致；要对比另一台电脑，请设置为对方同一参数。

5. 身体、头部尽量保持不动。窗口获得焦点后按键盘 **R** 清零测量基准：
   左手缓慢上移约 10 cm，停 2 秒，再回到原处；随后右手做同样动作。
   尽量不要转动手柄；可以再分别测试前后和左右移动。
   **R 仅清零统计基准，不重置真实或虚拟机器人姿态。**
6. 关闭窗口或按 Ctrl+C 结束。日志自动写入 `TWIST2/logs/pico_gmr_compare_时间.jsonl`。
   新日志使用独立文件名，不覆盖已有记录。把路径交给 agent 分析即可。
7. 结束诊断后，再按正常 GUI 流程启动生产会话。不要让诊断和 GUI 同时占用 SDK。

无桌面时，用下面的命令只记录数据和运行 GMR IK：

```bash
bash check_pico_gmr.sh --headless --seconds 120
```

## 拒绝启动 / 等待提示

- `PICO input is in use`：GUI 的 PICO 采集仍占用锁；先安全 END SESSION。
  程序还会检查旧版 teleop 进程。它不会自动终止任何进程，也不会改写生产 `pico_state.json`。
- `no advancing SDK packet`：当前没有持续进入 SDK 的数据，检查服务、头显发送状态和网络。
- `Body unavailable`：没有完整人体骨架，检查 Body Tracking 和校准。
- `Body timestamps are not advancing`：SDK 的骨架可用标志可能锁存了旧数据，但时间戳不再变化。
- `invalid Body joint`：指定骨骼位置/四元数无效；不会将它送入 GMR。
- 控制器无效不阻止有效的骨架进入 GMR，便于发现“两路输入质量不同”。

SDK 没有原子化的全数据快照或控制器独立 tracking-valid 标记。
本工具使用前后时间戳检查做尽力一致性验证，不能将“收到数据”当作位置一定准确。
重新连接或超过 0.5 秒的数据间断会建立新统计段，避免把重定位跳变当成真实运动。

## 结果判读

- 骨架手腕约 100 mm、控制器只有约 10–20 mm：两路输入确实不一致；
  下一步对比另一台 PC 的同一路控制器接口，不能直接归因于本机硬件。
- 两路原始位移均约 100 mm，GMR 正常：原接管数据处理/状态机/执行时序值得优先检查。
- 两路都偏小或不更新：检查 PC 接收链路、头显配置、校准和 SDK/服务差异。
- 骨架合理但 GMR 姿态不合理：进一步比较 GMR 版本、IK 配置和人体身高参数。

之前真机出现的 `publish_tick_gap`（控制循环超过 100 ms）是独立的停止原因；
本测试通过，不代表生产控制线程的时序问题已解决。

## 离线回归

```bash
cd /home/jwang3617/pipette
PYTHONPATH=GMR:.venv-pico/lib/python3.10/site-packages \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-gmr/bin/python -m pytest \
  TWIST2/tests/test_pico_gmr_diagnostic.py TWIST2/tests/test_pico_source.py -q
```

覆盖：原始位移不缩放、固定基准、坏骨架拒绝、无效控制器不阻塞骨架路径、
前后 SDK 时间戳不一致拒绝、有效 JSON 日志、真实 GMR 转换与有限 IK 输出、无机器人 publisher 导入。

如果需要在另一台电脑运行启动脚本，可以设置 `GMR_TEST_PY`、`GMR_ROOT`、`PICO_SDK_SITE`。
Python 与复用的 SDK 二进制必须 ABI 兼容；不要在 Python 3.12 环境中加载这里的 cp310 模块。
