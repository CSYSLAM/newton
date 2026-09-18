# W1 遥操整机卡死排查

先保存重启前的证据，再做一次只改变一个因素的对照。不要连续启动场景碰运气。

## 2026-09-18 这次观察

- 用户报告整机画面、键鼠不响应，必须强制重启。
- 上一次 boot 的系统日志在 10:30:12 启动 W1 遥操后中断；本次 boot 开始于 10:31:45。
- 启动参数包含 `--viewer null --graph-capture`。软体方块使用当前默认开启设置。
- 机器是 RTX 5090 D v2、NVIDIA Open Kernel Module 595.71.05、Ubuntu 22.04.5、内核 6.8.0-138。
- 桌面 framebuffer 也在 NVIDIA GPU 上。启动前日志记录温度 43°C、显存约 2964 MiB。
  日志中的 `600.00 W` 是 `power.limit` 上限，不是实时功耗。
- 遥操日志已输出 HTTP 服务地址，最后几行是物理接触和多层求解模块加载信息。
  目前没有逐帧日志，不能从最后一行断言是哪一个 kernel、是否处于 graph capture、运行了多少帧。
- 已保存的旧 `active-run` 含 PID 285940；启动器只在 `/healthz` 就绪成功后写入 PID，
  而此场景在首帧 GPU 状态下载完成后才设置就绪。因此根据现有代码和标记推断，
  这次至少完成了首帧及初始 graph capture，故障应继续向后续运行或设备状态切换排查。
- 本次 boot -1 没有保存到 Xid、OOM、kernel panic 或 PCIe AER 故障记录。
  内核最后的 `pm_runtime_work` 耗时警告在正常开机时也存在，不能单独判定为死机原因。
- 历史日志中有 9 月 8 日、14 日、17 日的 Xid 31 内存访问错误，来自不同进程；
  它们早于本次头部修改，但还不能证明与这次硬锁同源。
- journald 已持久化；本次 systemd-pstore 因条件不满足跳过，没有可直接使用的崩溃转储。

**当前结论：优先排查 GPU 内核执行、驱动/GSP 和 PCIe/供电路径。尚未证实具体根因。**
应用非法访问、长时间 GPU kernel、驱动故障或硬件问题都仍是候选。
没有 Xid 不代表 GPU 正常，也不能仅凭整机卡死就排除应用代码触发。

## 1. 重启后保存证据

在仓库根目录执行。这个脚本不会启动 CUDA、调用 nvidia-smi、停止进程或修改驱动：

```bash
uv run python scripts/collect_quest_webxr_w1_diagnostics.py
```

输出在 `recordings/diagnostics/<时间>-<PID>/`，包含上一轮内核日志、系统日志尾部、
历史错误、版本、工作区差异、最近三份遥操日志及 `active-run`。
该目录被 Git 忽略，不会自动上传。应在清除异常运行标记或再次启动前执行。

## 2. 下次手动启动时记录阶段和调用栈

以下命令会真正启动仿真。先保存证据，在有人能处理再次卡死的情况下手动执行。
如果异常运行标记阻止启动，先确认机器已重启且旧进程不存在，再运行场景停止脚本清理标记。
不要批量执行反复启动、停止测试。

第一组先关闭 graph 和方块：

```bash
NEWTON_WEBXR_GRAPH_CAPTURE=0 ./scripts/start_quest_webxr_w1_bag_packing_teleop.sh --no-soft-cube --diagnostics recordings/diagnostics/probe_a_01.log
```

每次使用新的诊断文件名；已有文件不会被覆盖。日志包括设备初始化、初始 IK、
物理求解器建立、最初几帧、graph capture/launch 和运行心跳。
超过 15 秒没有新检查点会定时打印 Python 线程调用栈。慢编译、暂停也可能触发打印，
不能把打印本身视为死锁。检查点执行 `fsync`，但整机硬锁仍可能阻止最后的写入或栈采集。
`host-return` 仅表示 CPU 调用返回，不表示 GPU 已经执行完成。
此模式没有加入 CUDA 同步，但日志落盘会改变时序，不适合用来测 FPS。

第一组稳定后，每次停止旧进程并更换日志名，按以下顺序手动对照：

| 组 | `NEWTON_WEBXR_GRAPH_CAPTURE` | 方块参数 | 目的 |
| --- | --- | --- | --- |
| A | `0` | `--no-soft-cube` | 检查不含 graph 和新增四面体的场景 |
| B | `0` | `--soft-cube` | 单独加入方块 |
| C | `1` | `--soft-cube` | 在 B 的基础上加入 graph |

一次不死机不足以说明已经解决。记录每组运行时长、是否接入 Quest、是否开始操作、
卡死发生于启动/运行/退出，以及另一台机器还能否 ping/SSH。
如果 A 也卡死，graph 和方块都不是必要条件；若只有 B/C 出现问题，再按阶段日志缩小范围。
目前没有自动修改正常启动的 graph、物理参数、驱动、BIOS 或功率设置。

## 3. 区分桌面卡死和内核硬锁

用另一台机器通过有线网 SSH 保持连接，提前运行：

```bash
ssh 用户名@仿真机IP 'journalctl -kf -o short-precise' | tee w1-kernel-live.log
```

若桌面死机但 SSH 仍能响应，立即保存日志和 NVIDIA 报告；不要先重启抹掉现场。
NVIDIA 提供以下诊断命令，需本机管理员权限，保存到明确目录后再决定是否提交给厂商：

```bash
sudo nvidia-bug-report.sh --safe-mode --extra-system-data --output-file /tmp/nvidia-w1-bug-report.log.gz
```

如果 SSH 也完全失联，下一步应配置有线网 `netconsole` 把内核日志发到另一台机器，
或配置适合本机的 pstore/kdump。SSH/journald 都依赖用户态仍能运行，不能保证捕获硬锁末尾。
netconsole 也不能捕获内核根本没有输出的错误。需要知道接收机 IP、MAC 和网卡后再配置，
不要直接套用任意地址或修改启动参数。

## 参考

- [NVIDIA Xid 收集与诊断说明](https://docs.nvidia.com/deploy/xid-errors/working-with-xid-errors.html)。
- [Linux netconsole 文档](https://www.kernel.org/doc/html/latest/networking/netconsole.html)。
- [同为 595.71.05 的 Blackwell 故障报告 #1151](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/1151)：
  这是另一块 RTX 5080 的公开报告，其 Xid 79 在本次日志中没有出现，只能作比对线索，不能当成本机根因。
