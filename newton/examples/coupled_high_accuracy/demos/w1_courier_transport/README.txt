W1 双臂快递袋：每只二指夹爪夹一袋，同时提起、抖动三次、松爪落地。

窗口运行（仓库根目录 PowerShell）：
.\.venv\Scripts\python.exe demos\demo\demo_w1_courier_transport.py

记录 15 秒物理仿真：
.\.venv\Scripts\python.exe demos\demo\demo_w1_courier_transport.py --viewer=false

检查并渲染预览视频：
.\.venv\Scripts\python.exe demos\w1_courier_transport\make_video.py

输出目录：build/w1-courier
视频：w1_dual_courier_transport.mp4
Blender 工程：reference_scene.blend（需要同目录 simulation.usdc 缓存）
原始轨迹：trajectory.npz
检查报告：validation.json、video_verification.json

机器人资产位于项目 assets/w1/robot/w1-030/robot.urdf，袋子贴图位于 assets/w1/courier。
路径根据脚本所在位置解析，不需要本机外部 Newton 工作区。
参考双向抓取示例的 W1 V030 模型和二指夹爪，物理统一使用 s3d.NewtonSolver；不使用原例的 MuJoCo/VBD 求解器。
默认 8 子步、4 次非线性迭代、50 次线性迭代。机械臂通过 IK 产生关节驱动目标，包裹靠接触与摩擦运动。
左右同时操作；0–3 秒靠近，3–4 秒夹紧，4–4.5 秒保持夹紧，4.5–7 秒提起，7–10 秒抖动，10 秒后打开夹爪并后撤。
快递袋材质与包裹几何沿用前一场景；软包内部保留有限刚度粘附弹簧，没有机器人到袋子的固定约束。
为了满足 ABD 正定要求，仅在本场景导入时对退化惯量的第二矩做最小正值修正，原 URDF 文件未修改。
这不是实物机器人控制标定，也没有把袋子材料标定为某一真实产品。包含关系检查采用离散采样。
抓取位置按完整原生夹指碰撞网格校核：TCP 向袋口外侧退 20 mm、降低 4 mm，避免夹指进入载荷区域。
提起时向前回移 20 mm，抖动和后撤保留原来的水平工作区，避免把末端推到 IK 可达边界。
夹指内侧增加 3 mm 厚的平面橡胶接触垫，稳定夹住收拢的袋膜；载荷仍通过接触和摩擦传递。
橡胶垫随动态夹指运动，继承机器人关闭自碰撞的设置，并使用 URDF 的完整夹指质量。
控制台新增右、左夹爪实际指间距（mm）；记录中保存夹爪到袋子的对应关系及抓取、停留、搬运参数。
validate_w1.py 对新记录额外检查抖动期间实际指间距和袋膜与双指的距离；旧记录仍使用原有检查。
2026-11-16 复测：两次独立的 15 秒运行均通过 18 项检查和机械臂末段静止检查，10 项回归测试通过。
两次抖动期间硬包最低抬升约 20 cm、软包约 17 cm，载荷的每秒采样包含率均为 100%。
复测数据位于 build/w1-grasp-review/pads-filtered-1、pads-final-2，汇总为 build/w1-grasp-review/review.json。
新场景不修改 Piper 场景；先前 Piper 高清渲染已停止，已完成的高清帧仍保留，可另行续渲。

早期预览的历史记录通过 14 项轨迹检查。抖动期间硬包最小抬升约 11.8 cm、软包约 14.4 cm；
两侧夹爪各有三次往复峰值。两个载荷的采样包含率均为 100%，最小四面体体积比约 0.325。
物理粒子的最低高度约 0.86082 m，台面高度 0.86 m。
15 秒仿真记录耗时约 339 秒，不是实时性能。包含率采用每秒采样，不能代替连续无穿透证明。

高清稳定版（2026-10-29）：
输出目录 build/w1-courier-stable3，视频 w1_dual_courier_transport_1080p.mp4。
采用固定站姿：ANKLE、KNEE、BUTTOCK、WAIST、NECK1、NECK2 在零位固定，双臂与四个夹指仍为动态关节。
原机器人 URDF 不修改；导入时生成 build/w1_fixed_stance.urdf，网格路径由项目内资产位置解析。
机械臂位置刚度 12000、速度阻尼 350，目标速度与位置轨迹一致；夹指维持 50000/100。
松爪保持在 10–10.2 秒，后撤延长至 10.8 秒，随后保持姿态。
末段实际运动检查：最后一秒全机器人最大位置跨度约 0.0103 mm（旧版约 59.1 mm），
最大相邻帧位移约 0.00645 mm、转角约 0.00202 度，未使用渲染后处理平滑。
新版再次通过 14 项抓放检查，两个载荷在抖动段最小抬升均约 20 cm。

重新记录并渲染高清版：
.\.venv\Scripts\python.exe demos\demo\demo_w1_courier_transport.py --viewer=false --output build/w1-courier-stable3
.\.venv\Scripts\python.exe demos\w1_courier_transport\make_video.py --output build/w1-courier-stable3 --hd
原生 1920x1080、64 次 Cycles 采样、降噪、30 fps、H.264 CRF 16。
