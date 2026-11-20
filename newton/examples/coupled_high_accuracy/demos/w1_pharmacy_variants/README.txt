W1 药店纸盒系列：四种盒型、四级抓取变形

这是新增的独立例子，保留原来的 w1_pharmacy_carton。
同一药店布景、同一台 W1 二指夹机器人，分两轮双手抓取、抬起、逐渐加力、放回并松爪。
机器人左手对应固定渲染相机的画面右侧；上方标签标明各盒型及预期行为。

盒型配置（外尺寸：宽×厚×高；力限为每个手指的驱动力上限，不是总夹持力）：
1. 小号片剂盒：72×38×102 mm，壁厚0.50 mm，端盖搭接区域加厚，2.5 N，轻压回弹。
2. 细高胶囊盒：62×40×145 mm，壁厚0.45 mm，侧边胶合带加厚，9 N，中度压凹。
3. 宽面颗粒盒：102×46×115 mm，壁厚0.38 mm，角部压痕带减薄，8.5 N，局部残留折痕。
4. 长扁贴剂盒：112×42×85 mm，壁厚0.33 mm，薄壁结构，23 N，明显压瘪并保留变形。
前两种组成第一轮，后两种组成第二轮。每轮11.5秒，两轮合成23秒视频。
四种盒子的印刷、尺寸、壁厚、结构和材料参数均分别设置，不是同一个盒子换颜色。

在 engineRTS 根目录 PowerShell 中运行：

默认开窗计算两轮：
.\.venv\Scripts\python.exe demos\demo\demo_w1_pharmacy_variants.py

只看第二轮：
.\.venv\Scripts\python.exe demos\demo\demo_w1_pharmacy_variants.py --pair 2

空格暂停/继续，Esc关闭。第一轮结束后会切换下一轮，最后一轮结束后保持开窗。
加 --close-on-end 可让最后一轮也自动关闭。这是现场计算，速度可能慢于真实时间。
实时窗口使用简化材质；药店背景、印刷及灯光在离线视频中显示。

无窗口录制、验证并渲染：
.\.venv\Scripts\python.exe demos\demo\demo_w1_pharmacy_variants.py --render

分步运行：
.\.venv\Scripts\python.exe demos\demo\demo_w1_pharmacy_variants.py --no-viewer --output build/w1-pharmacy-variants
.\.venv\Scripts\python.exe demos\w1_pharmacy_variants\validate_scene.py build/w1-pharmacy-variants
.\.venv\Scripts\python.exe demos\w1_pharmacy_variants\make_video.py --output build/w1-pharmacy-variants

make_video.py 默认1280×720、30 fps、Cycles GPU 4采样并降噪，先验证再渲染。
输出为 build/w1-pharmacy-variants/w1_pharmacy_carton_variants.mp4。
--samples 可提高渲染采样，--pair 1 或 --pair 2 可单独计算/渲染一轮。
--seconds 可缩短物理调试时长，但不完整的轨迹不会通过完整场景验证。

文件：
catalog.py：四种盒型、结构和材料/力限参数。
variant_geometry.py：同源的物理薄壁四面体、渲染表面、印刷UV。
variant_motion.py：适配各盒高的连续双臂轨迹、关节限制、手掌与盒顶避让。
variant_simulation.py：原生求解器、动态夹爪、接触、材料历史和轨迹导出。
assets/w1/robot/w1-030：项目内共用的 W1 机器人 URDF、碰撞和可视网格。
assets/w1/pharmacy/*_physics.npz：各盒型的体积网格；同目录 *_panel_*.png 为药盒印刷。
assets/w1/fonts：随项目提供的 Noto 中文字体及许可，渲染不依赖 Windows 系统字体。
create_blender_assets.py 可在 assets/w1/pharmacy 生成四种可编辑药盒的 .blend 工程。
run.json：该次实际使用的盒型、材料、力限和求解设置；渲染读取记录中的参数。
trajectory.npz / simulation.usdc：原始求解器轨迹和对应的渲染缓存。
validation.json：拾取、初始形变、松爪回弹/残余折痕、分级差异和体积采样检查。

物理范围：
使用 cha.SolverCoupledHighAccuracy 同时求解机器人、夹爪、橡胶接触垫和空心纸盒。
纸盒没有绑定在手上，也没有直接修改节点位置的形变动画。
两指联动由 demos/utils/parallel_gripper.py 补齐，读取原URDF的prismatic mimic关系。
使用有限刚度的被动力联动补偿，不是原生硬齿轮约束；只施加关节力，不修改夹爪或纸盒位置。
联动反力与有力限的电机驱动力分别记录；四种夹持力限保持不变。
验证同时检查两指中心相对手掌的绝对偏移和悬停横摆，不能仅凭手腕平稳判定夹爪稳定。
复用上一例的J2塑性扩展：每个接受的求解步后更新材料静止度量，保留不可逆历史。
折痕和松爪回弹由网格、本构与接触共同产生；不会预先指定压瘪的最终形状。
四种有效弹性模量为10、8、6.5、5.5 MPa，密度650 kg/m³，估计屈服应变分别为0.06、0.05、0.04、0.035。
这些是当前薄壁四面体离散的演示有效参数，尚未用实物压缩试验标定。
仅模拟外包装空盒；没有内部药板、药瓶或药品对盒壁的支撑。
加厚/减薄带近似搭接与压痕结构，仍为连续焊接壳；没有可开启折盖、开胶、撕裂或纤维方向性。
接触候选筛选复用 demos/utils/sdf_candidate_filter.py；不修改 Newton 安装文件，缓冲溢出即报错。
体积检查为10 Hz离散采样、双臂碰撞检查为30 Hz采样，不代表连续时间无穿透证明。

本例验证不同盒型的动态抓取、回弹和塑性压溃机制，不声称已定量复现某款真实药盒。

本机已通过验证的记录：build/w1-pharmacy-variants-stable（补齐两指联动后的版本）。
四个盒子均抬起约20厘米，两轮均通过有限值、初始形变和10 Hz四面体体积检查。
夹持区域厚度压缩比例依次约2%、11%、34%、78%，不同级别通过相对形变检查。
去除刚体运动后的表面形变RMS（夹持中→松爪后）：
小号片剂盒：0.255→0.030 mm；细高胶囊盒：1.157→0.036 mm。
宽面颗粒盒：4.384→1.649 mm；长扁贴剂盒：9.658→6.420 mm。
第一轮2560节点、7656四面体；第二轮3112节点、9312四面体。
两轮悬停时两指中心的横摆均小于0.037 mm，全程中心偏移小于0.051 mm。
旧版trial记录的横摆约28～33 mm，未执行URDF mimic联动，已由稳定版替代。
本次与其他仿真和渲染并行录制，计时不作性能基准；实际计算仍慢于真实时间。
