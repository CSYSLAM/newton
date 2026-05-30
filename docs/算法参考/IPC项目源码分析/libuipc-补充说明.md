# libuipc 补充说明

本文件只保留一个提醒：

- `libuipc` 目录巨大，本文档只抓了 CUDA backend 的主推进链。
- 如果后续你要继续深挖，最值得继续拆的是这四组模块：
  - `finite_element/*_constitution*`
  - `affine_body/*`
  - `active_set_system/*`
  - `contact_system/contact_models/*`

建议的进一步阅读顺序：

1. `engine/advance_ipc.cu`
2. `collision_detection/global_trajectory_filter.cu`
3. `contact_system/global_contact_manager.cu`
4. `finite_element/fem_linear_subsystem.cu`
5. `affine_body/abd_linear_subsystem.cu`
6. `engine/advance_al.cu`
