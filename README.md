# CarSimEnv 使用说明 (最新版)

该环境提供一个轻量、接近 Gym 风格的 CarSim 求解器封装，并针对性能进行了以下优化：
1. 持久化 ctypes 数组，减少每步重复分配。
2. NumPy 零拷贝视图 (`import_np`, `export_np`) 直接映射底层求解器数据。
3. `control_step(action, inner_steps)` 支持一个控制周期内执行多个内部积分步，只在周期末进行一次 Python 层导出转换。
4. 初始化阶段缓存 `t_start/t_stop/t_step` 并在热路径中复用，减少属性查找。
5. `_write_action` 使用 NumPy 向量化赋值同步 ctypes 缓冲区与 Python 列表，显著降低大批量输入时的 Python 循环开销。

## 环境准备
- 确保 `simfile.sim` 是有效的 CarSim 仿真文件，包含所需的车辆/场景配置与时间步长。
- 保持 `vs_solver.py` 在工作区中（被环境内部调用）。
- 安装依赖：`numpy`。

```powershell
pip install numpy
```

## 快速开始
```python
from carsim_env import CarSimEnv

sim_path = r"D:\ZDY_Drift\pyDrift\simfile.sim"
env = CarSimEnv(sim_path)
obs = env.reset()
print("Initial obs first 6:", obs[:6])

# 单步 (仅做一次内部积分)
obs, reward, done, info = env.step([0.2, 0.0, 0.1])

env.close()
```

## 控制周期批量积分 (推荐)
```python
from carsim_env import CarSimEnv

sim_path = r"D:\ZDY_Drift\pyDrift\simfile.sim"
control_dt = 0.1  # 控制器周期 100ms
env = CarSimEnv(sim_path)
obs = env.reset()
print("Initial obs first 6:", obs[:6])

t_step = env.config['t_step']
inner_steps = max(1, int(round(control_dt / t_step)))
done = False
while not done:
    action = (0.2, 0.0, 0.1)  # 任意长度序列，会截断/补零映射到前 n_import
    obs, reward, done, info = env.control_step(action, inner_steps)
    if info.get('error'):
        print('Solver stop reason:', info['error'])
        break

env.close()
```

## 动作与观测
- 动作为任意长度序列：前 `n_import` 元素写入求解器输入，超出部分忽略，不足补 0；标量会被广播到首个输入。
- `_write_action` 采用 `np.copyto` 直接写入 ctypes 缓冲区，同时维护 Python `import_vars` 列表，方便调试。
- 未做方向翻转或归一化；如需特殊语义请在外部自行处理。
- `export_np` 是零拷贝 NumPy 数组视图，数值变化即时可见；`export_vars` 仅在 `reset()` 和每次 `control_step()` 末尾同步一次。

## 关键方法
- `reset()`：初始化仿真到 `t_start`，刷新导出变量。
- `step(action)`：单一积分步（内部调用 `control_step(inner_steps=1)`）。
- `control_step(action, inner_steps)`：推荐，用一个动作执行多个内部积分步，减少 Python 层开销。
- `observation()`：返回当前导出变量的快照（与 `export_vars` 同步）。
- `close()`：手动结束当前仿真（若未自动终止）。

## Info 字段
`info['return_code']`：求解器返回整数；0 表示继续，非零表示停止。
- 停止时尝试分类：
  - `info['error']`：求解器报告错误（如许可证问题），由 `vs_get_error_message` 直接给出 ASCII 描述。
  - `info['end_reason']`：模型或事件请求正常提前结束。

## 性能建议
- 使用 `control_step` 聚合多个内部步，减少 Python 与 ctypes 边界调用。
- 仅在需要时访问 `export_vars`；高频监控请直接使用 `export_np`。
- 若出现二次 `reset()` 状态未清零的问题，考虑切换到重载版本 `carsim_env_reload.py` 测试许可证或模型状态。
- 初始化后的 `import_np` 与 `export_np` 会在整个过程中复用；调用 `reset()` 时无需额外分配，保持缓存热。
- `vs_solver.copy_export_vars_into(..., return_list=False)` 现可跳过 Python 列表复制，若你手动读取导出缓冲区，可直接复用 NumPy 视图获得更低开销。

## 示例运行
```powershell
python carsim_env.py
```

输出包括每个 episode 的起始观测、控制周期参数、结束状态及总耗时。根据需求调整 `num_episodes`、`control_dt_s`、`max_steps_per_episode`。

## 常见问题
1. 初始状态非零：检查是否为二次运行且使用旧 DLL 状态，或许可证错误导致初始化失败。
2. 过早 `done`：查看 `info['error']` 是否包含许可证提示；确保 License Manager 运行。
3. 性能不足：确认使用 `control_step()` 而不是频繁单步；减少打印；使用 `export_np` 直接读取。

## 下一步扩展建议
- 自定义奖励函数：在外部根据 `export_np` 计算并替换当前 reward。
- 动作预处理/归一化：新增一个包装器在调用 `control_step` 前转换 agent 输出。
- 数据记录：批量收集 `export_np.copy()` 到列表用于训练或分析。


