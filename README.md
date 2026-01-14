# Python CarSim Environment

一个基于 CarSim/VehicleSim 的 Python 强化学习环境封装，提供类似 OpenAI Gym 的接口，支持车辆动力学仿真与自动控制算法开发。

## 项目概述

本项目将 CarSim 车辆仿真器封装为 Python 环境，通过 ctypes 调用 CarSim 的 C API，实现高性能的车辆动力学仿真。项目包含：

- **CarSim 环境封装** (`carsim_env.py`): 提供 Gym 风格的 `reset()` / `step()` 接口
- **VS Solver 接口** (`vs_solver.py`): 底层 CarSim DLL 调用封装
- **传统控制器** (`simple_controller.py`): PID 路径跟踪控制器
- **强化学习训练** (`train_carsim_sac_v3.py`): 基于 SAC 算法的路径跟踪训练脚本
- **SAC 算法实现** (`SACModule.py`): Soft Actor-Critic 强化学习算法模块

## 核心特性

### 性能优化
1. **零拷贝 NumPy 视图**:  使用 `np.ctypeslib.as_array()` 直接映射 ctypes 缓冲区，避免数据拷贝
2. **持久化缓冲区**: ctypes 数组在 `reset()` 后复用，减少内存分配开销
3. **批量积分步**:  `control_step(action, inner_steps)` 支持一个控制动作执行多个仿真步，减少 Python 层调用
4. **向量化赋值**: 使用 NumPy `copyto()` 进行动作写入，避免 Python 循环

### 接口设计
- **Gym 兼容**: 类 OpenAI Gym 接口，易于集成现有 RL 框架
- **灵活的动作空间**: 支持任意长度的动作序列，自动截断或补零
- **丰富的观测空间**: 直接访问 CarSim 的导出变量
- **错误处理**: 完善的 VS Solver 错误信息捕获与分类

## 环境准备

### 系统要求
- Windows 操作系统（CarSim 需要）
- Python 3.7+
- CarSim/VehicleSim 许可证

### 依赖安装
```bash
pip install numpy torch gymnasium tensorboard
```

### 仿真文件
确保有有效的 CarSim 仿真文件 (`*.sim`)，包含：
- 车辆模型配置
- 场景设置（道路、环境等）
- 时间步长参数（`t_start`, `t_stop`, `t_step`）
- 输入/输出变量定义（Import/Export 变量）

## 快速开始

### 基础使用示例

```python
from carsim_env import CarSimEnv

# 创建环境
sim_path = r"./simfile.sim"
env = CarSimEnv(sim_path)

# 重置环境并获取初始观测
obs = env.reset()
print("初始观测 (前6个变量):", obs[:6])

# 执行单步
action = [0.2, 0.0, 0.1]  # [油门, 刹车, 转向角]
obs, reward, done, info = env.step(action)

# 关闭环境
env.close()
```

### 多步控制周期（推荐）

```python
from carsim_env import CarSimEnv

sim_path = r"./simfile.sim"
control_dt = 0.1  # 控制周期 100ms

env = CarSimEnv(sim_path)
obs = env.reset()

# 计算每个控制周期需要多少个仿真步
t_step = env.config['t_step']
inner_steps = max(1, int(round(control_dt / t_step)))

done = False
while not done:
    # 根据观测选择动作
    action = [0.2, 0.0, 0.1]  # 示例动作
    
    # 执行 inner_steps 个仿真步
    obs, reward, done, info = env.control_step(action, inner_steps)
    
    if info. get('error'):
        print('仿真错误:', info['error'])
        break

env.close()
```

### 使用简单控制器

```python
from carsim_env import CarSimEnv
from simple_controller import SimplePathFollower

sim_path = r"./simfile.sim"
control_dt = 0.01
target_speed = 50  # km/h

env = CarSimEnv(sim_path)
controller = SimplePathFollower()

obs = env.reset()
controller.reset()

t_step = env.config['t_step']
inner_steps = max(1, int(round(control_dt / t_step)))

done = False
while not done:
    # 从观测中提取状态
    lateral_error = obs[4]  # 横向误差
    current_speed = obs[5]  # 当前车速
    
    # 计算控制动作
    throttle, brake, steering = controller.control(
        current_speed, target_speed, lateral_error, dt=control_dt
    )
    
    action = [throttle, brake, steering]
    obs, reward, done, info = env.control_step(action, inner_steps)

env.close()
```

## API 详解

### CarSimEnv 类

#### 主要方法

**`__init__(sim_path:  str)`**
- 初始化环境，加载 CarSim DLL
- `sim_path`: CarSim 仿真文件路径 (`.sim` 文件)

**`reset() -> Tuple[float, ...]`**
- 重置仿真到初始状态
- 返回初始观测（Export 变量元组）

**`step(action:  Sequence[float]) -> Tuple[Tuple[float, ...], float, bool, dict]`**
- 执行单个仿真步（调用 `control_step(action, inner_steps=1)`）
- 返回:  `(obs, reward, done, info)`

**`control_step(action: Sequence[float], inner_steps: int) -> Tuple[Tuple[float, ...], float, bool, dict]`**
- 推荐使用：用一个动作执行多个仿真步
- `action`: 控制输入序列
- `inner_steps`: 内部积分步数
- 返回: `(obs, reward, done, info)`

**`observation() -> Tuple[float, ...]`**
- 返回当前观测值的快照

**`close()`**
- 终止当前仿真

#### 属性

- `import_np`: NumPy 数组视图，映射输入变量（零拷贝）
- `export_np`: NumPy 数组视图，映射输出变量（零拷贝）
- `import_vars`: 输入变量的 Python 列表（用于调试）
- `export_vars`: 输出变量的 Python 列表（每步同步）
- `config`: 仿真配置字典（包含 `t_start`, `t_stop`, `t_step`, `n_import`, `n_export`）

### 动作空间

动作为任意长度的序列：
- 前 `n_import` 个元素写入 CarSim 的 Import 变量
- 超出部分被忽略
- 不足部分补零
- 标量会被广播到第一个输入变量

示例：
```python
# 完整动作
action = [0.2, 0.0, 0.1]  # [油门, 刹车, 转向]

# 标量动作（仅控制第一个输入）
action = 0.5

# 过长动作（自动截断）
action = [0.2, 0.0, 0.1, 0.0, 0.0]  # 多余的被忽略
```

### 观测空间

观测为 `n_export` 长度的浮点数元组，对应 CarSim 的 Export 变量。

常见变量（顺序取决于 `.sim` 文件配置）：
- 时间 (s)
- X/Y 位置 (m)
- 车速 (km/h)
- 横向偏差 (m)
- 航向角偏差 (rad)
- 行驶距离 (m)
- ...

### Info 字典

```python
info = {
    'return_code': 0,        # VS Solver 返回码（0=继续，非0=停止）
    'error': ".. .",          # 错误信息（仅当发生错误时）
    'end_reason': "..."      # 停止原因（正常结束时）
}
```

## 强化学习训练

### SAC 算法训练示例

```bash
python train_carsim_sac_v3.py --sim-path ./simfile.sim --episodes 1000
```

### 训练脚本特性

**观测空间设计 (v3)**:
- 不包含行驶距离 `s`，避免策略记忆特定位置
- 包含上一帧动作 `[lon_prev, steer_prev]`，提供历史信息
- 观测向量:  `[lat_n, spd_n, tgt_spd_n, lon_prev, steer_prev]`
  - `lat_n`: 横向误差归一化 ∈ [-1, 1]
  - `spd_n`: 车速归一化 ∈ [-1, 1] (0~160 km/h)
  - `tgt_spd_n`: 目标车速归一化 ∈ [-1, 1]
  - `lon_prev`, `steer_prev`: 上一帧动作 ∈ [-1, 1]

**动作空间**:
- 2 维连续动作 ∈ [-1, 1]²
- `action[0]`: 纵向控制 (lon)
  - `lon ≥ 0`: `throttle = lon`, `brake = 0`
  - `lon < 0`: `throttle = 0`, `brake = -lon * 10.0`
- `action[1]`: 转向控制 (steer_raw)
  - `steer = steer_raw * 540.0` (度)

**奖励设计**:
- 前进距离奖励
- 车道保持奖励（横向误差惩罚）
- 速度跟踪奖励
- 动作平滑性惩罚
- 低速惩罚

## 性能建议

1. **使用 `control_step()` 而非 `step()`**: 减少 Python-C 边界调用
2. **直接读取 `export_np`**: 避免频繁访问 `export_vars` 列表
3. **批量控制**: 设置合理的 `control_dt` 与 `inner_steps`
4. **减少打印**: 避免在主循环中打印大量信息

## 常见问题

### Q: 初始状态非零？
**A**:  可能是：
- 二次运行时 DLL 状态未清零（尝试重启 Python 进程）
- 许可证错误导致初始化失败（检查 `info['error']`）
- CarSim 模型本身的初始状态设置

### Q: 过早 `done=True`？
**A**: 检查：
- `info['error']` 是否包含许可证提示
- CarSim License Manager 是否正常运行
- 仿真文件中的停止条件（事件、时间限制等）

### Q: 性能不佳？
**A**: 优化建议：
- 使用 `control_step()` 批量执行多步
- 减少 Python 层的数据转换
- 直接使用 `export_np` 读取观测
- 减少不必要的列表拷贝

### Q: 如何自定义动作/观测？
**A**: 
- 修改 `.sim` 文件中的 Import/Export 变量定义
- 在外部包装 `CarSimEnv` 进行预处理/后处理
- 参考 `train_carsim_sac_v3.py` 中的 `CarSimGymWrapperV3` 示例

## 项目结构

```
python_carsim_env/
├── carsim_env. py              # 主环境封装
├── vs_solver. py               # VS Solver API 接口
├── simple_controller. py       # PID 控制器
├── SACModule.py              # SAC 算法实现
├── train_carsim_sac_v3.py    # SAC 训练脚本
├── simfile.sim               # CarSim 仿真文件（示例）
└── README.md                 # 本文档
```

## 扩展建议

1. **自定义奖励函数**:  在外部根据 `export_np` 计算奖励
2. **动作归一化**: 新增包装器在调用 `control_step()` 前转换动作
3. **数据记录**: 批量收集 `export_np.copy()` 用于训练或分析
4. **多智能体**:  支持多车仿真（需要 CarSim 多车配置）
5. **域随机化**: 随机化车辆参数、道路条件等

## 参考资料

- [CarSim 官方文档](https://www.carsim.com/)
- [OpenAI Gym 文档](https://gymnasium.farama.org/)
- [Soft Actor-Critic 论文](https://arxiv.org/abs/1812.05905)

## 许可证

- `vs_solver.py`: Copyright (c) 2019-2020, Mechanical Simulation Corporation
- 其他代码:  请查看仓库许可证文件

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题，请在 GitHub 仓库中创建 Issue。
