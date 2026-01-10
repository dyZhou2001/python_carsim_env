import ctypes
from typing import Iterable, Sequence, Tuple
import numpy as np

from vs_solver import vs_solver


def _load_dll_with_fallback(path: str):
    """Return CDLL handle, falling back to WinDLL if needed."""
    try:
        return ctypes.CDLL(path)
    except OSError:
        return ctypes.WinDLL(path)


class CarSimEnv:
    """Wrapper that exposes the VS solver with a gym-like interface."""

    def __init__(self, sim_path: str):
        self.sim_path = sim_path

        # 创建 solver 对象并加载 DLL（DLL 在进程内是单例）
        self.solver = vs_solver()
        self.dll_path = self.solver.get_dll_path(sim_path)
        if self.dll_path is None:
            raise RuntimeError("Solver DLL path could not be determined.")

        self.dll_handle = _load_dll_with_fallback(self.dll_path)
        if not self.solver.get_api(self.dll_handle):
            raise RuntimeError("Solver API validation failed.")

        # 初始值占位；真正的配置在每次 reset() 里通过 read_configuration 获取
        self.config = None
        self.n_import = 0
        self.n_export = 0
        self.t_start = 0.0
        self.t_stop = 0.0
        self.t_step = 0.0
        self.t_current = 0.0

        # 这些在 reset() 时按当前配置重新分配
        self._import_c_array = None
        self._export_c_array = None
        self.import_np = None
        self.export_np = None
        self.import_vars = []
        self.export_vars = []

        self.initialized = False
        self.done = True

        # 可选：关闭 VS 的错误弹窗（等价于 VS Command: OPT_ERROR_DIALOG = 0）
        try:
            self.solver.dll_handle.vs_set_opt_error_dialog(0)
        except Exception:
            pass

    # -------------- 核心 API：reset / step / control_step / close --------------

    def reset(self) -> Tuple[float, ...]:
        """
        开始一个新的 episode。

        正确的 VS 调用序列：
          vs_read_configuration(simfile)  # 内含 setdef + initialize
          -> 多次 vs_integrate_io(...)
          -> vs_terminate_run(...)
        """
        # 如果上一个 episode 还没正常结束，先终止上一 run
        if self.initialized and not self.done:
            self.close()

        # 每个 episode 重新读配置 + 初始化 VS Solver
        self.config = self.solver.read_configuration(self.sim_path)
        if self.config is None:
            raise RuntimeError("Configuration read failed.")

        self.n_import = int(self.config.get("n_import", 0))
        self.n_export = int(self.config.get("n_export", 0))
        self.t_start = float(self.config.get("t_start", 0.0))
        self.t_stop = float(self.config.get("t_stop", 0.0))
        self.t_step = float(self.config.get("t_step", 0.0))

        if self.n_import <= 0 or self.n_export <= 0:
            print(self.config)
            raise RuntimeError("Configuration did not report valid import/export counts.")

        self.t_current = self.t_start

        # 每次 run 按当前配置重新分配 Import/Export buffer 和 numpy 视图
        self._import_c_array = (ctypes.c_double * self.n_import)()
        self._export_c_array = (ctypes.c_double * self.n_export)()
        self.import_np = np.ctypeslib.as_array(self._import_c_array)
        self.export_np = np.ctypeslib.as_array(self._export_c_array)

        self.import_np.fill(0.0)
        self.import_vars = [0.0] * self.n_import

        # 初始化时，VS 已经算好第一步输出，用 vs_copy_export_vars 读出来
        self.solver.copy_export_vars_into(self._export_c_array, self.n_export, return_list=False)
        self.export_vars = self.export_np.tolist()

        self.initialized = True
        self.done = False
        return tuple(self.export_vars)

    def step(self, action: Sequence[float]) -> Tuple[Tuple[float, ...], float, bool, dict]:
        """单步控制（一个 action 对应一个积分步）。"""
        return self.control_step(action, inner_steps=1)

    def control_step(self, action: Sequence[float], inner_steps: int) -> Tuple[Tuple[float, ...], float, bool, dict]:
        """
        Perform multiple internal integration steps with one action.

        Parameters
        ----------
        action : Sequence[float]
            控制输入，映射到 Import 数组。
        inner_steps : int
            在同一 action 下进行的 VS 内部积分步数。

        Returns
        -------
        obs : tuple
            当前观测（Export 变量快照）。
        reward : float
            当前步的奖励（这里先返回 0.0，占位）。
        done : bool
            episode 是否结束。
        info : dict
            附加信息：包括 VS 返回码、错误信息等。
        """
        if not self.initialized:
            raise RuntimeError("Call reset() before stepping.")
        if self.done:
            raise RuntimeError("Episode finished. Call reset() to start a new one.")

        # 将 action 写入 Import 数组
        self._write_action(action)

        solver = self.solver
        reward = 0.0
        code = 0
        t_current = self.t_current
        t_step = self.t_step
        t_stop = self.t_stop
        integrate = solver.integrate_io_inplace
        import_c = self._import_c_array
        export_c = self._export_c_array

        for _ in range(inner_steps):
            code = integrate(t_current, import_c, export_c, self.n_export)
            t_current += t_step
            # code != 0 或模拟时间达到 t_stop 都视为 run 结束条件
            if code != 0 or (t_stop > 0.0 and t_current >= t_stop):
                break

        self.t_current = t_current
        info = {"return_code": code}

        # 处理 VS 返回码和错误信息
        if code != 0:
            try:
                if solver.dll_handle.vs_error_occurred():
                    msg_ptr = solver.dll_handle.vs_get_error_message()
                    if msg_ptr:
                        info["error"] = ctypes.c_char_p(msg_ptr).value.decode("ascii", errors="ignore")
                else:
                    info["end_reason"] = "model_requested_stop"
            except Exception:
                info["error"] = "Non-zero return; error classification failed"
            self.done = True

        if t_stop > 0.0 and self.t_current >= t_stop:
            self.done = True

        if self.done:
            # 正确终止这次 run
            solver.terminate_run(self.t_current)

        # 只在最后一次更新 Export 快照
        exports_snapshot = self.export_np.tolist()
        self.export_vars = exports_snapshot

        return tuple(exports_snapshot), reward, self.done, info

    def close(self) -> None:
        """终止当前 VS run（如果还在进行）。"""
        if self.initialized and not self.done:
            # 把当前时间传给 terminate_run
            self.solver.terminate_run(self.t_current)
        self.initialized = False
        self.done = True

    # -------------- 辅助方法 --------------

    def _write_action(self, action: Iterable[float]) -> None:
        """将 action 写入 Import 数组。

        - 如果 action 长度少于 n_import，多余的 import 通道填 0。
        - 如果 action 长度大于 n_import，忽略多余的。
        """
        values = np.asarray(action, dtype=np.float64)
        if values.ndim == 0:
            # 标量 action：只写入第一个 import 通道
            values = np.full(1, float(values))

        limit = min(values.size, self.n_import)
        if limit:
            np.copyto(self.import_np[:limit], values[:limit], casting="unsafe")
        if limit < self.n_import:
            self.import_np[limit:self.n_import] = 0.0

        # 同步到 Python list 版本（便于调试等用途）
        if len(self.import_vars) != self.n_import:
            self.import_vars = [0.0] * self.n_import
        if limit:
            self.import_vars[:limit] = values[:limit].tolist()
        if limit < self.n_import:
            self.import_vars[limit:self.n_import] = [0.0] * (self.n_import - limit)

    def observation(self) -> Tuple[float, ...]:
        """返回当前 Export 变量快照（Python tuple）。"""
        return tuple(self.export_vars)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def make_carsim_env(sim_path: str) -> CarSimEnv:
    """Convenience constructor that mirrors gym.make style."""
    return CarSimEnv(sim_path)


# -------------------- 示例 main，与原脚本类似 --------------------

if __name__ == "__main__":
    import time
    from simple_controller import SimplePathFollower

    example_sim = r"./simfile.sim"
    num_episodes = 35
    max_steps_per_episode = 20000
    control_dt_s = 0.01  # 控制器更新周期

    total_start = time.perf_counter()
    controller = SimplePathFollower()

    for ep in range(num_episodes):
        with CarSimEnv(example_sim) as env:
            obs = env.reset()
            controller.reset()
            k = min(6, len(obs))
            print(f"Episode {ep+1}/{num_episodes} - initial obs (first {k}):", obs[:k])

            start_wall = time.perf_counter()
            start_sim_t = env.t_current
            steps = 0
            done = False
            info = {}

            t_step = env.config["t_step"]
            inner_steps_per_control = max(1, int(round(control_dt_s / t_step)))
            print(f"control_dt={control_dt_s}s, t_step={t_step}s, inner_steps={inner_steps_per_control}")

            target_speed = 50  # km/h

            while not done and steps < max_steps_per_episode:
                lateral_error = obs[4]
                current_speed = obs[5]

                action = controller.control(current_speed, target_speed, lateral_error, dt=control_dt_s)
                obs, reward, done, info = env.control_step(action, inner_steps_per_control)
                steps += 1

            wall = time.perf_counter() - start_wall
            sim_advanced = env.t_current - start_sim_t
            print(
                f"Episode {ep+1} finished: steps={steps}, sim_time={sim_advanced:.6f}s, "
                f"wall={wall:.4f}s, done={done}, info={info}"
            )

    total_wall = time.perf_counter() - total_start
    print(f"All {num_episodes} episodes total wall time: {total_wall:.4f}s")
