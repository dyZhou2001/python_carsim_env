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
    """Very small wrapper that exposes the VS solver with a gym-like interface."""

    def __init__(self, sim_path: str):
        self.sim_path = sim_path
        self.solver = vs_solver()
        self.dll_path = self.solver.get_dll_path(sim_path)
        if self.dll_path is None:
            raise RuntimeError("Solver DLL path could not be determined.")

        self.dll_handle = _load_dll_with_fallback(self.dll_path)
        if not self.solver.get_api(self.dll_handle):
            raise RuntimeError("Solver API validation failed.")

        self.config = self.solver.read_configuration(self.sim_path)
        
        if self.config is None:
            raise RuntimeError("Configuration read failed.")

        self.n_import = int(self.config.get("n_import", 0))
        self.n_export = int(self.config.get("n_export", 0))
        if self.n_import <= 0 or self.n_export <= 0:
            raise RuntimeError("Configuration did not report valid import/export counts.")

        self.import_vars = [0.0] * self.n_import
        self.export_vars = [0.0] * self.n_export
        # Persistent ctypes arrays to avoid per-step allocation
        self._import_c_array = (ctypes.c_double * self.n_import)()  # raw memory for solver
        self._export_c_array = (ctypes.c_double * self.n_export)()
        # Zero-copy NumPy views (will reflect solver writes immediately)
        self.import_np = np.ctypeslib.as_array(self._import_c_array)
        self.export_np = np.ctypeslib.as_array(self._export_c_array)
        self.t_start = float(self.config.get("t_start", 0.0))
        self.t_stop = float(self.config.get("t_stop", 0.0))
        self.t_step = float(self.config.get("t_step", 0.0))
        self.t_current = self.t_start
        self.initialized = False
        self.done = True

    def reset(self) -> Tuple[float, ...]:
        if self.initialized:
            self.close()
        self.solver.initialize(self.t_start)
        self.t_current = self.t_start
        self.import_np.fill(0.0)
        self.import_vars = [0.0] * self.n_import
        # Fill export array once; use numpy-backed snapshot for Python exposure
        self.solver.copy_export_vars_into(self._export_c_array, self.n_export, return_list=False)
        self.export_vars = self.export_np.tolist()
        self.initialized = True
        self.done = False
        return tuple(self.export_vars)

    def step(self, action: Sequence[float]) -> Tuple[Tuple[float, ...], float, bool, dict]:
        if not self.initialized:
            raise RuntimeError("Call reset() before step().")
        if self.done:
            raise RuntimeError("Episode finished. Call reset() to start a new one.")

        return self.control_step(action, inner_steps=1)

    def control_step(self, action: Sequence[float], inner_steps: int) -> Tuple[Tuple[float, ...], float, bool, dict]:
        """Perform multiple internal integration steps with one action.

        Only converts export data to Python list once at the end (or upon early termination).
        Returns observation tuple (snapshot after last inner step), reward (currently 0.0), done flag and info dict.
        """
        if not self.initialized:
            raise RuntimeError("Call reset() before stepping.")
        if self.done:
            raise RuntimeError("Episode finished. Call reset() to start a new one.")

        # Write action directly into ctypes import array (also keep import_vars list in sync if needed)
        self._write_action(action)  # populates self.import_vars and self.import_np

        solver = self.solver
        reward = 0.0
        code = 0
        t_current = self.t_current
        t_step = self.t_step
        t_stop = self.t_stop
        integrate = solver.integrate_io_inplace
        import_c = self._import_c_array
        export_c = self._export_c_array
        n_export = self.n_export

        for _ in range(inner_steps):
            code = integrate(t_current, import_c, export_c, n_export)
            t_current += t_step
            if code != 0 or (t_stop > 0.0 and t_current >= t_stop):
                break

        self.t_current = t_current
        info = {"return_code": code}
        if code != 0:
            # Try classify error vs model-requested stop
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
            solver.terminate_run(self.t_current)
        # Update export_vars snapshot only once here
        exports_snapshot = self.export_np.tolist()
        self.export_vars = exports_snapshot
        return tuple(exports_snapshot), reward, self.done, info

        # info = {"return_code": code}
        # reward = 0.0
        # if code != 0:
        #     info["error"] = "Solver reported a non-zero return code."
        #     self.done = True
        # if (self.config["t_stop"]>0) and (self.t_current >= self.config["t_stop"]):
        #     self.done = True
        # if self.done:
        #     self.solver.terminate_run(self.t_current)
        # return tuple(self.export_vars), reward, self.done, info

    def close(self) -> None:
        if self.initialized and not self.done:
            self.solver.terminate_run(self.t_current)
        self.initialized = False
        self.done = True

    def _write_action(self, action: Iterable[float]) -> None:
        """Map an arbitrary-length action to the solver import array.

        - If action has fewer than n_import elements: remaining inputs are zero.
        - If action has more than n_import elements: extra elements are ignored.
        - No special semantics (e.g., sign flips) are applied here to keep it generic.
        """
        # values = list(action)
        values = np.asarray(action, dtype=np.float64)
        if values.ndim == 0:
            # Broadcast scalar action to first import channel
            values = np.full(1, float(values))

        limit = min(values.size, self.n_import)
        if limit:
            np.copyto(self.import_np[:limit], values[:limit], casting="unsafe")
        if limit < self.n_import:
            self.import_np[limit:self.n_import] = 0.0

        if len(self.import_vars) != self.n_import:
            self.import_vars = [0.0] * self.n_import
        if limit:
            self.import_vars[:limit] = values[:limit].tolist()
        if limit < self.n_import:
            self.import_vars[limit:self.n_import] = [0.0] * (self.n_import - limit)

    def observation(self) -> Tuple[float, ...]:
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


if __name__ == "__main__":
    import time
    from simple_controller import SimplePathFollower

    example_sim = r"D:\ZDY_Drift\pyDrift\simfile.sim"
    # 每个 episode 新建一个独立的环境实例，避免内部状态残留
    num_episodes = 1
    max_steps_per_episode = 20000  # 控制器步（不是内部积分步）上限
    control_dt_s = 0.01  # 控制器更新周期：100ms
    total_start = time.perf_counter()

    # 1. 控制器在 episode 循环外创建，模拟持续存在的 agent
    controller = SimplePathFollower()

    for ep in range(num_episodes):
        with CarSimEnv(example_sim) as env:
            obs = env.reset()
            # 2. 每个 episode 开始时重置控制器状态
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

            target_speed = 50  # 设定目标车速 (km/h)

            while not done and steps < max_steps_per_episode:
                # 从观测值中获取控制器所需的状态
                lateral_error = obs[4]
                current_speed = obs[5]

                # 控制器计算动作
                action = controller.control(current_speed, target_speed, lateral_error, dt=control_dt_s)
                # print(f"Step {steps+1}: Action={action}")
                # 环境执行动作
                obs, reward, done, info = env.control_step(action, inner_steps_per_control)
                # print(f"Step {steps+1}: Obs (first {5})={obs[4:]}, Reward={reward}, Done={done}, Info={info}")
                steps += 1

            wall = time.perf_counter() - start_wall
            sim_advanced = env.t_current - start_sim_t
            print(
                f"Episode {ep+1} finished: steps={steps}, sim_time={sim_advanced:.6f}s, wall={wall:.4f}s, done={done}, info={info}"
            )

    total_wall = time.perf_counter() - total_start
    print(f"All {num_episodes} episodes total wall time: {total_wall:.4f}s")
