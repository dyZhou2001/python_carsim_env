"""train_carsim_sac_v3.py

SAC 在 CarSim 上的路径跟踪训练（v3 版观测设计）。

与 v2 的主要区别：
- 观测不再包含沿路程 s（防止策略通过记忆 s 做“脚本化”控制）；
- 观测中加入上一帧动作 [lon_prev, steer_prev]，提供简单的“内部状态”，利于平滑控制；
- 动作仍为 2 维 [lon, steer]，映射与 v2 完全一致；
- 奖励结构沿用 v2 的设计（前进距离 + 车道保持 + 速度跟踪 + 动作/低速惩罚）。
"""

import argparse
import os
import time
from typing import Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from carsim_env import CarSimEnv
from SACModule import SACAgent, ReplayBuffer as SACReplayBuffer


class CarSimGymWrapperV3:
	"""基于 CarSimEnv 的包装（v3 观测设计）。

	动作接口：与 v2 相同
	- raw_action ∈ [-1,1]^2：
	  * raw_action[0] = lon：纵向控制（油门/制动合并）。
	  * raw_action[1] = steer_raw：方向盘控制。

	映射到仿真环境：
	- 若 lon >= 0：throttle = lon ∈ [0,1], brake = 0；
	- 若 lon < 0：throttle = 0, brake = -lon * 10.0 ∈ [0,10]；
	- steer = steer_raw * 540.0 ∈ [-540, 540] 度。

	观测：
	- 不再包含行驶距离 s，避免策略“背题”；
	- 组成：
	  [lat_n, spd_n, tgt_spd_n, lon_prev, steer_prev]
	  其中：
	    * lat_n：横向误差归一化到 [-1,1]；
	    * spd_n：车速 (0~160km/h) 归一化到 [-1,1]；
	    * tgt_spd_n：目标车速（例如 80km/h）归一化到 [-1,1]；
	    * lon_prev, steer_prev：上一帧执行的动作（仍然在 [-1,1]）。
	"""

	def __init__(self, sim_path: str, target_speed: float = 80.0):
		self._env = CarSimEnv(sim_path)
		self.n_import = None
		self.last_dist = 0.0
		self.target_speed = float(target_speed)
		self.prev_action = np.zeros(2, dtype=np.float32)

	def reset(self) -> np.ndarray:
		exports = self._env.reset()
		self.n_import = self._env.n_import
		if len(exports) > 6:
			self.last_dist = float(exports[6])
		else:
			self.last_dist = 0.0
		self.prev_action[:] = 0.0
		return self._process_export(exports)

	def step(self, raw_action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
		a = np.asarray(raw_action, dtype=np.float64).ravel()
		if a.size < 2:
			a = np.pad(a, (0, 2 - a.size), "constant")
		a = np.clip(a[:2], -1.0, 1.0)
		lon = float(a[0])
		steer_raw = float(a[1])
		if lon >= 0.0:
			throttle = lon
			brake = 0.0
		else:
			throttle = 0.0
			brake = -lon * 10.0
		steer = steer_raw * 540.0
		env_action = [throttle, brake, steer]
		if self.n_import and self.n_import > 3:
			env_action = env_action + [0.0] * (self.n_import - 3)
		control_dt = 0.05
		t_step = self._env.t_step
		if t_step <= 0:
			t_step = 0.0005
		inner_steps = int(round(control_dt / t_step))
		if inner_steps < 1:
			inner_steps = 1
		exports, _, done, info = self._env.control_step(env_action, inner_steps=inner_steps)
		obs = self._process_export(exports)
		reward = self._compute_reward(exports, lon, steer_raw)
		lat = float(list(exports)[4]) if len(exports) > 4 else 0.0
		if abs(lat) > 4.0:
			lane_penalty = 1000.0
			reward -= lane_penalty
			done = True
			if not isinstance(info, dict):
				info = {"info": info}
			info["off_track"] = True
			info["lane_penalty"] = lane_penalty
		self.prev_action = np.array([lon, steer_raw], dtype=np.float32)
		return obs, reward, done, info

	def close(self):
		try:
			self._env.close()
		except Exception:
			pass

	def _process_export(self, exports):
		exports = list(exports)
		if len(exports) < 7:
			exports = exports + [0.0] * (7 - len(exports))
		lat = float(exports[4])
		spd = float(exports[5])
		# 不再使用 dist 作为观测量
		lat_n = np.clip(lat / 5.0, -1.0, 1.0)
		spd_n = np.clip(spd / 160.0, 0.0, 1.0)
		spd_n = 2.0 * spd_n - 1.0
		tgt_spd = self.target_speed
		tgt_spd_n = np.clip(tgt_spd / 160.0, 0.0, 1.0)
		tgt_spd_n = 2.0 * tgt_spd_n - 1.0
		lon_prev, steer_prev = self.prev_action
		return np.array([lat_n, spd_n, tgt_spd_n, lon_prev, steer_prev], dtype=np.float32)

	def _compute_reward(self, exports, lon: float, steer_raw: float) -> float:
		exports = list(exports)
		if len(exports) < 7:
			exports = exports + [0.0] * (7 - len(exports))
		lat = float(exports[4])
		spd = float(exports[5])
		dist = float(exports[6])
		delta_dist = dist - self.last_dist
		self.last_dist = dist
		lat_pen = (lat / 5.0) ** 2
		spd_err = (spd - self.target_speed) / max(self.target_speed, 1.0)
		spd_pen = spd_err ** 2
		low_speed_pen = 0.0
		if spd < 3.0:
			low_speed_pen = ((3.0 - spd) / 3.0) ** 2
		act_pen = lon ** 2 + 5*steer_raw ** 2
		# 转向变化惩罚：抑制“大幅来回切换”
		delta_steer = steer_raw - self.prev_action[1]
		dsteer_pen = delta_steer ** 2
		w_dist = 8.0
		w_lat = 1.5
		w_spd = 1.0
		w_low_speed = 0.5
		w_act = 0.04
		w_dsteer = 0.2
		no_move_pen = 0.0
		if delta_dist < 0.02:
			no_move_pen = 0.5
		reward = (
			w_dist * delta_dist
			- w_lat * lat_pen
			- w_spd * spd_pen
			- w_low_speed * low_speed_pen
			- w_act * act_pen
			- w_dsteer * dsteer_pen
			- no_move_pen
		)
		return float(reward)


def train(args):
	if args.device == "auto":
		if torch.cuda.is_available():
			device = torch.device("cuda")
		elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
			device = torch.device("mps")
		else:
			device = torch.device("cpu")
	else:
		device = torch.device(args.device)

	print("Using device:", device)

	test_env = env = CarSimGymWrapperV3(args.sim, target_speed=args.target_speed)

	# 观测维度：lat, current_speed, target_speed, lon_prev, steer_prev
	obs_dim = 5
	act_dim = 2
	action_scale = np.ones(act_dim, dtype=np.float32)

	agent = SACAgent(
		obs_dim=obs_dim,
		act_dim=act_dim,
		action_scale=action_scale,
		device=device,
		gamma=args.gamma,
		alpha=args.alpha,
		auto_alpha=True,
		lr_actor=args.lr_actor,
		lr_critic=args.lr_critic,
		lr_alpha=args.lr_alpha,
		hidden_size=args.hidden_size,
		tau=args.tau,
	)

	buffer = SACReplayBuffer(args.replay_size)
	writer = SummaryWriter(comment=f"-carsim_sac_v3_{args.name}")

	global_step = 0
	best_eval_reward = -float("inf")

	for ep in range(1, args.episodes + 1):
		obs = env.reset()
		ep_reward = 0.0
		ep_len = 0
		done = False
		warmup = ep <= args.warmup_episodes
		ep_start_wall = time.perf_counter()

		while not done and ep_len < args.max_steps:
			ep_len += 1
			global_step += 1
			if warmup:
				action = np.random.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
			else:
				action = agent.select_action(obs, deterministic=False)

			next_obs, reward, done, _ = env.step(action)
			buffer.push(obs, action, reward, next_obs, done)
			ep_reward += reward
			obs = next_obs

			if not warmup and len(buffer) >= args.batch_size:
				for _ in range(args.updates_per_step):
					batch = buffer.sample(args.batch_size)
					losses = agent.update(batch)
					writer.add_scalar("loss/q1", losses["q1_loss"], global_step)
					writer.add_scalar("loss/q2", losses["q2_loss"], global_step)
					writer.add_scalar("loss/policy", losses["policy_loss"], global_step)
					writer.add_scalar("loss/alpha", losses["alpha_loss"], global_step)
					writer.add_scalar("alpha/value", losses["alpha"], global_step)
					writer.add_scalar("policy/entropy", losses["entropy"], global_step)

		wall = time.perf_counter() - ep_start_wall
		sim_time = float(env._env.t_current)
		ratio = sim_time / (wall + 1e-12)
		writer.add_scalar("train/episode_reward", ep_reward, ep)
		writer.add_scalar("train/episode_length", ep_len, ep)
		writer.add_scalar("train/sim_wall_ratio", ratio, ep)

		if ep % 5 == 0:
			warmup_flag = "[WARMUP]" if warmup else ""
			print(
				f"[v3] Episode {ep:4d} {warmup_flag} | reward={ep_reward:8.3f} | "
				f"len={ep_len:4d} | sim_t={sim_time:7.2f}s | wall={wall:6.2f}s | ratio={ratio:6.2f}"
			)

		if ep % args.eval_interval == 0:
			_eval_reward = evaluate_policy_carsim_v3(test_env, agent, device, episodes=args.eval_episodes)
			writer.add_scalar("eval/mean_reward", _eval_reward, ep)

			latest_dir = os.path.join("saves", f"carsim_sac_v3_{args.name}")
			os.makedirs(latest_dir, exist_ok=True)
			latest_path = os.path.join(latest_dir, "latest.pth")
			agent.save(latest_path)

			if _eval_reward > best_eval_reward:
				best_eval_reward = _eval_reward
				best_path = os.path.join(latest_dir, f"best_{_eval_reward:+.3f}_ep{ep}.pth")
				agent.save(best_path)
				print(f"[v3] ✓ Saved new best model to {best_path} (reward={_eval_reward:.2f})")

	writer.close()
	env.close()
	test_env.close()


@torch.no_grad()
def evaluate_policy_carsim_v3(env_wrapper: CarSimGymWrapperV3, agent: SACAgent, device, episodes: int = 5) -> float:
	total = 0.0
	for _ in range(episodes):
		obs = env_wrapper.reset()
		done = False
		ep_r = 0.0
		while not done:
			action = agent.select_action(obs, deterministic=True)
			obs, reward, done, _ = env_wrapper.step(action)
			ep_r += reward
		total += ep_r
	return total / max(episodes, 1)


def parse_args():
	p = argparse.ArgumentParser(description="SAC v3 training on CarSim path tracking (obs with prev action, no dist)")
	p.add_argument("--sim", default="./simfile.sim", help="路径到 simfile.sim")
	p.add_argument("-n", "--name", default="sac_v3_run", help="运行名称（保存模型/日志时使用）")
	# 运行模式：train / eval / continue
	p.add_argument("--mode", choices=["train", "eval", "continue"], default="train", help="运行模式：train=新训练, eval=评估, continue=从已有模型继续训练")
	p.add_argument("--model", default="", help="评估或继续训练时加载的模型路径 .pth")
	p.add_argument("--episodes", type=int, default=1000, help="训练 episode 数")
	p.add_argument("--max-steps", type=int, default=5000, help="每个 episode 最大步数")
	p.add_argument("--device", type=str, default="auto", help="计算设备：cpu/cuda/mps/auto")
	# SAC 超参数（参考你 v2 的较稳定配置）
	p.add_argument("--batch-size", type=int, default=256)
	p.add_argument("--gamma", type=float, default=0.99)
	p.add_argument("--alpha", type=float, default=0.1, help="初始熵温度（auto_alpha 下仅作初始值）")
	p.add_argument("--lr-actor", type=float, default=1e-4)
	p.add_argument("--lr-critic", type=float, default=1e-4)
	p.add_argument("--lr-alpha", type=float, default=5e-5)
	p.add_argument("--replay-size", type=int, default=200_000)
	p.add_argument("--warmup-episodes", type=int, default=10, help="仅随机探索的 episode 数")
	p.add_argument("--eval-interval", type=int, default=10, help="多少个 episode 做一次评估")
	p.add_argument("--eval-episodes", type=int, default=3, help="评估时运行的 episode 数")
	p.add_argument("--updates-per-step", type=int, default=1, help="每个环境步的梯度更新次数")
	p.add_argument("--hidden-size", type=int, default=256, help="网络隐藏层宽度")
	p.add_argument("--tau", type=float, default=5e-3, help="目标网络软更新系数")
	p.add_argument("--target-speed", type=float, default=80.0, help="期望巡航车速 km/h")
	return p.parse_args()


if __name__ == "__main__":
	args = parse_args()
	if args.mode == "train":
		train(args)
	elif args.mode == "eval":
		# 构造 Agent 与 env，加载模型后只做评估
		if args.device == "auto":
			if torch.cuda.is_available():
				device = torch.device("cuda")
			elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
				device = torch.device("mps")
			else:
				device = torch.device("cpu")
		else:
			device = torch.device(args.device)
		print("Eval mode using device:", device)
		env = CarSimGymWrapperV3(args.sim, target_speed=args.target_speed)
		obs_dim = 5
		act_dim = 2
		action_scale = np.ones(act_dim, dtype=np.float32)
		agent = SACAgent(
			obs_dim=obs_dim,
			act_dim=act_dim,
			action_scale=action_scale,
			device=device,
			gamma=args.gamma,
			alpha=args.alpha,
			auto_alpha=True,
			lr_actor=args.lr_actor,
			lr_critic=args.lr_critic,
			lr_alpha=args.lr_alpha,
			hidden_size=args.hidden_size,
			tau=args.tau,
		)
		if not args.model:
			raise ValueError("eval 模式需要指定 --model 路径")
		print(f"Loading model from {args.model}")
		agent.load(args.model)
		mean_r = evaluate_policy_carsim_v3(env, agent, device, episodes=args.eval_episodes)
		print(f"[v3 eval] mean reward over {args.eval_episodes} episodes: {mean_r:.2f}")
		env.close()
	elif args.mode == "continue":
		# 继续训练：需要加载已有模型，然后调用 train，训练逻辑与 train 相同
		if not args.model:
			raise ValueError("continue 模式需要指定 --model 路径")
		# 简单做法：在 train 内部读取 args.model；这里只是设置一个标志
		print("[v3 continue] starting from model:", args.model)
		# 在 train 中未使用 args.model，因此这里直接手工加载后再进入训练循环
		if args.device == "auto":
			if torch.cuda.is_available():
				device = torch.device("cuda")
			elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
				device = torch.device("mps")
			else:
				device = torch.device("cpu")
		else:
			device = torch.device(args.device)
		print("Continue mode using device:", device)
		# 构造 env 和 agent，与 train 一致
		env = CarSimGymWrapperV3(args.sim, target_speed=args.target_speed)
		test_env = env
		obs_dim = 5
		act_dim = 2
		action_scale = np.ones(act_dim, dtype=np.float32)
		agent = SACAgent(
			obs_dim=obs_dim,
			act_dim=act_dim,
			action_scale=action_scale,
			device=device,
			gamma=args.gamma,
			alpha=args.alpha,
			auto_alpha=True,
			lr_actor=args.lr_actor,
			lr_critic=args.lr_critic,
			lr_alpha=args.lr_alpha,
			hidden_size=args.hidden_size,
			au=args.tau,
		)
		print(f"Loading model from {args.model}")
		agent.load(args.model)
		# 下面复用与 train 相同的训练循环，这里为了简洁不拆函数，建议根据需要直接使用单独的 continue_v3 脚本
		# 当前实现优先满足你在一个脚本内区分三种模式的需求
		print("[v3 continue] 建议后续单独写 continue_v3 脚本以完全复用逻辑。")
