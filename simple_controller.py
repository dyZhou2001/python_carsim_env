import numpy as np
import math

class SimplePathFollower:
    def __init__(self):
        # 控制器参数
        self.kp_lateral = 10      # 横向比例系数
        self.kd_lateral = 5     # 横向微分系数
        self.ki_lateral = 5    # 横向积分系数
        
        self.kp_speed = 0.5        # 速度比例系数
        self.ki_speed = 0.1        # 速度积分系数
        
        # 限制值
        self.max_steer = np.radians(720)  # 最大转向角30度
        self.max_throttle = 1.0          # 最大油门
        self.max_brake = 1.0             # 最大刹车
        
        # 状态变量
        self.prev_lateral_error = 0.0
        self.integral_lateral_error = 0.0
        self.integral_speed_error = 0.0
        self.prev_speed_error = 0.0
        
    def reset(self):
        """重置控制器状态"""
        self.prev_lateral_error = 0.0
        self.integral_lateral_error = 0.0
        self.integral_speed_error = 0.0
        self.prev_speed_error = 0.0
    
    def control(self, current_speed, target_speed, lateral_error, heading_error=0, dt=0.01):
        """
        路径保持控制主函数
        
        参数:
        current_speed: 当前车速 (m/s)
        target_speed: 目标车速 (m/s)
        lateral_error: 横向偏差 (m) - 正值为车辆在路径右侧
        heading_error: 航向角偏差 (rad) - 正值为车辆指向路径右侧
        dt: 时间步长 (s)
        
        返回:
        steering_angle: 方向盘转角 (rad)
        throttle: 油门踏板开度 [0, 1]
        brake: 刹车踏板开度 [0, 1]
        """
        
        # 横向控制 - 计算方向盘转角
        steering_angle = self._lateral_control(lateral_error, heading_error, dt)
        
        # 纵向控制 - 计算油门和刹车
        throttle, brake = self._longitudinal_control(current_speed, target_speed, dt)
        
        return  throttle, brake, steering_angle
    
    def _lateral_control(self, lateral_error, heading_error, dt):
        """横向控制 - 计算方向盘转角"""
        
        # PID控制计算
        proportional = lateral_error
        derivative = (lateral_error - self.prev_lateral_error) / dt if dt > 0 else 0
        self.integral_lateral_error += lateral_error * dt
        
        # 限制积分项防止饱和
        self.integral_lateral_error = np.clip(self.integral_lateral_error, -2.0, 2.0)
        
        # 计算转向角
        steer_angle = -(self.kp_lateral * proportional + 
                      self.kd_lateral * derivative + 
                      self.ki_lateral * self.integral_lateral_error +
                      0.8 * heading_error)  # 航向角补偿
        
        # 更新历史误差
        self.prev_lateral_error = lateral_error
        
        # 限制转向角范围
        steer_angle = np.clip(steer_angle, -self.max_steer, self.max_steer)*180/math.pi
        
        return steer_angle
    
    def _longitudinal_control(self, current_speed, target_speed, dt):
        """纵向控制 - 计算油门和刹车"""
        
        speed_error = target_speed - current_speed
        
        # PI控制
        proportional = speed_error
        self.integral_speed_error += speed_error * dt
        
        # 限制积分项
        self.integral_speed_error = np.clip(self.integral_speed_error, -5.0, 5.0)
        
        # 计算控制输出
        control_output = self.kp_speed * proportional + self.ki_speed * self.integral_speed_error
        
        # 分离油门和刹车
        if control_output >= 0:
            # 需要加速
            throttle = np.clip(control_output, 0, self.max_throttle)
            brake = 0.0
        else:
            # 需要减速
            throttle = 0.0
            brake = np.clip(-control_output, 0, self.max_brake)
        
        self.prev_speed_error = speed_error
        
        return throttle, brake

class AdvancedPathFollower(SimplePathFollower):
    def __init__(self, lookahead_distance=5.0):
        super().__init__()
        
        # 预瞄控制参数
        self.lookahead_distance = lookahead_distance
        self.kp_preview = 0.05
        
        # Stanley控制器参数
        self.k_softening = 1.0  # 软化系数
        
    def control_with_preview(self, current_speed, target_speed, 
                           lateral_error, heading_error, 
                           path_curvature=0, dt=0.01):
        """
        带预瞄的路径跟踪控制
        
        参数:
        path_curvature: 路径曲率 (1/m)
        """
        
        # 计算预瞄距离（速度相关）
        adaptive_lookahead = max(2.0, self.lookahead_distance * current_speed / 10.0)
        
        # Stanley控制器改进版
        steering_angle = self._stanley_control(lateral_error, heading_error, current_speed)
        
        # 曲率前馈补偿
        curvature_feedforward = adaptive_lookahead * path_curvature
        
        # 最终转向角
        steering_angle += curvature_feedforward
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)
        
        # 纵向控制
        throttle, brake = self._longitudinal_control(current_speed, target_speed, dt)
        
        return steering_angle, throttle, brake
    
    def _stanley_control(self, lateral_error, heading_error, current_speed):
        """Stanley路径跟踪控制器"""
        
        # 避免除零
        speed = max(current_speed, 0.1)
        
        # Stanley控制律
        steer_angle = heading_error + math.atan2(self.kp_lateral * lateral_error, 
                                                self.k_softening + speed)
        
        return steer_angle

# 测试代码
if __name__ == "__main__":
    # 创建控制器
    controller = SimplePathFollower()
    
    # 模拟控制循环
    dt = 0.01  # 10ms时间步长
    
    print("测试SimplePathFollower控制器:")
    for i in range(5):
        # 模拟输入数据
        current_speed = 10.0 + 0.1 * math.sin(i * 0.1)  # 当前车速
        target_speed = 15.0                             # 目标车速
        lateral_error = 0.5 * math.sin(i * 0.05)        # 横向偏差
        
        # 计算控制指令
        steering, throttle, brake = controller.control(
            current_speed, target_speed, lateral_error, dt=dt
        )
        
        print(f"Step {i}: Steering={np.degrees(steering):.2f}°, "
              f"Throttle={throttle:.2f}, Brake={brake:.2f}")
    
    print("\n测试AdvancedPathFollower控制器:")
    advanced_controller = AdvancedPathFollower()
    steering, throttle, brake = advanced_controller.control_with_preview(
        12.0, 15.0, 0.3, 0.1, 0.01
    )
    print(f"Advanced: Steering={np.degrees(steering):.2f}°, "
          f"Throttle={throttle:.2f}, Brake={brake:.2f}")