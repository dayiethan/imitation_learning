import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import entropy
import ot  # Optimal Transport library
import csv
import os
from typing import List, Tuple, Union

# 设置随机种子确保结果可复现
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# 定义模仿学习神经网络模型
class ImitationNet(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, output_size=2):
        super(ImitationNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def calculate_kl_divergence(p_data: List[np.ndarray], q_data: List[np.ndarray]) -> float:
    """计算两组轨迹分布之间的KL散度"""
    def process_data(data: List[np.ndarray]) -> np.ndarray:
        """将轨迹列表展平为点集"""
        return np.concatenate([np.array(traj) for traj in data])
    
    p_points = process_data(p_data)
    q_points = process_data(q_data)

    # 添加小扰动确保数值稳定性
    epsilon = 1e-6
    p_points += np.random.normal(0, epsilon, p_points.shape)
    q_points += np.random.normal(0, epsilon, q_points.shape)

    # 计算均值和协方差矩阵
    mu_p = np.mean(p_points, axis=0)
    mu_q = np.mean(q_points, axis=0)
    sigma_p = np.cov(p_points, rowvar=False) + epsilon * np.eye(p_points.shape[1])
    sigma_q = np.cov(q_points, rowvar=False) + epsilon * np.eye(q_points.shape[1])

    # 计算KL散度
    k = mu_p.shape[0]
    sigma_q_inv = np.linalg.inv(sigma_q)
    tr_term = np.trace(sigma_q_inv @ sigma_p)
    delta = mu_p - mu_q
    quadratic_term = delta.T @ sigma_q_inv @ delta
    logdet_term = np.log(np.linalg.det(sigma_q) / np.linalg.det(sigma_p))

    kl = 0.5 * (tr_term + quadratic_term - k + logdet_term)
    return kl

def calculate_mse(expert_trajectory: np.ndarray, generated_trajectory: np.ndarray) -> float:
    """计算两条轨迹之间的均方误差"""
    # 确保轨迹长度一致
    min_len = min(len(expert_trajectory), len(generated_trajectory))
    return np.mean((expert_trajectory[:min_len] - generated_trajectory[:min_len]) ** 2)

# 离散Frechet距离计算（用于EMD计算）
def discrete_frechet(curve1: List[Tuple[float, float]], curve2: List[Tuple[float, float]]) -> float:
    """计算两条轨迹之间的离散Frechet距离"""
    n, m = len(curve1), len(curve2)
    if n == 0 or m == 0:
        return float('inf')
    
    ca = np.zeros((n, m))
    
    for i in range(n):
        for j in range(m):
            d = np.linalg.norm(np.array(curve1[i]) - np.array(curve2[j]))
            if i == 0 and j == 0:
                ca[i, j] = d
            elif i > 0 and j == 0:
                ca[i, j] = max(ca[i-1, j], d)
            elif i == 0 and j > 0:
                ca[i, j] = max(ca[i, j-1], d)
            elif i > 0 and j > 0:
                ca[i, j] = max(min(ca[i-1, j], ca[i-1, j-1], ca[i, j-1]), d)
    
    return ca[n-1, m-1]

def calculate_emd(expert_trajectories: List[np.ndarray], generated_trajectories: List[np.ndarray]) -> float:
    """计算两组轨迹之间的地球移动距离(EMD)"""
    n_expert = len(expert_trajectories)
    n_gen = len(generated_trajectories)
    
    if n_expert == 0 or n_gen == 0:
        return float('inf')
    
    # 构建距离矩阵
    D = np.zeros((n_expert, n_gen))
    for i in range(n_expert):
        for j in range(n_gen):
            D[i, j] = discrete_frechet(expert_trajectories[i], generated_trajectories[j])
    
    # 添加异常处理，防止出现NaN或Inf
    if np.isnan(D).any() or np.isinf(D).any():
        print("Warning: Distance matrix contains NaN or Inf values, replacing with finite values")
        D = np.nan_to_num(D)
    
    # 计算EMD
    w_expert = np.ones(n_expert) / n_expert
    w_gen = np.ones(n_gen) / n_gen
    return ot.emd2(w_expert, w_gen, D)

# 生成单条轨迹的辅助函数
def generate_trajectory(
    model: ImitationNet, 
    initial_point: np.ndarray, 
    final_point: np.ndarray, 
    points_per_trajectory: int
) -> np.ndarray:
    """使用训练好的模型生成一条完整轨迹"""
    with torch.no_grad():
        state = np.hstack([initial_point, final_point])
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        trajectory = [initial_point]

        for _ in range(points_per_trajectory - 1):
            next_state = model(state).numpy().squeeze()
            trajectory.append(next_state)
            state = torch.tensor(np.hstack([next_state, final_point]), dtype=torch.float32).unsqueeze(0)
    
    return np.array(trajectory)

# 生成多条轨迹的函数
def generate_multiple_trajectories(
    model: ImitationNet, 
    initial_point: np.ndarray, 
    final_point: np.ndarray, 
    points_per_trajectory: int,
    num_trajectories: int = 100
) -> List[np.ndarray]:
    """生成多条轨迹用于评估"""
    return [
        generate_trajectory(model, initial_point, final_point, points_per_trajectory)
        for _ in range(num_trajectories)
    ]


# ---------------------- 主程序 ----------------------
def main():
    # 定义环境参数
    initial_point_up = np.array([0.0, 0.0])
    final_point_up = np.array([20.0, 0.0])
    initial_point_down = np.array([20.0, 0.0])
    final_point_down = np.array([0.0, 0.0])
    obstacle = (10, 0, 4.0)  # 中心障碍物: (x, y, radius)
    
    # 尝试读取专家数据
    try:
        with open('/mnt/data1/chendazhong/imitation_learning/data/single_uni_full_traj_up.csv', 'r') as file:
            reader = csv.reader(file)
            all_up_points = [tuple(map(float, row[2:4])) for row in reader]

        with open('/mnt/data1/chendazhong/imitation_learning/data/single_uni_full_traj_down.csv', 'r') as file:
            reader = csv.reader(file)
            all_down_points = [tuple(map(float, row[2:4])) for row in reader]
            all_down_points = list(reversed(all_down_points))
        
        print(f"Upward trajectory {len(all_up_points)} points, Downward trajectory {len(all_down_points)} points")

    # 数据准备
    num_trajectories = 1000
    points_per_trajectory = 100

    expert_data_up = [
        all_up_points[i * points_per_trajectory:(i + 1) * points_per_trajectory]
        for i in range(num_trajectories)
    ]

    expert_data_down = [
        all_down_points[i * points_per_trajectory:(i + 1) * points_per_trajectory]
        for i in range(num_trajectories)
    ]

    # 准备训练数据
    def prepare_training_data(expert_data: List[List[Tuple[float, float]]], final_point: np.ndarray):
        X, Y = [], []
        for traj in expert_data:
            for i in range(len(traj) - 1):
                X.append(np.hstack([traj[i], final_point]))  # 当前状态 + 目标
                Y.append(traj[i + 1])  # 下一个状态
        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)

    X_train_up, Y_train_up = prepare_training_data(expert_data_up, final_point_up)
    X_train_down, Y_train_down = prepare_training_data(expert_data_down, final_point_down)

    # 初始化模型、损失函数和优化器
    model_up = ImitationNet(input_size=4, hidden_size=64, output_size=2)
    model_down = ImitationNet(input_size=4, hidden_size=64, output_size=2)
    
    criterion_up = nn.MSELoss()
    criterion_down = nn.MSELoss()
    
    optimizer_up = optim.Adam(model_up.parameters(), lr=0.001)
    optimizer_down = optim.Adam(model_down.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 5000
    losses_up, losses_down = [], []
    

    for epoch in range(num_epochs):
        # 训练上行模型
        optimizer_up.zero_grad()
        predictions_up = model_up(X_train_up)
        loss_up = criterion_up(predictions_up, Y_train_up)
        loss_up.backward()
        optimizer_up.step()
        losses_up.append(loss_up.item())

        # 训练下行模型
        optimizer_down.zero_grad()
        predictions_down = model_down(X_train_down)
        loss_down = criterion_down(predictions_down, Y_train_down)
        loss_down.backward()
        optimizer_down.step()
        losses_down.append(loss_down.item())

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss Up: {loss_up.item():.6f}, Loss Down: {loss_down.item():.6f}')

    

    num_eval_trajectories = 100  # 用于评估的轨迹数量
    
    generated_trajectories_up = generate_multiple_trajectories(
        model_up, initial_point_up, final_point_up, points_per_trajectory, num_eval_trajectories
    )
    
    generated_trajectories_down = generate_multiple_trajectories(
        model_down, initial_point_down, final_point_down, points_per_trajectory, num_eval_trajectories
    )

    # 计算评估指标

    kl_div_up = calculate_kl_divergence(expert_data_up, generated_trajectories_up)
    kl_div_down = calculate_kl_divergence(expert_data_down, generated_trajectories_down)

    # 计算第一条生成轨迹的MSE
    mse_up = calculate_mse(expert_data_up[0], generated_trajectories_up[0])
    mse_down = calculate_mse(expert_data_down[0], generated_trajectories_down[0])

    # 计算EMD
    emd_value_up = calculate_emd(expert_data_up[:num_eval_trajectories], generated_trajectories_up)
    emd_value_down = calculate_emd(expert_data_down[:num_eval_trajectories], generated_trajectories_down)

    # 打印评估结果
    print("\n===== Evaluation Results =====")
    print(f"Upward Trajectory - KL Divergence: {kl_div_up:.4f}, MSE: {mse_up:.4f}, EMD: {emd_value_up:.4f}")
    print(f"Downward Trajectory - KL Divergence: {kl_div_down:.4f}, MSE: {mse_down:.4f}, EMD: {emd_value_down:.4f}")

    
    # 1. 绘制轨迹对比图
    plt.figure(figsize=(15, 10))
    
    # 绘制部分专家轨迹作为参考
    for i, traj in enumerate(expert_data_up[:5]):
        x = [point[0] for point in traj]
        y = [point[1] for point in traj]
        plt.plot(x, y, 'b--', alpha=0.3, label='专家上行轨迹' if i == 0 else "")

    for i, traj in enumerate(expert_data_down[:5]):
        x = [point[0] for point in traj]
        y = [point[1] for point in traj]
        plt.plot(x, y, 'g--', alpha=0.3, label='专家下行轨迹' if i == 0 else "")

    # 绘制生成的轨迹
    for i, traj in enumerate(generated_trajectories_up[:10]):
        plt.plot(traj[:, 0], traj[:, 1], 'r-', alpha=0.7, label='生成上行轨迹' if i == 0 else "")

    for i, traj in enumerate(generated_trajectories_down[:10]):
        plt.plot(traj[:, 0], traj[:, 1], 'y-', alpha=0.7, label='生成下行轨迹' if i == 0 else "")

    # 绘制障碍物
    ox, oy, r = obstacle
    circle = plt.Circle((ox, oy), r, color='gray', alpha=0.3)
    plt.gca().add_patch(circle)

    # 标记起点和终点
    plt.scatter(initial_point_up[0], initial_point_up[1], c='green', s=100, label='起点(上行)')
    plt.scatter(final_point_up[0], final_point_up[1], c='red', s=100, label='终点(上行)')
    plt.scatter(initial_point_down[0], initial_point_down[1], c='purple', s=100, label='起点(下行)')
    plt.scatter(final_point_down[0], final_point_down[1], c='orange', s=100, label='终点(下行)')

    plt.legend(loc='upper right')
    plt.title(f'模仿学习轨迹对比 (上行EMD: {emd_value_up:.2f}, 下行EMD: {emd_value_down:.2f})')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.grid(True)
    plt.axis('equal')  # 确保X和Y轴比例相同
    save_figure('/mnt/data1/chendazhong/imitation_learning/figures/two_agent/single_mode/TASM_with_emd.png')
    
    # 2. 绘制训练损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(losses_up, label='上行模型损失')
    plt.plot(losses_down, label='下行模型损失')
    plt.title('训练损失曲线')
    plt.xlabel('迭代轮次')
    plt.ylabel('损失值')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')  # 使用对数刻度更清晰地查看损失下降
    save_figure('/mnt/data1/chendazhong/imitation_learning/figures/two_agent_single_mode/loss_graph.png')
    
    print("所有图表已生成完成!")

if __name__ == "__main__":
    main()