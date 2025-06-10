import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import csv
from scipy.stats import entropy
import ot  # Optimal Transport library
import os
from typing import List, Tuple, Union

# 设置随机种子确保结果可复现
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# 定义KL散度计算函数
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

# 生成器网络 - 用于生成轨迹
class Generator(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, output_size=2):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 判别器网络 - 用于区分专家轨迹和生成轨迹
class Discriminator(nn.Module):
    def __init__(self, input_size=6, hidden_size=64):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # 二分类 (专家或生成)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)

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
        print("Warning: Distance matrix contains NaN or Inf, replacing with finite values")
        D = np.nan_to_num(D)
    
    # 计算EMD
    w_expert = np.ones(n_expert) / n_expert
    w_gen = np.ones(n_gen) / n_gen
    return ot.emd2(w_expert, w_gen, D)

# 生成单条轨迹的辅助函数
def generate_trajectory(
    model: Generator, 
    initial_point: np.ndarray, 
    final_point: np.ndarray, 
    points_per_trajectory: int
) -> np.ndarray:
    """使用训练好的生成器生成一条完整轨迹"""
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
    model: Generator, 
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

# 保存图表的辅助函数
def save_figure(path: str, fig: plt.Figure = None, close: bool = True) -> None:
    """保存图表并添加错误处理"""
    if fig is None:
        fig = plt.gcf()
    
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            print(f"Warning: Failed to create directory: {directory}, Error: {e}")
            return
    
    try:
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {path}")
    except Exception as e:
        print(f"Error: Failed to save figure: {e}")
    
    if close:
        plt.close(fig)

# 训练函数
def train_gan(
    generator: Generator,
    discriminator: Discriminator,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    num_epochs: int = 5000,
    print_every: int = 50
) -> Tuple[List[float], List[float]]:
    """训练GAN模型"""
    optimizer_generator = optim.Adam(generator.parameters(), lr=0.001)
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=0.001)
    
    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCELoss()  # 判别器使用的二元交叉熵损失
    
    generator_losses = []
    discriminator_losses = []
    
    for epoch in range(num_epochs):
        # 训练判别器
        optimizer_discriminator.zero_grad()
        
        # 真实样本
        expert_labels = torch.ones(X_train.size(0), 1)
        real_pred = discriminator(torch.cat([X_train, Y_train], dim=1))
        loss_real = criterion_bce(real_pred, expert_labels)
        
        # 生成样本
        generated = generator(X_train)
        generated_labels = torch.zeros(X_train.size(0), 1)
        fake_pred = discriminator(torch.cat([X_train, generated.detach()], dim=1))
        loss_fake = criterion_bce(fake_pred, generated_labels)
        
        # 判别器总损失
        loss_discriminator = loss_real + loss_fake
        loss_discriminator.backward()
        optimizer_discriminator.step()
        
        # 训练生成器
        optimizer_generator.zero_grad()
        fake_pred = discriminator(torch.cat([X_train, generated], dim=1))
        
        # 生成器损失 = 对抗损失 + MSE损失
        loss_generator = criterion_bce(fake_pred, expert_labels) + 0.1 * criterion_mse(generated, Y_train)
        loss_generator.backward()
        optimizer_generator.step()
        
        # 记录损失
        generator_losses.append(loss_generator.item())
        discriminator_losses.append(loss_discriminator.item())
        
        # 打印训练进度
        if (epoch + 1) % print_every == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Generator Loss: {loss_generator.item():.6f}, '
                  f'Discriminator Loss: {loss_discriminator.item():.6f}')
    
    return generator_losses, discriminator_losses

# ---------------------- 主程序 ----------------------
def main():
    # 定义环境参数
    initial_point_up = np.array([0.0, 0.0])
    final_point_up = np.array([20.0, 0.0])
    initial_point_down = np.array([20.0, 0.0])
    final_point_down = np.array([0.0, 0.0])
    obstacle = (10, 0, 4.0)  # 中心障碍物: (x, y, radius)
    
    # 尝试读取专家数据
    with open('/mnt/data1/chendazhong/imitation_learning/data/single_uni_full_traj_up.csv', 'r') as file:
        reader = csv.reader(file)
        all_up_points = []
        for row in reader:
            x, y = float(row[2]), float(row[3])
            all_up_points.append((x, y))

    with open('/mnt/data1/chendazhong/imitation_learning/data/single_uni_full_traj_down.csv', 'r') as file:
        reader = csv.reader(file)
        all_down_points = []
        for row in reader:
            x, y = float(row[2]), float(row[3])
            all_down_points.append((x, y))
        all_down_points = list(reversed(all_down_points))
        
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

    # 初始化生成器和判别器
    generator_up = Generator(input_size=4, hidden_size=64, output_size=2)
    discriminator_up = Discriminator(input_size=6, hidden_size=64)

    generator_down = Generator(input_size=4, hidden_size=64, output_size=2)
    discriminator_down = Discriminator(input_size=6, hidden_size=64)

    # 训练模型
    print("Starting training for upward trajectory model...")
    losses_generator_up, losses_discriminator_up = train_gan(
        generator_up, discriminator_up, X_train_up, Y_train_up, num_epochs=5000
    )
    
    print("\nStarting training for downward trajectory model...")
    losses_generator_down, losses_discriminator_down = train_gan(
        generator_down, discriminator_down, X_train_down, Y_train_down, num_epochs=5000
    )

    
    # 生成多条轨迹用于评估
    print("Generate trajectories for evaluation...")
    num_eval_trajectories = 100  # 用于评估的轨迹数量
    
    generated_trajectories_up = generate_multiple_trajectories(
        generator_up, initial_point_up, final_point_up, points_per_trajectory, num_eval_trajectories
    )
    
    generated_trajectories_down = generate_multiple_trajectories(
        generator_down, initial_point_down, final_point_down, points_per_trajectory, num_eval_trajectories
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
    print("\n===== Evaluation Result =====")
    print(f"Upward trajectory - KL Divergence: {kl_div_up:.4f}, MSE: {mse_up:.4f}, EMD: {emd_value_up:.4f}")
    print(f"Downward trajectory - KL Divergence: {kl_div_down:.4f}, MSE: {mse_down:.4f}, EMD: {emd_value_down:.4f}")


    
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
    plt.title(f'生成对抗模仿学习轨迹对比 (上行EMD: {emd_value_up:.2f}, 下行EMD: {emd_value_down:.2f})')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.grid(True)
    plt.axis('equal')  # 确保X和Y轴比例相同
    save_figure('/mnt/data1/chendazhong/imitation_learning/figures/gan_based/TASM_GAN_with_emd.png')
    
    # 2. 绘制训练损失曲线
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(losses_generator_up, label='上行生成器损失')
    plt.plot(losses_discriminator_up, label='上行判别器损失')
    plt.title('上行模型训练损失')
    plt.xlabel('迭代轮次')
    plt.ylabel('损失值')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')  # 使用对数刻度更清晰地查看损失下降
    
    plt.subplot(1, 2, 2)
    plt.plot(losses_generator_down, label='下行生成器损失')
    plt.plot(losses_discriminator_down, label='下行判别器损失')
    plt.title('下行模型训练损失')
    plt.xlabel('迭代轮次')
    plt.ylabel('损失值')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')  # 使用对数刻度更清晰地查看损失下降
    
    plt.tight_layout()
    save_figure('/mnt/data1/chendazhong/imitation_learning/figures/gan_based/loss_graph.png')
    
    print("所有图表已生成完成!")

if __name__ == "__main__":
    main()