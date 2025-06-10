import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial.distance import cdist
import ot  # 需安装：pip install pot
import csv

# ---------------------- 配置与种子设置 ----------------------
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# ---------------------- 神经网络定义 ----------------------
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

# ---------------------- 轨迹评估函数 ----------------------
def calculate_kl_divergence(p_data, q_data):
    """计算高斯分布的KL散度"""
    def process_data(data):
        return np.concatenate([np.array(traj) for traj in data]) if isinstance(data[0], (list, np.ndarray)) else np.array(data)
    
    p_flat = process_data(p_data)
    q_flat = process_data([q_data])  # 适配单条轨迹输入
    
    mu_p, mu_q = p_flat.mean(0), q_flat.mean(0)
    cov_p = np.cov(p_flat, rowvar=False) + 1e-8 * np.eye(2)
    cov_q = np.cov(q_flat, rowvar=False) + 1e-8 * np.eye(2)
    
    det_p, det_q = np.linalg.det(cov_p), np.linalg.det(cov_q)
    tr_term = np.trace(np.linalg.inv(cov_q) @ cov_p)
    quad_term = (mu_p - mu_q).T @ np.linalg.inv(cov_q) @ (mu_p - mu_q)
    kl = 0.5 * (tr_term + quad_term - 2 + np.log(det_q / det_p))
    return kl

def calculate_mse(expert_traj, gen_traj):
    """计算单条轨迹的MSE"""
    return np.mean((expert_traj - gen_traj) ** 2)

def discrete_frechet(curve1, curve2):
    """计算离散Frechet距离（动态规划实现）"""
    m, n = len(curve1), len(curve2)
    dp = np.full((m+1, n+1), np.inf)
    dp[0, 0] = 0
    
    for i in range(m):
        for j in range(n):
            d = np.linalg.norm(curve1[i] - curve2[j])
            dp[i+1, j+1] = d + min(dp[i, j+1], dp[i+1, j], dp[i, j])
    return dp[m, n]

def calculate_emd(expert_trajectories, generated_trajectories):
    """计算两组轨迹的EMD"""
    n_expert = len(expert_trajectories)
    n_gen = len(generated_trajectories)
    D = np.zeros((n_expert, n_gen))
    
    for i in range(n_expert):
        for j in range(n_gen):
            D[i, j] = discrete_frechet(expert_trajectories[i], generated_trajectories[j])
    
    w_expert = np.ones(n_expert) / n_expert
    w_gen = np.ones(n_gen) / n_gen
    return ot.emd2(w_expert, w_gen, D)

# ---------------------- 数据处理与训练 ----------------------
initial_point = np.array([0.0, 0.0])
final_point = np.array([20.0, 0.0])
obstacle = (10, 0, 4.0)

# 解析专家数据
with open('/mnt/data1/chendazhong/imitation_learning/data/single_uni_full_traj.csv', 'r') as file:
    reader = csv.reader(file)
    all_points = np.array([[float(row[2]), float(row[3])] for row in reader])

num_trajectories = 1000
points_per_trajectory = 100
expert_data = [
    all_points[i*points_per_trajectory : (i+1)*points_per_trajectory]
    for i in range(num_trajectories)
]

# 准备训练数据
X_train, Y_train = [], []
for traj in expert_data:
    for t in range(len(traj)-1):
        X_train.append(np.hstack([traj[t], final_point]))
        Y_train.append(traj[t+1])

X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
Y_train = torch.tensor(np.array(Y_train), dtype=torch.float32)

# 训练模型
model = ImitationNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

losses = []
for epoch in range(5000):
    pred = model(X_train)
    loss = criterion(pred, Y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch+1) % 500 == 0:
        print(f'Epoch {epoch+1}/5000, Loss: {loss.item():.4f}')

# ---------------------- 生成多条轨迹并评估 ----------------------
num_generated = 100  # 生成20条轨迹用于EMD计算
generated_trajectories = []

for _ in range(num_generated):
    with torch.no_grad():
        state = np.hstack([initial_point, final_point])
        traj = [initial_point.copy()]
        for _ in range(points_per_trajectory-1):
            next_state = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).numpy().squeeze()
            traj.append(next_state)
            state = np.hstack([next_state, final_point])
        generated_trajectories.append(np.array(traj))

# ---------------------- 评估指标计算 ----------------------
# 1. KL散度（专家全集 vs 生成单条轨迹，保留原逻辑）
kl_div_single = calculate_kl_divergence(expert_data, generated_trajectories[0])

# 2. MSE（单条对比）
mse_single = calculate_mse(expert_data[0], generated_trajectories[0])

# 3. EMD（轨迹集合对比，20×20矩阵）
emd_value = calculate_emd(expert_data[:num_generated], generated_trajectories)

print(f"Evaluation Results：")
print(f"KL Divergence: {kl_div_single:.4f}")
print(f"MSE (first trajectory): {mse_single:.4f}")
print(f"EMD: {emd_value:.4f}")

# ---------------------- 可视化 ----------------------
plt.figure(figsize=(15, 8))

# 绘制专家轨迹（前5条）
for traj in expert_data[:5]:
    plt.plot(traj[:, 0], traj[:, 1], 'b--', alpha=0.5, label='Expert' if traj is expert_data[0] else "")

# 绘制生成轨迹
for traj in generated_trajectories:
    plt.plot(traj[:, 0], traj[:, 1], 'r-', alpha=0.7, label='Generated' if traj is generated_trajectories[0] else "")

# 绘制障碍物
circle = plt.Circle(obstacle[:2], obstacle[2], color='gray', alpha=0.3)
plt.gca().add_patch(circle)
plt.scatter(*initial_point, c='green', s=100, label='Start')
plt.scatter(*final_point, c='red', s=100, label='End')

plt.legend()
plt.title(f'轨迹对比 (EMD: {emd_value:.2f})')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(-2, 22)
plt.ylim(-5, 5)
plt.grid(True)
plt.savefig('figures/single_agent/dual_mode/SADM_with_emd.png')
plt.show()

# 绘制损失曲线
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.title('训练损失曲线')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True)
plt.savefig('/mnt/data1/chendazhong/imitation_learning/figures/single_mode/loss_curve.png')
plt.show()