import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import entropy
import ot  # Optimal Transport library
import csv

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Define the Neural Network for Imitation Learning
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

def calculate_kl_divergence(p_data, q_data):
    """ Compute KL Divergence between two Gaussian distributions """
    def process_data(data):
        if isinstance(data, list) and isinstance(data[0], (list, np.ndarray)):
            # Flatten list of trajectories
            return np.concatenate([np.array(traj) for traj in data])
        return np.array(data)
    
    p_points = process_data(p_data)
    q_points = process_data(q_data)

    # Add small epsilon for numerical stability
    epsilon = 1e-6
    p_points += np.random.normal(0, epsilon, p_points.shape)  # Prevent identical points
    q_points += np.random.normal(0, epsilon, q_points.shape)

    # Calculate means and covariance matrices
    mu_p = np.mean(p_points, axis=0)
    mu_q = np.mean(q_points, axis=0)
    sigma_p = np.cov(p_points, rowvar=False) + epsilon * np.eye(p_points.shape[1])
    sigma_q = np.cov(q_points, rowvar=False) + epsilon * np.eye(q_points.shape[1])

    k = mu_p.shape[0]
    sigma_q_inv = np.linalg.inv(sigma_q)
    tr_term = np.trace(sigma_q_inv @ sigma_p)
    delta = mu_p - mu_q
    quadratic_term = delta.T @ sigma_q_inv @ delta
    logdet_term = np.log(np.linalg.det(sigma_q) / np.linalg.det(sigma_p))

    kl = 0.5 * (tr_term + quadratic_term - k + logdet_term)
    return kl

def calculate_mse(expert_trajectory, generated_trajectory):
    """ Compute Mean Squared Error between two trajectories """
    # 确保轨迹长度一致
    min_len = min(len(expert_trajectory), len(generated_trajectory))
    return np.mean((expert_trajectory[:min_len] - generated_trajectory[:min_len]) ** 2)

# Discrete Frechet Distance (used for EMD calculation)
def discrete_frechet(curve1, curve2):
    """ Calculate discrete Frechet distance (using dynamic programming) """
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

def calculate_emd(expert_trajectories, generated_trajectories):
    """ Calculate the Earth Mover's Distance (EMD) between two sets of trajectories """
    n_expert = len(expert_trajectories)
    n_gen = len(generated_trajectories)
    
    if n_expert == 0 or n_gen == 0:
        return float('inf')
    
    D = np.zeros((n_expert, n_gen))
    
    for i in range(n_expert):
        for j in range(n_gen):
            D[i, j] = discrete_frechet(expert_trajectories[i], generated_trajectories[j])
    
    w_expert = np.ones(n_expert) / n_expert
    w_gen = np.ones(n_gen) / n_gen
    
    # 添加异常处理，防止出现NaN
    if np.isnan(D).any() or np.isinf(D).any():
        print("Warning: Distance matrix contains NaN or Inf")
        D = np.nan_to_num(D)
    
    return ot.emd2(w_expert, w_gen, D)

# 生成单条轨迹的辅助函数
def generate_trajectory(model, initial_point, final_point, points_per_trajectory):
    with torch.no_grad():
        state = np.hstack([initial_point, final_point])
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        trajectory = [initial_point]

        for _ in range(points_per_trajectory - 1):
            next_state = model(state).numpy().squeeze()
            trajectory.append(next_state)
            state = torch.tensor(np.hstack([next_state, final_point]), dtype=torch.float32).unsqueeze(0)
    
    return np.array(trajectory)

# ---------------------- Data Processing and Training ----------------------
initial_point = np.array([0.0, 0.0])
final_point = np.array([20.0, 0.0])
initial_point_rev = np.array([20.0, 0.0])
final_point_rev = np.array([0.0, 0.0])
obstacle = (10, 0, 4.0)  # Single central obstacle: (x, y, radius)

# Parse expert data
try:
    with open('/mnt/data1/chendazhong/imitation_learning/data/single_uni_full_traj.csv', 'r') as file:
        reader = csv.reader(file)
        all_points = [tuple(map(float, row[2:4])) for row in reader]
except FileNotFoundError:
    print("Error: Expert data file not found. Using synthetic data for demonstration.")
    # 生成模拟数据用于演示
    all_points = []
    for i in range(10000):
        x = i / 500.0
        y = np.sin(x) * 2.0  # 简单的正弦曲线作为示例轨迹
        all_points.append((x, y))

# Data preparation
num_trajectories = 1000
points_per_trajectory = 100
expert_data = [all_points[i * points_per_trajectory:(i + 1) * points_per_trajectory] for i in range(num_trajectories)]
expert_data_rev = [list(reversed(traj)) for traj in expert_data]  # 直接反转轨迹，更高效

# Prepare Data for Training
X_train = []
Y_train = []
for traj in expert_data:
    for i in range(len(traj) - 1):
        X_train.append(np.hstack([traj[i], final_point]))  # Current state + goal
        Y_train.append(traj[i + 1])  # Next state

X_train_rev = []
Y_train_rev = []
for traj in expert_data_rev:
    for i in range(len(traj) - 1):
        X_train_rev.append(np.hstack([traj[i], final_point_rev]))  # Current state + goal
        Y_train_rev.append(traj[i + 1])  # Next state

X_train = torch.tensor(np.array(X_train), dtype=torch.float32)  # Shape: (N, 4)
Y_train = torch.tensor(np.array(Y_train), dtype=torch.float32)  # Shape: (N, 2)

X_train_rev = torch.tensor(np.array(X_train_rev), dtype=torch.float32)  # Shape: (N, 4)
Y_train_rev = torch.tensor(np.array(Y_train_rev), dtype=torch.float32)  # Shape: (N, 2)

# Initialize Models, Loss Function, and Optimizers
model = ImitationNet(input_size=4, hidden_size=64, output_size=2)
model_rev = ImitationNet(input_size=4, hidden_size=64, output_size=2)
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(list(model.parameters()) + list(model_rev.parameters()), lr=0.001)

# Train the Model
num_epochs = 5000
losses = []
for epoch in range(num_epochs):
    predictions = model(X_train)
    predictions_rev = model_rev(X_train_rev)
    loss = criterion(predictions, Y_train)
    loss_rev = criterion(predictions_rev, Y_train_rev)

    joint_loss = 0.5 * loss + 0.5 * loss_rev

    # Backpropagation and optimization
    optimizer.zero_grad()
    joint_loss.backward()
    optimizer.step()

    losses.append(joint_loss.item())
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Generate multiple trajectories for evaluation
num_generated = 100  # 与专家轨迹数量匹配

# 生成上行轨迹
generated_trajectories_up = [
    generate_trajectory(model, initial_point, final_point, points_per_trajectory)
    for _ in range(num_generated)
]

# 生成下行轨迹
generated_trajectories_down = [
    generate_trajectory(model_rev, initial_point_rev, final_point_rev, points_per_trajectory)
    for _ in range(num_generated)
]

# 计算评估指标
# 1. KL Divergence (distribution-level similarity)
kl_div_up = calculate_kl_divergence(expert_data, generated_trajectories_up)
kl_div_down = calculate_kl_divergence(expert_data_rev, generated_trajectories_down)

# 2. MSE (average point-wise error for the first generated trajectory)
mse_up = calculate_mse(expert_data[0], generated_trajectories_up[0])
mse_down = calculate_mse(expert_data_rev[0], generated_trajectories_down[0])

# 3. EMD (distribution-level similarity considering trajectory structure)
emd_value_up = calculate_emd(expert_data[:num_generated], generated_trajectories_up)
emd_value_down = calculate_emd(expert_data_rev[:num_generated], generated_trajectories_down)

print(f"Evaluation Metrics:")
print(f"Upward Trajectories - KL Divergence: {kl_div_up:.4f}, MSE: {mse_up:.4f}, EMD: {emd_value_up:.4f}")
print(f"Downward Trajectories - KL Divergence: {kl_div_down:.4f}, MSE: {mse_down:.4f}, EMD: {emd_value_down:.4f}")

# Plot the Expert and Generated Trajectories with a Single Central Obstacle
plt.figure(figsize=(20, 8))

# 绘制部分专家轨迹作为参考
for i, traj in enumerate(expert_data[:5]):
    x = [point[0] for point in traj]
    y = [point[1] for point in traj]
    plt.plot(x, y, 'b--', alpha=0.3, label='Expert' if i == 0 else "")

# 绘制生成的轨迹
for i, traj in enumerate(generated_trajectories_up[:10]):
    plt.plot(traj[:, 0], traj[:, 1], 'r-', alpha=0.7, label='Generated Up' if i == 0 else "")

for i, traj in enumerate(generated_trajectories_down[:10]):
    plt.plot(traj[:, 0], traj[:, 1], 'y-', alpha=0.7, label='Generated Down' if i == 0 else "")

# Plot the single central obstacle as a circle
ox, oy, r = obstacle
circle = plt.Circle((ox, oy), r, color='gray', alpha=0.3)
plt.gca().add_patch(circle)

# Mark start and end points
plt.scatter(initial_point[0], initial_point[1], c='green', s=100, label='Start')
plt.scatter(final_point[0], final_point[1], c='red', s=100, label='End')

plt.legend()
plt.title(f'Imitation Learning: Expert vs Generated Trajectories (EMD Up: {emd_value_up:.2f}, Down: {emd_value_down:.2f})')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.tight_layout()
try:
    plt.savefig('/mnt/data1/chendazhong/imitation_learning/figures/two_agent/dual_mode/TADM_noexpert.png')
except:
    print("Warning: Failed to save figure. Check directory permissions.")
plt.show()

# Plot the Training Loss
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
try:
    plt.savefig('/mnt/data1/chendazhong/imitation_learning/figures/two_agents_shared/loss_graph.png')
except:
    print("Warning: Failed to save figure. Check directory permissions.")
plt.show()