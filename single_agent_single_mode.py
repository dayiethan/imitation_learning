import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial.distance import cdist
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
    # Convert input data to 2D point arrays
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

    # Calculate KL divergence components
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
    return np.mean((expert_trajectory - generated_trajectory) ** 2)

def discrete_frechet(curve1, curve2):
    """
    Compute the discrete Frechet distance between two curves.
    This is the path-independent distance measure between trajectories.
    """
    n, m = len(curve1), len(curve2)
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
    """
    Calculate the Earth Mover's Distance (EMD) between two sets of trajectories.
    This measures the similarity between two distributions of trajectories.
    """
    n_expert = len(expert_trajectories)
    n_gen = len(generated_trajectories)
    
    # Compute the distance matrix using Frechet distance
    distance_matrix = np.zeros((n_expert, n_gen))
    for i in range(n_expert):
        for j in range(n_gen):
            distance_matrix[i, j] = discrete_frechet(expert_trajectories[i], generated_trajectories[j])
    
    # Define uniform weights for both distributions
    a = np.ones(n_expert) / n_expert
    b = np.ones(n_gen) / n_gen
    
    # Compute EMD using Optimal Transport
    return ot.emd2(a, b, distance_matrix)

# Define initial and final points, and a single central obstacle
initial_point = np.array([0.0, 0.0])
final_point = np.array([20.0, 0.0])
obstacle = (10, 0, 4.0)  # Single central obstacle: (x, y, radius)

# Parse expert data from single_uni_full_traj.csv
with open('/mnt/data1/chendazhong/imitation_learning/data/single_uni_full_traj_up.csv', 'r') as file:
    reader = csv.reader(file)
    all_points = []
    for row in reader:
        x, y = float(row[2]), float(row[3])
        all_points.append((x, y))

num_trajectories = 1000
points_per_trajectory = 100

expert_data = [
    all_points[i * points_per_trajectory:(i + 1) * points_per_trajectory]
    for i in range(num_trajectories)
]

# Prepare Data for Training
# Create input-output pairs (state + goal -> next state)
X_train = []
Y_train = []

for traj in expert_data:
    for i in range(len(traj) - 1):
        X_train.append(np.hstack([traj[i], final_point]))  # Current state + goal
        Y_train.append(traj[i + 1])  # Next state

X_train = torch.tensor(np.array(X_train), dtype=torch.float32)  # Shape: (N, 4)
Y_train = torch.tensor(np.array(Y_train), dtype=torch.float32)  # Shape: (N, 2)

# Initialize Model, Loss Function, and Optimizers
model = ImitationNet(input_size=4, hidden_size=64, output_size=2)
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the Model
num_epochs = 5000
losses = []

for epoch in range(num_epochs):
    predictions = model(X_train)
    loss = criterion(predictions, Y_train)

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Generate multiple trajectories using the trained model
num_generated_trajectories = 100
generated_trajectories = []

for _ in range(num_generated_trajectories):
    with torch.no_grad():
        state = np.hstack([initial_point, final_point])  # Initial state + goal
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        trajectory = [initial_point]

        for _ in range(points_per_trajectory - 1):
            next_state = model(state).numpy().squeeze()
            trajectory.append(next_state)
            state = torch.tensor(np.hstack([next_state, final_point]), dtype=torch.float32).unsqueeze(0)
    
    generated_trajectories.append(np.array(trajectory))

# Calculate evaluation metrics
# 1. KL Divergence (distribution-level similarity)
kl_div = calculate_kl_divergence(expert_data, generated_trajectories)

# 2. MSE (average point-wise error for the first generated trajectory)
mse_single = calculate_mse(np.array(expert_data[0]), generated_trajectories[0])

# 3. EMD (distribution-level similarity considering trajectory structure)
emd_value = calculate_emd(expert_data[:num_generated_trajectories], generated_trajectories)

print(f"Evaluation Metrics:")
print(f"KL Divergence: {kl_div:.4f}")
print(f"MSE (first trajectory): {mse_single:.4f}")
print(f"EMD: {emd_value:.4f}")

# Plot the Expert and Generated Trajectories with a Single Central Obstacle
plt.figure(figsize=(20, 8))

# Plot a few expert trajectories
for traj in expert_data[:10]:
    x = [point[0] for point in traj]
    y = [point[1] for point in traj]
    plt.plot(x, y, 'b--', alpha=0.5, label='Expert' if traj is expert_data[0] else "")

# Plot the generated trajectories
for i, traj in enumerate(generated_trajectories):
    plt.plot(traj[:, 0], traj[:, 1], 'r-', alpha=0.7, label='Generated' if i == 0 else "")

# Plot the single central obstacle as a circle
ox, oy, r = obstacle
circle = plt.Circle((ox, oy), r, color='gray', alpha=0.3)
plt.gca().add_patch(circle)

# Mark start and end points
plt.scatter(initial_point[0], initial_point[1], c='green', s=100, label='Start')
plt.scatter(final_point[0], final_point[1], c='red', s=100, label='End')

plt.legend()
plt.title(f'Imitation Learning: Expert vs Generated Trajectories (EMD: {emd_value:.2f})')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.savefig('figures/single_agent/dual_mode/SADM_with_emd.png')
plt.show()

# Plot the Training Loss
plt.figure()
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('/mnt/data1/chendazhong/imitation_learning/figures/single_mode/loss_5000epochs_1000expert.png')
plt.show()