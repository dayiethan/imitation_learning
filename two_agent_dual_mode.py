import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import entropy

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
# Define initial and final points, and a single central obstacle
initial_point = np.array([0.0, 0.0])
final_point = np.array([20.0, 0.0])
initial_point_rev = np.array([20.0, 0.0])
final_point_rev = np.array([0.0, 0.0])
obstacle = (10, 0, 4.0)  # Single central obstacle: (x, y, radius)

# Parse expert data from single_uni_full_traj.csv
import csv
with open('data/single_uni_full_traj.csv', 'r') as file:
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
first_trajectory = expert_data[0]
x = [point[0] for point in first_trajectory]
y = [point[1] for point in first_trajectory]

all_points_rev = list(reversed(all_points))

expert_data_rev = [
    all_points_rev[i * points_per_trajectory:(i + 1) * points_per_trajectory]
    for i in range(num_trajectories)
]
first_trajectory_rev = expert_data_rev[0]
x_rev = [point[0] for point in first_trajectory_rev]
y_rev = [point[1] for point in first_trajectory_rev]


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

X_train_rev = []
Y_train_rev = []

for traj in expert_data_rev:
    for i in range(len(traj) - 1):
        X_train_rev.append(np.hstack([traj[i], final_point]))  # Current state + goal
        Y_train_rev.append(traj[i + 1])  # Next state

X_train_rev = torch.tensor(np.array(X_train_rev), dtype=torch.float32)  # Shape: (N, 4)
Y_train_rev = torch.tensor(np.array(Y_train_rev), dtype=torch.float32)  # Shape: (N, 2)


# Initialize Model, Loss Function, and Optimizers
model = ImitationNet(input_size=4, hidden_size=64, output_size=2)
model_rev = ImitationNet(input_size=4, hidden_size=64, output_size=2)
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

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

# Generate a New Trajectory Using the Trained Model
with torch.no_grad():
    state = np.hstack([initial_point, final_point])  # Initial state + goal
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    generated_trajectory = [initial_point]

    state_rev = np.hstack([initial_point_rev, final_point_rev])  # Initial state + goal
    state_rev = torch.tensor(state_rev, dtype=torch.float32).unsqueeze(0)
    generated_trajectory_rev = [initial_point_rev]

    for _ in range(points_per_trajectory - 1):  # 100 steps total
        next_state = model(state).numpy().squeeze()
        generated_trajectory.append(next_state)
        state = torch.tensor(np.hstack([next_state, final_point]), dtype=torch.float32).unsqueeze(0)

        next_state_rev = model_rev(state_rev).numpy().squeeze()
        generated_trajectory_rev.append(next_state_rev)
        state_rev = torch.tensor(np.hstack([next_state_rev, final_point_rev]), dtype=torch.float32).unsqueeze(0)

generated_trajectory = np.array(generated_trajectory)
generated_trajectory_rev = np.array(generated_trajectory_rev)
# After generating the trajectories for both agents
generated_trajectory_up = np.array(generated_trajectory)  # Assuming this is agent 1 (up)
generated_trajectory_down = np.array(generated_trajectory_rev)  # Assuming this is agent 2 (down)

# Calculate KL Divergence and MSE for both agents
kl_div_up = calculate_kl_divergence(expert_data, generated_trajectory_up)
mse_up = calculate_mse(expert_data, generated_trajectory_up)

kl_div_down = calculate_kl_divergence(expert_data_rev, generated_trajectory_down)
mse_down = calculate_mse(expert_data_rev, generated_trajectory_down)

# Print the results for each agent separately
print(f"KL Divergence Up: {kl_div_up:.4f}, MSE Up: {mse_up:.4f}")
print(f"KL Divergence Down: {kl_div_down:.4f}, MSE Down: {mse_down:.4f}")

# Plot the Expert and Generated Trajectories with a Single Central Obstacle
plt.figure(figsize=(20, 8))
# for traj in expert_data[:20]:  # Plot a few expert trajectories
#     first_trajectory = traj
#     x = [point[0] for point in first_trajectory]
#     y = [point[1] for point in first_trajectory]
#     plt.plot(x, y, 'b--')

# for traj in expert_data_rev[:20]:  # Plot a few expert trajectories
#     first_trajectory = traj
#     x = [point[0] for point in first_trajectory]
#     y = [point[1] for point in first_trajectory]
#     plt.plot(x, y, 'b--')

# Plot the generated trajectory
plt.plot(generated_trajectory[:, 0], generated_trajectory[:, 1], 'r-', label='Generated')
plt.plot(generated_trajectory_rev[:, 0], generated_trajectory_rev[:, 1], 'y-', label='Generated')

# Plot the single central obstacle as a circle
ox, oy, r = obstacle
circle = plt.Circle((ox, oy), r, color='gray', alpha=0.3)
plt.gca().add_patch(circle)

# Mark start and end points
plt.scatter(initial_point[0], initial_point[1], c='red', s=100, label='Start/End')
plt.scatter(final_point[0], final_point[1], c='red', s=100, label='Start/End')

# plt.legend()
# plt.title('Smooth Imitation Learning: Expert vs Generated Trajectories')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.savefig('figures/two_agent/dual_mode/TADM_noexpert.png')
plt.show()

# # Plot the Training Loss
# plt.figure()
# plt.plot(losses)
# plt.title('Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.grid(True)
# # plt.savefig('figures/single_mode/loss_5000epochs_1000expert.png')
# plt.show()