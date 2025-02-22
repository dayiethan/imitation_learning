import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import csv
from scipy.stats import entropy

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Define the Generator
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

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_size=6, hidden_size=64):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # Binary classification (Expert or Generator)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)

# Define initial and final points, and a single central obstacle
initial_point_up = np.array([0.0, 0.0])
final_point_up = np.array([20.0, 0.0])
final_point_down = np.array([0.0, 0.0])
initial_point_down = np.array([20.0, 0.0])
obstacle = (10, 0, 4.0)  # Single central obstacle: (x, y, radius)

# Load expert data
with open('data/single_uni_full_traj_up.csv', 'r') as file:
    reader = csv.reader(file)
    all_up_points = []
    for row in reader:
        x, y = float(row[2]), float(row[3])
        all_up_points.append((x, y))

with open('data/single_uni_full_traj_down.csv', 'r') as file:
    reader = csv.reader(file)
    all_down_points = []
    for row in reader:
        x, y = float(row[2]), float(row[3])
        all_down_points.append((x, y))
    all_down_points = list(reversed(all_down_points))

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

# Prepare data for training
X_train_up = []
Y_train_up = []

for traj in expert_data_up:
    for i in range(len(traj) - 1):
        X_train_up.append(np.hstack([traj[i], final_point_up]))  # Current state + goal
        Y_train_up.append(traj[i + 1])  # Next state

X_train_up = torch.tensor(np.array(X_train_up), dtype=torch.float32)
Y_train_up = torch.tensor(np.array(Y_train_up), dtype=torch.float32)

X_train_down = []
Y_train_down = []

for traj in expert_data_down:
    for i in range(len(traj) - 1):
        X_train_down.append(np.hstack([traj[i], final_point_down]))  # Current state + goal
        Y_train_down.append(traj[i + 1])  # Next state

X_train_down = torch.tensor(np.array(X_train_down), dtype=torch.float32)
Y_train_down = torch.tensor(np.array(Y_train_down), dtype=torch.float32)

# Initialize Generator and Discriminator
generator_up = Generator(input_size=4, hidden_size=64, output_size=2)
discriminator_up = Discriminator(input_size=6, hidden_size=64)

generator_down = Generator(input_size=4, hidden_size=64, output_size=2)
discriminator_down = Discriminator(input_size=6, hidden_size=64)

# Define optimizers
optimizer_generator_up = optim.Adam(generator_up.parameters(), lr=0.001)
optimizer_discriminator_up = optim.Adam(discriminator_up.parameters(), lr=0.001)

optimizer_generator_down = optim.Adam(generator_down.parameters(), lr=0.001)
optimizer_discriminator_down = optim.Adam(discriminator_down.parameters(), lr=0.001)

# Loss functions
criterion_mse = nn.MSELoss()
criterion_bce = nn.BCELoss()  # Binary Cross-Entropy for Discriminator

# Training loop
num_epochs = 5000
for epoch in range(num_epochs):
    # Train Discriminator for "up" trajectory
    optimizer_discriminator_up.zero_grad()
    expert_labels_up = torch.ones(X_train_up.size(0), 1)
    generated_up = generator_up(X_train_up)
    generated_labels_up = torch.zeros(X_train_up.size(0), 1)

    real_pred_up = discriminator_up(torch.cat([X_train_up, Y_train_up], dim=1))
    fake_pred_up = discriminator_up(torch.cat([X_train_up, generated_up.detach()], dim=1))

    loss_real_up = criterion_bce(real_pred_up, expert_labels_up)
    loss_fake_up = criterion_bce(fake_pred_up, generated_labels_up)
    loss_discriminator_up = loss_real_up + loss_fake_up
    loss_discriminator_up.backward()
    optimizer_discriminator_up.step()

    # Train Generator for "up" trajectory
    optimizer_generator_up.zero_grad()
    fake_pred_up = discriminator_up(torch.cat([X_train_up, generated_up], dim=1))
    loss_generator_up = criterion_bce(fake_pred_up, expert_labels_up) + criterion_mse(generated_up, Y_train_up)
    loss_generator_up.backward()
    optimizer_generator_up.step()

    # Train Discriminator for "down" trajectory
    optimizer_discriminator_down.zero_grad()
    expert_labels_down = torch.ones(X_train_down.size(0), 1)
    generated_down = generator_down(X_train_down)
    generated_labels_down = torch.zeros(X_train_down.size(0), 1)

    real_pred_down = discriminator_down(torch.cat([X_train_down, Y_train_down], dim=1))
    fake_pred_down = discriminator_down(torch.cat([X_train_down, generated_down.detach()], dim=1))

    loss_real_down = criterion_bce(real_pred_down, expert_labels_down)
    loss_fake_down = criterion_bce(fake_pred_down, generated_labels_down)
    loss_discriminator_down = loss_real_down + loss_fake_down
    loss_discriminator_down.backward()
    optimizer_discriminator_down.step()

    # Train Generator for "down" trajectory
    optimizer_generator_down.zero_grad()
    fake_pred_down = discriminator_down(torch.cat([X_train_down, generated_down], dim=1))
    loss_generator_down = criterion_bce(fake_pred_down, expert_labels_down) + criterion_mse(generated_down, Y_train_down)
    loss_generator_down.backward()
    optimizer_generator_down.step()

    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss Generator Up: {loss_generator_up.item():.4f}, Loss Discriminator Up: {loss_discriminator_up.item():.4f}')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss Generator Down: {loss_generator_down.item():.4f}, Loss Discriminator Down: {loss_discriminator_down.item():.4f}')

# Generate a New Trajectory Using the Trained Generator
with torch.no_grad():
    state_up = np.hstack([initial_point_up, final_point_up])
    state_up = torch.tensor(state_up, dtype=torch.float32).unsqueeze(0)
    generated_trajectory_up = [initial_point_up]

    state_down = np.hstack([initial_point_down, final_point_down])
    state_down = torch.tensor(state_down, dtype=torch.float32).unsqueeze(0)
    generated_trajectory_down = [initial_point_down]

    for _ in range(points_per_trajectory - 1):
        next_state_up = generator_up(state_up).numpy().squeeze()
        generated_trajectory_up.append(next_state_up)
        state_up = torch.tensor(np.hstack([next_state_up, final_point_up]), dtype=torch.float32).unsqueeze(0)

        next_state_down = generator_down(state_down).numpy().squeeze()
        generated_trajectory_down.append(next_state_down)
        state_down = torch.tensor(np.hstack([next_state_down, final_point_down]), dtype=torch.float32).unsqueeze(0)

generated_trajectory_up = np.array(generated_trajectory_up)
generated_trajectory_down = np.array(generated_trajectory_down)

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

# Example of calculating KL Divergence and MSE for generated and expert trajectories
kl_div_up = calculate_kl_divergence(expert_data_up, generated_trajectory_up)
kl_div_down = calculate_kl_divergence(expert_data_down, generated_trajectory_down)

mse_up = calculate_mse(expert_data_up, generated_trajectory_up)
mse_down = calculate_mse(expert_data_down, generated_trajectory_down)

# Print the results
print(f"KL Divergence Up: {kl_div_up:.4f}, MSE Up: {mse_up:.4f}")
print(f"KL Divergence Down: {kl_div_down:.4f}, MSE Down: {mse_down:.4f}")
