import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import csv
import os

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Define the Neural Network for Imitation Learning
class ImitationNet(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, output_size=2):
        super(ImitationNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define initial and final points for both agents
initial_point_agent1 = np.array([0.0, 0.0])
final_point_agent1 = np.array([20.0, 0.0])

initial_point_agent2 = np.array([20.0, 0.0])
final_point_agent2 = np.array([0.0, 0.0])

obstacle = (10, 0, 4.0)  # Obstacle: (x, y, radius)

# Function to read expert data from CSV files
def read_expert_data(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        all_points = []
        for row in reader:
            x, y = float(row[2]), float(row[3])
            all_points.append((x, y))
    return all_points

# Read expert data for both agents
# Agent 1: Moving from (0,0) to (20,0)
expert_data_agent1 = read_expert_data('data/single_uni_full_traj_up.csv')

# Agent 2: Moving from (20,0) to (0,0)
expert_data_agent2 = read_expert_data('data/single_uni_full_traj_down.csv')  # You need to provide this file

# Prepare trajectories
num_trajectories = 1000
points_per_trajectory = 100

# Function to split data into trajectories
def split_into_trajectories(all_points, num_trajectories, points_per_trajectory):
    return [
        all_points[i * points_per_trajectory:(i + 1) * points_per_trajectory]
        for i in range(num_trajectories)
    ]

trajectories_agent1 = split_into_trajectories(expert_data_agent1, num_trajectories, points_per_trajectory)
trajectories_agent2 = split_into_trajectories(expert_data_agent2, num_trajectories, points_per_trajectory)

# Prepare Data for Training
# Create input-output pairs (state + goal -> next state) for both agents
X_train_up = []
Y_train_up = []
X_train_down = []
Y_train_down = []

# Agent 1 data
for traj in trajectories_agent1:
    for i in range(len(traj) - 1):
        state = np.hstack([traj[i], final_point_agent1])  # Current state + goal
        next_state = traj[i + 1]  # Next state
        X_train_up.append(state)
        Y_train_up.append(next_state)

# Agent 2 data
for traj in trajectories_agent2:
    for i in range(len(traj) - 1):
        state = np.hstack([traj[i], final_point_agent2])  # Current state + goal
        next_state = traj[i + 1]  # Next state
        X_train_down.append(state)
        Y_train_down.append(next_state)

X_train_up = torch.tensor(np.array(X_train_up), dtype=torch.float32)  # Shape: (N, 4)
Y_train_up = torch.tensor(np.array(Y_train_up), dtype=torch.float32)  # Shape: (N, 2)
X_train_down = torch.tensor(np.array(X_train_down), dtype=torch.float32)  # Shape: (N, 4)
Y_train_down = torch.tensor(np.array(Y_train_down), dtype=torch.float32)  # Shape: (N, 2)

# Initialize Model, Loss Function, and Optimizer
model_up = ImitationNet(input_size=4, hidden_size=64, output_size=2)
model_down = ImitationNet(input_size=4, hidden_size=64, output_size=2)
optimizer = optim.Adam(list(model_up.parameters()) + list(model_down.parameters()), lr=0.001)
criterion = nn.MSELoss()
alpha, beta = 0.5, 0.5


# Train the Model
num_epochs = 5000
batch_size = 256  # You can adjust the batch size
dataset_up = torch.utils.data.TensorDataset(X_train_up, Y_train_up)
dataset_down = torch.utils.data.TensorDataset(X_train_down, Y_train_down)
dataloader_up = torch.utils.data.DataLoader(dataset_up, batch_size=batch_size, shuffle=True)
dataloader_down = torch.utils.data.DataLoader(dataset_down, batch_size=batch_size, shuffle=True)

losses = []

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_Xu, batch_Yu in dataloader_up:
        batch_Xd, batch_Yd = next(iter(dataloader_down)) 
        predictions_up = model_up(batch_Xu)
        predictions_down = model_down(batch_Xd)
        loss_up = criterion(predictions_up, batch_Yu)
        loss_down = criterion(predictions_down, batch_Yd)

        joint_loss = alpha * loss_up + beta * loss_down

        # Backpropagation and optimization
        optimizer.zero_grad()
        joint_loss.backward()
        optimizer.step()

        epoch_loss += joint_loss.item() * batch_Xu.size(0)

    epoch_loss /= len(dataset_up)
    losses.append(epoch_loss)
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Generate New Trajectories Using the Trained Model
with torch.no_grad():
    # Agent 1 trajectory
    state_agent1 = np.hstack([initial_point_agent1, final_point_agent1])  # Initial state + goal
    state_agent1 = torch.tensor(state_agent1, dtype=torch.float32).unsqueeze(0)
    generated_trajectory_agent1 = [initial_point_agent1]

    for _ in range(points_per_trajectory - 1):
        next_state = model_up(state_agent1).numpy().squeeze()
        generated_trajectory_agent1.append(next_state)
        state_agent1 = torch.tensor(np.hstack([next_state, final_point_agent1]), dtype=torch.float32).unsqueeze(0)

    generated_trajectory_agent1 = np.array(generated_trajectory_agent1)

    # Agent 2 trajectory
    state_agent2 = np.hstack([initial_point_agent2, final_point_agent2])  # Initial state + goal
    state_agent2 = torch.tensor(state_agent2, dtype=torch.float32).unsqueeze(0)
    generated_trajectory_agent2 = [initial_point_agent2]

    for _ in range(points_per_trajectory - 1):
        next_state = model_down(state_agent2).numpy().squeeze()
        generated_trajectory_agent2.append(next_state)
        state_agent2 = torch.tensor(np.hstack([next_state, final_point_agent2]), dtype=torch.float32).unsqueeze(0)

    generated_trajectory_agent2 = np.array(generated_trajectory_agent2)

# Plot the Expert and Generated Trajectories with the Obstacle
plt.figure(figsize=(20, 8))

# Plot some expert trajectories for Agent 1
for traj in trajectories_agent1[:10]:
    x = [point[0] for point in traj]
    y = [point[1] for point in traj]
    plt.plot(x, y, 'b--', alpha=0.5)

# Plot some expert trajectories for Agent 2
for traj in trajectories_agent2[:10]:
    x = [point[0] for point in traj]
    y = [point[1] for point in traj]
    plt.plot(x, y, 'g--', alpha=0.5)

# Plot the generated trajectory for Agent 1
plt.plot(generated_trajectory_agent1[:, 0], generated_trajectory_agent1[:, 1], 'r-', label='Generated Agent 1')

# Plot the generated trajectory for Agent 2
plt.plot(generated_trajectory_agent2[:, 0], generated_trajectory_agent2[:, 1], 'm-', label='Generated Agent 2')

# Plot the obstacle as a circle
ox, oy, r = obstacle
circle = plt.Circle((ox, oy), r, color='gray', alpha=0.3)
plt.gca().add_patch(circle)

# Mark start and end points for both agents
plt.scatter(initial_point_agent1[0], initial_point_agent1[1], c='blue', s=100, label='Start Agent 1')
plt.scatter(final_point_agent1[0], final_point_agent1[1], c='cyan', s=100, label='End Agent 1')

plt.scatter(initial_point_agent2[0], initial_point_agent2[1], c='orange', s=100, label='Start Agent 2')
plt.scatter(final_point_agent2[0], final_point_agent2[1], c='purple', s=100, label='End Agent 2')

plt.legend()
plt.title('Imitation Learning: Expert vs Generated Trajectories for Two Agents')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)

# Ensure the directory exists
if not os.path.exists('figures/two_agents_shared/'):
    os.makedirs('figures/two_agents_shared/')

plt.savefig('figures/two_agents_shared/expert_vs_generated_trajectories.png')
plt.show()

# Plot the Training Loss
plt.figure()
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('figures/two_agents_shared/training_loss.png')
plt.show()
