import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random

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

# Define initial and final points, and a single central obstacle
initial_point_up = np.array([0.0, 0.0])
final_point_up = np.array([20.0, 0.0])
final_point_down = np.array([0.0, 0.0])
initial_point_down = np.array([20.0, 0.0])
obstacle = (10, 0, 4.0)  # Single central obstacle: (x, y, radius)

# Parse expert data from single_uni_full_traj.csv
import csv
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
first_trajectory_up = expert_data_up[0]
x_up = [point[0] for point in first_trajectory_up]
y_up = [point[1] for point in first_trajectory_up]

expert_data_down = [
    all_down_points[i * points_per_trajectory:(i + 1) * points_per_trajectory]
    for i in range(num_trajectories)
]
first_trajectory_down = expert_data_down[0]
x_down = [point[0] for point in first_trajectory_down]
y_down = [point[1] for point in first_trajectory_down]


# Prepare Data for Training
# Create input-output pairs (state + goal -> next state)
X_train_up = []
Y_train_up = []

for traj in expert_data_up:
    for i in range(len(traj) - 1):
        X_train_up.append(np.hstack([traj[i], final_point_up]))  # Current state + goal
        Y_train_up.append(traj[i + 1])  # Next state

X_train_up = torch.tensor(np.array(X_train_up), dtype=torch.float32)  # Shape: (N, 4)
Y_train_up = torch.tensor(np.array(Y_train_up), dtype=torch.float32)  # Shape: (N, 2)

X_train_down = []
Y_train_down = []

for traj in expert_data_down:
    for i in range(len(traj) - 1):
        X_train_down.append(np.hstack([traj[i], final_point_down]))  # Current state + goal
        Y_train_down.append(traj[i + 1])  # Next state

X_train_down = torch.tensor(np.array(X_train_down), dtype=torch.float32)  # Shape: (N, 4)
Y_train_down = torch.tensor(np.array(Y_train_down), dtype=torch.float32)  # Shape: (N, 2)

# Initialize Model, Loss Function, and Optimizers
model_up = ImitationNet(input_size=4, hidden_size=64, output_size=2)
model_down = ImitationNet(input_size=4, hidden_size=64, output_size=2)
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(list(model_up.parameters()) + list(model_down.parameters()), lr=0.001)

# Train the Model
num_epochs = 5000
losses_up = []
losses_down = []

for epoch in range(num_epochs):
    predictions_up = model_up(X_train_up)
    loss_up = criterion(predictions_up, Y_train_up)
    predictions_down = model_down(X_train_down)
    loss_down = criterion(predictions_down, Y_train_down)
    joint_loss = 0.5 * loss_up + 0.5 * loss_down
    optimizer.zero_grad()
    joint_loss.backward()
    optimizer.step()
    losses_up.append(loss_up.item())
    losses_down.append(loss_down.item())
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss Up: {loss_up.item():.4f}, Loss Down: {loss_down.item():.4f}')

# Generate a New Trajectory Using the Trained Model
with torch.no_grad():
    state_up = np.hstack([initial_point_up, final_point_up])  # Initial state + goal
    state_up = torch.tensor(state_up, dtype=torch.float32).unsqueeze(0)
    generated_trajectory_up = [initial_point_up]

    state_down = np.hstack([initial_point_down, final_point_down])  # Initial state + goal
    state_down = torch.tensor(state_down, dtype=torch.float32).unsqueeze(0)
    generated_trajectory_down = [initial_point_down]

    for _ in range(points_per_trajectory - 1):  # 100 steps total
        next_state_up = model_up(state_up).numpy().squeeze()
        generated_trajectory_up.append(next_state_up)
        state_up = torch.tensor(np.hstack([next_state_up, final_point_up]), dtype=torch.float32).unsqueeze(0)

        next_state_down = model_down(state_down).numpy().squeeze()
        generated_trajectory_down.append(next_state_down)
        state_down = torch.tensor(np.hstack([next_state_down, final_point_down]), dtype=torch.float32).unsqueeze(0)

generated_trajectory_up = np.array(generated_trajectory_up)
generated_trajectory_down = np.array(generated_trajectory_down)


# Plot the Expert and Generated Trajectories with a Single Central Obstacle
plt.figure(figsize=(20, 8))
# for traj in expert_data_up[:20]:  # Plot a few expert trajectories
#     first_trajectory = traj
#     x = [point[0] for point in first_trajectory]
#     y = [point[1] for point in first_trajectory]
#     plt.plot(x, y, 'b--')
# for traj in expert_data_down[:20]:  # Plot a few expert trajectories
#     first_trajectory = traj
#     x = [point[0] for point in first_trajectory]
#     y = [point[1] for point in first_trajectory]
#     plt.plot(x, y, 'g--')

# Plot the generated trajectory
plt.plot(generated_trajectory_up[:, 0], generated_trajectory_up[:, 1], 'r-', label='Generated')
plt.plot(generated_trajectory_down[:, 0], generated_trajectory_down[:, 1], 'y-', label='Generated')

# Plot the single central obstacle as a circle
ox, oy, r = obstacle
circle = plt.Circle((ox, oy), r, color='gray', alpha=0.3)
plt.gca().add_patch(circle)

# Mark start and end points
plt.scatter(initial_point_up[0], initial_point_up[1], c='red', s=100, label='Start/End')
plt.scatter(final_point_up[0], final_point_up[1], c='red', s=100, label='Start/End')

# plt.legend()
# plt.title('Smooth Imitation Learning: Expert vs Generated Trajectories')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.savefig('figures/two_agents_shared/expert_vs_generated_trajectories_noexpert.png')
plt.show()

# Plot the Training Loss
plt.figure()
plt.plot(losses_up, label='Up')
plt.plot(losses_down, label='Down')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('figures/two_agents_shared/loss_graph.png')
plt.show()
