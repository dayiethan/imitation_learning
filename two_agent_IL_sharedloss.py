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
initial_point1 = np.array([0.0, 0.0])
final_point1 = np.array([20.0, 0.0])
initial_point2 = np.array([20.0, 0.0])
final_point2 = np.array([0.0, 0.0])
obstacle = (10, 0, 4.0)  # Single central obstacle: (x, y, radius)


# Parsing data
num_trajectories = 1000
points_per_trajectory = 100

# import csv
# with open('data/single_uni_full_traj_up.csv', 'r') as file:
#     reader = csv.reader(file)
#     all_points1 = []
#     for row in reader:
#         x, y = float(row[2]), float(row[3])
#         all_points1.append((x, y))

# expert_data1 = [
#     all_points1[i * points_per_trajectory:(i + 1) * points_per_trajectory]
#     for i in range(num_trajectories)
# ]
# first_trajectory1 = expert_data1[0]
# x1 = [point[0] for point in first_trajectory1]
# y1 = [point[1] for point in first_trajectory1]

# import csv
# with open('data/single_uni_full_traj_down.csv', 'r') as file:
#     reader = csv.reader(file)
#     all_points2 = []
#     for row in reader:
#         x, y = float(row[2]), float(row[3])
#         all_points2.append((x, y))
#     all_points2 = list(reversed(all_points2))

# expert_data2 = [
#     all_points2[i * points_per_trajectory:(i + 1) * points_per_trajectory]
#     for i in range(num_trajectories)
# ]
# first_trajectory2 = expert_data2[0]
# x2 = [point[0] for point in first_trajectory2]
# y2 = [point[1] for point in first_trajectory2]
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

expert_data1 = [
    all_up_points[i * points_per_trajectory:(i + 1) * points_per_trajectory]
    for i in range(num_trajectories)
]
first_trajectory1 = expert_data1[0]
x_up = [point[0] for point in first_trajectory1]
y_up = [point[1] for point in first_trajectory1]

expert_data2 = [
    all_down_points[i * points_per_trajectory:(i + 1) * points_per_trajectory]
    for i in range(num_trajectories)
]
first_trajectory2 = expert_data2[0]
x_down = [point[0] for point in first_trajectory2]
y_down = [point[1] for point in first_trajectory2]


# Prepare Data for Training
# Create input-output pairs (state + goal -> next state)
X_train1 = []
Y_train1 = []

for traj in expert_data1:
    for i in range(len(traj) - 1):
        X_train1.append(np.hstack([traj[i], final_point1]))  # Current state + goal
        Y_train1.append(traj[i + 1])  # Next state

X_train1 = torch.tensor(np.array(X_train1), dtype=torch.float32)  # Shape: (N, 4)
Y_train1 = torch.tensor(np.array(Y_train1), dtype=torch.float32)  # Shape: (N, 2)

X_train2 = []
Y_train2 = []

for traj in expert_data2:
    for i in range(len(traj) - 1):
        X_train2.append(np.hstack([traj[i], final_point2]))  # Current state + goal
        Y_train2.append(traj[i + 1])  # Next state

X_train2 = torch.tensor(np.array(X_train2), dtype=torch.float32)  # Shape: (N, 4)
Y_train2 = torch.tensor(np.array(Y_train2), dtype=torch.float32)  # Shape: (N, 2)



# Initialize Model, Loss Function, and Optimizers
model1 = ImitationNet(input_size=4, hidden_size=64, output_size=2)
model2 = ImitationNet(input_size=4, hidden_size=64, output_size=2)
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(list(model1.parameters()) + list(model1.parameters()), lr=0.001)
alpha, beta = 0.5, 0.5

# Train the Model
num_epochs = 1000
losses = []

for epoch in range(num_epochs):
    predictions1 = model1(X_train1)
    predictions2 = model2(X_train2)
    loss1 = criterion(predictions1, Y_train1)
    loss2 = criterion(predictions2, Y_train2)
    loss = alpha * loss1 + beta * loss2

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Generate a New Trajectory Using the Trained Model
with torch.no_grad():
    state1 = np.hstack([initial_point1, final_point1])  # Initial state + goal
    state1 = torch.tensor(state1, dtype=torch.float32).unsqueeze(0)
    generated_trajectory1 = [initial_point1]

    for _ in range(points_per_trajectory - 1):  # 100 steps total
        next_state1 = model1(state1).numpy().squeeze()
        generated_trajectory1.append(next_state1)
        state1 = torch.tensor(np.hstack([next_state1, final_point1]), dtype=torch.float32).unsqueeze(0)

    state2 = np.hstack([initial_point2, final_point2])  # Initial state + goal
    state2 = torch.tensor(state2, dtype=torch.float32).unsqueeze(0)
    generated_trajectory2 = [initial_point2]

    for _ in range(points_per_trajectory - 1):  # 100 steps total
        next_state2 = model2(state2).numpy().squeeze()
        generated_trajectory1.append(next_state2)
        state2 = torch.tensor(np.hstack([next_state2, final_point2]), dtype=torch.float32).unsqueeze(0)

generated_trajectory1 = np.array(generated_trajectory1)
generated_trajectory2 = np.array(generated_trajectory2)

# Plot the Expert and Generated Trajectories with a Single Central Obstacle
plt.figure(figsize=(20, 8))
for traj in expert_data1[:20]:  # Plot a few expert trajectories
    first_trajectory = traj
    x = [point[0] for point in first_trajectory]
    y = [point[1] for point in first_trajectory]
    plt.plot(x, y, 'g--')
for traj in expert_data2[:20]:  # Plot a few expert trajectories
    first_trajectory = traj
    x = [point[0] for point in first_trajectory]
    y = [point[1] for point in first_trajectory]
    plt.plot(x, y, 'g--')

# Plot the generated trajectory
plt.plot(generated_trajectory1[:, 0], generated_trajectory1[:, 1], 'r-', label='Generated Agent 1')
plt.plot(generated_trajectory2[:, 0], generated_trajectory2[:, 1], 'b-', label='Generated Agent 2')

# Plot the single central obstacle as a circle
ox, oy, r = obstacle
circle = plt.Circle((ox, oy), r, color='gray', alpha=0.3)
plt.gca().add_patch(circle)

# Mark start and end points
plt.scatter(initial_point1[0], initial_point1[1], c='green', s=100)
plt.scatter(final_point1[0], final_point1[1], c='red', s=100)

plt.legend()
plt.title('Smooth Imitation Learning: Expert vs Generated Trajectories')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
# plt.savefig('figures/single_mode/expertlearned_5000epochs_1000expert.png')
plt.show()

# Plot the Training Loss
plt.figure()
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
# plt.savefig('figures/single_mode/loss_5000epochs_1000expert.png')
plt.show()
