import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the MLP model for imitation learning
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

# Example expert_data in the updated format
# Each element is a trajectory with positions of both agents: [(x1, y1, x2, y2), ...]
# Generate example expert_data without using ellipsis
# Each trajectory contains sequences of (x1, y1, x2, y2) for both agents
expert_data = [
    [(x1, np.sin(x1 / 2), 20 - x1, -np.sin(x1 / 2)) for x1 in np.linspace(0, 20, 50)]
    for _ in range(100)
]
import csv
with open('data/full_traj_two_simple.csv', 'r') as file:
    reader = csv.reader(file)
    all_points = []
    for row in reader:
        x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        all_points.append((x1, y1, x2, y2))

num_trajectories = 1000
points_per_trajectory = 100

expert_data = [
    all_points[i * points_per_trajectory:(i + 1) * points_per_trajectory]
    for i in range(num_trajectories)
]
first_trajectory = expert_data[0]
x1 = [point[0] for point in first_trajectory]
y1 = [point[1] for point in first_trajectory]
x2 = [point[2] for point in first_trajectory]
y2 = [point[3] for point in first_trajectory]


# Define goals for each agent
agent1_goal = [20, 0]
agent2_goal = [0, 0]
agent_goal = [20, 0, 0, 0]

# Prepare the data for training
X_train, Y_train = [], []

for trajectory in expert_data:
    for i in range(len(trajectory) - 1):
        x1, y1, x2, y2 = trajectory[i]            # Current positions of agent 1 and agent 2
        next_x1, next_y1, next_x2, next_y2 = trajectory[i + 1]  # Next positions

        X_train.append(np.hstack([trajectory[i], agent_goal]))
        Y_train.append(trajectory[i + 1])

# Convert lists to PyTorch tensors with the correct shapes
X_train = torch.tensor(X_train, dtype=torch.float32)  # Shape: (num_samples, 4)
Y_train = torch.tensor(Y_train, dtype=torch.float32)  # Shape: (num_samples, 2)

# Verify dimensions
print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")

# Initialize model, loss function, and optimizer
model = ImitationNet(input_size=8, hidden_size=64, output_size=4)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
losses = []
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)


for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    mse_loss = criterion(outputs, Y_train)

    # Additional penalties
    goal_penalty = torch.mean((outputs - Y_train)**2)

    total_loss = mse_loss + 0.1 * goal_penalty
    total_loss.backward()
    optimizer.step()

    scheduler.step()  # Adjust learning rate
    losses.append(total_loss.item())


    losses.append(total_loss.item())
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.4f}")
        print(goal_penalty.item())

# Generate a trajectory using the learned model for each agent
def generate_trajectory(model, start, goal, steps=20):
    model.eval()
    trajectory = [start]
    with torch.no_grad():
        current = np.array(start)
        for _ in range(steps):
            input_tensor = torch.tensor([*current, *goal], dtype=torch.float32).unsqueeze(0)
            next_step = model(input_tensor).numpy().squeeze()
            trajectory.append(next_step)
            current = next_step
    return np.array(trajectory)

# Define start and goal positions for each agent
agent1_start, agent1_goal = [0, 0], [20, 0]
agent2_start, agent2_goal = [20, 0], [0, 0]
agent_start  = [0, 0, 20, 0]
agent_goal = [20, 0, 0, 0]

# Generate trajectories
# agent1_trajectory = generate_trajectory(model, agent1_start, agent1_goal)
# agent2_trajectory = generate_trajectory(model, agent2_start, agent2_goal)
agent_trajectory = generate_trajectory(model, agent_start, agent_goal)

# Plot the generated trajectories
plt.figure(figsize=(10, 6))
plt.plot(agent_trajectory[:, 0], agent_trajectory[:, 1], label='Agent 1 Trajectory')
plt.plot(agent_trajectory[:, 2], agent_trajectory[:, 3], label='Agent 2 Trajectory')

# Plot the start and end points
plt.scatter(agent_start[0], agent_start[1], c='blue', s=100, label='Agent 1 Start')
plt.scatter(agent_goal[0], agent_goal[1], c='cyan', s=100, label='Agent 1 Goal')
plt.scatter(agent_start[2], agent_start[3], c='red', s=100, label='Agent 2 Start')
plt.scatter(agent_goal[2], agent_goal[3], c='orange', s=100, label='Agent 2 Goal')

# Plot the obstacle
obstacle_circle = plt.Circle((10, 0), 1, color='gray', alpha=0.3)
plt.gca().add_patch(obstacle_circle)

# for traj in expert_data[:20]:  # Plot a few expert trajectories
#     first_trajectory = traj
#     x = [point[0] for point in first_trajectory]
#     y = [point[1] for point in first_trajectory]
#     plt.plot(x, y, 'b--')
#     x = [point[2] for point in first_trajectory]
#     y = [point[3] for point in first_trajectory]
#     plt.plot(x, y, 'g--')

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Generated Trajectories for Agents with Obstacle Avoidance")
plt.grid(True)
plt.show()
