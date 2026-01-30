import torch
import matplotlib.pyplot as plt

torch.manual_seed(42)

# Data
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

# Model
model = torch.nn.Linear(1, 1)
print(f"Initial random weight: {model.weight.item():0.5f}")
print(f"Initial random bias:   {model.bias.item():0.5f}")

learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# For plotting
X_plot = torch.linspace(0, 5, 100).view(-1, 1)
plt.figure(figsize=(8, 6))

# Initial random line
init_weight = model.weight.item()
init_bias = model.bias.item()
y_init = init_weight * X_plot + init_bias
plt.plot(X_plot, y_init, "r:", linewidth=2, label="Initial random line")

# Target line
plt.plot(X_plot, 2 * X_plot + 1, "k--", label="Target line")

# Data points
plt.scatter(X, y, color="black")

for epoch in range(5):
    # Forward
    y_pred = model(X)

    # Loss
    loss = torch.mean((y_pred - y) ** 2)

    # Backward
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Plot current line
    with torch.no_grad():
        y_line = model(X_plot)
        plt.plot(X_plot, y_line, label=f"Epoch {epoch}")

    print(
        f"Epoch {epoch} | "
        f"w = {model.weight.item():.3f}, "
        f"b = {model.bias.item():.3f}, "
        f"loss = {loss.item():.4f}"
    )

plt.legend()
plt.xlabel("x (Number of km driven)")
plt.ylabel("y (Final Taxi Price)")
plt.title("Line learning step-by-step")
plt.show()
