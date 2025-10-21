import torch
import torch.nn as nn

# Goal: fit a linear regression model
# y = 5x + 10

# create data tensors
input_tensors = torch.linspace(10, -10, 100, dtype=torch.float32).unsqueeze(1)
output_tensor = 5*input_tensors + 10 + (0.8 * torch.randn_like(input_tensors))
print(f'input tensors: {input_tensors} \noutput tensors: {output_tensor}')

# define linear regression model
class linearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # this linear layer contains weight (m) and bias (+b)
        self.linear = nn.Linear(1, 1) # x -> y

    def forward(self, x):
        return self.linear(x)

# define model
model = linearRegression()

# create loss function and optimizer for fitting data
loss_func = nn.MSELoss() # use mean squared error as loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # stochatic gradient descent for optimization

# Print weight (m) and bias (+b) before model loop
m = model.linear.weight.data
b = model.linear.bias.data
print(f'Before training: m:{m}, b: {b}')

# run through several iterations of model
epochs = 100
for i in range(epochs):
    # zero our gradient for this training loop
    optimizer.zero_grad()

    # obtain predictions
    y_pred = model(input_tensors)

    # (y -\hat{y})
    loss = loss_func(y_pred, output_tensor)

    # compute gradient
    loss.backward()

    # recalculate model parameters based on gradient
    optimizer.step()

# print m and b after training loop
print(f'After training: m:{m}, b: {b} | target: m: {5}, b: {10}')