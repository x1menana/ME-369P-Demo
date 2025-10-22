# Demo for implementing a simple linear regression model using pytorch

## Prerequisites:
- Python 3.9 or later 

- Pytorch

Install pytorch via pip: 
```
pip install torch
```
## Running code:
To run via command line:
```
python3 demo.py
```
## Code Walkthrough:
```
import torch
import torch.nn as nn
```
- Imports torch module to python script

```
input_tensors = torch.linspace(10, -10, 100, dtype=torch.float32).unsqueeze(1)
output_tensor = 5*input_tensors + 10 + (0.8 * torch.randn_like(input_tensors))
```
- Creates data tensors x and y

```
class linearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # this linear layer contains weight (m) and bias (+b)
        self.linear = nn.Linear(1, 1) # x -> y

    def forward(self, x):
        return self.linear(x)

model = linearRegression()
```
- Defines a neural network model with a singular linear layer. Computes the problem: y = mx + b. 
- Declares a model object

```
loss_func = nn.MSELoss() # use mean squared error as loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # stochatic gradient descent for optimization
```
- Loss function: Criterion for optimization. Here we use Mean Squared Error
- Optimizer: Algorithm used for adjusting model weights and biases.

```
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
```
- A single training epoch corresponds to one pass over the entire data set. Here we loop through the data set 100 times.
  
At each iteration: 

1: zero out the gradient. 

2: Obtain the current model predictions.

3: Calculate the loss between the predicted values versus actual. 

4: Use the loss to compute the gradient

5: Adjust the parameters accordingly. 

