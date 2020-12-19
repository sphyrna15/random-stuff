import numpy as np 
import torch
from math import pi

""" Basics in PyTorch """
# Following PyTorch Tutorial to learn basics :
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

x = np.linspace(-pi, pi, 2000)
y = np.sin(x)

# Random Weights
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

alpha = 1e-6
for k in range(2000):

    # Forward Pass
    y_pred = a*x**3 + b*x**2 + c*x + d

    loss = np.square(y_pred - y).sum()
    if k % 200 == 199:
        print(k, loss)

    # Backprop compute gradients
    grad_y_pred = 2.0 * (y_pred - y)
    grad_d = grad_y_pred.sum()
    grad_c = (grad_y_pred * x).sum()
    grad_b = (grad_y_pred * x**2).sum()
    grad_a = (grad_y_pred * x**3).sum()

    a -= alpha * grad_a
    b -= alpha * grad_b
    c -= alpha * grad_c
    d -= alpha * grad_d


# Print results
print()
print(f'Result with numpy: y = {d} + {c} x + {b} x^2 + {a} x^3')
print()

#Calculate accurate polyfit version

p = np.polyfit(x, y, 3)
#print(params)
print(f'Result with polyfit: y = {p[3]} + {p[2]} x + {p[1]} x^2 + {p[0]} x^3')




################## NOW IN PYTORCH ################################

dtype = torch.float
dev = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU


x = torch.linspace(-pi, pi, 2000, device=dev, dtype=dtype)
y = torch.sin(x)

# Random Weights
a = torch.randn((), device=dev, dtype=dtype)
b = torch.randn((), device=dev, dtype=dtype)
c = torch.randn((), device=dev, dtype=dtype)
d = torch.randn((), device=dev, dtype=dtype)

alpha = 1e-6
for k in range(2000):

    # Forward Pass
    y_pred = a*x**3 + b*x**2 + c*x + d

    loss = np.square(y_pred - y).pow(2).sum()
    if k % 200 == 199:
        print(k, loss.item())

    # Backprop compute gradients
    grad_y_pred = 2.0 * (y_pred - y)
    grad_d = grad_y_pred.sum()
    grad_c = (grad_y_pred * x).sum()
    grad_b = (grad_y_pred * x**2).sum()
    grad_a = (grad_y_pred * x**3).sum()

    a -= alpha * grad_a
    b -= alpha * grad_b
    c -= alpha * grad_c
    d -= alpha * grad_d

print()
print(f'Result in PyTorch: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
print()


################### NOW AUTOGRAD ###########################

a = torch.randn((), device=dev, dtype=dtype, requires_grad=True)
b = torch.randn((), device=dev, dtype=dtype, requires_grad=True)
c = torch.randn((), device=dev, dtype=dtype, requires_grad=True)
d = torch.randn((), device=dev, dtype=dtype, requires_grad=True)

alpha = 1e-6
for k in range(2000):

    # Forward Pass
    y_pred = a*x**3 + b*x**2 + c*x + d

    loss = (y_pred - y).pow(2).sum()
    if k % 200 == 199:
        print(k, loss.item())

    # Different Backpass

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call a.grad, b.grad. c.grad and d.grad will be Tensors holding
    # the gradient of the loss with respect to a, b, c, d respectively.
    loss.backward()

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    with torch.no_grad():

        a -= alpha * a.grad
        b -= alpha * b.grad
        c -= alpha * c.grad
        d -= alpha * d.grad

        # Manually Zero all gradients before next round
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print()
print(f'Result with Autograd: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
print()


#### SUbclass note 
# In PyTroch, subclass autograd functions can easily be built on top of 
# Existing autograd class - see Link of Tutorial 

############## NOW WITH nn MODULE (LAYERS AND STUFF) ###############

# Input and Output Tensors
xprep = torch.linspace(-pi, pi, 2000)
y = torch.sin(xprep)

# For this example, the output y is a linear function of (x, x^2, x^3), so
# we can consider it as a linear layer neural network.
# x.unsqueeze(-1) has shape (2000, 1), and p has shape
# (3,), for this case, broadcasting semantics will apply to obtain a tensor
# of shape (2000, 3) 
p = torch.tensor([1, 2, 3])
x = xprep.unsqueeze(-1).pow(p)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output.
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)

# Define Loss Fct
loss_fct = torch.nn.MSELoss(reduction='sum')

alpha = 1e-6

for k in range(2000):

    # Forward Pass Compute Prediction
    y_pred = model(x)

    # Compute Loss
    loss = loss_fct(y_pred, y)
    if k % 200 == 199:
        print(k, loss.item())

    # Zero Gradients
    model.zero_grad()

    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor
    with torch.no_grad():
        for param in model.parameters():
            param -= alpha * param.grad


# You can access the first layer of `model` like accessing the first item of a list
ll = model[0]

# For linear layer, its parameters are stored as `weight` and `bias`.
print(f'Result: y = {ll.bias.item()} + {ll.weight[:, 0].item()} x + {ll.weight[:, 1].item()} x^2 + {ll.weight[:, 2].item()} x^3')


## For Optimizers torch.optim and custom nn Modules, see the link





