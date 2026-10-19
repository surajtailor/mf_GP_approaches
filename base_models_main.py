import torch
from data_generation import generate_lofi_data, generate_hifi_data, generate_test_data, create_scalers, scale_data
import random
import matplotlib.pyplot as plt
import numpy as np
import gpytorch
from models.base_models import BaseGPModel, BaseNNModel

## Set device and seeds for numpy, torch and random
seed = 2025
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = 'cpu'
scale_data_switch = True

## Set directory name
direc_name = "GP_MF_2D_Example"

## Set plotting parameters
title_fontsize = 9
axis_fontsize = 9
legend_fontsize = 6
linewidth = 0.7
scatter_size = 2

## Generate Lo-Fi and Hi-Fi data
n = 50

# split [0,1] into n equal bins
edges = np.linspace(0, 1, n+1)

# build regions as list of [(x_min,x_max), (y_min,y_max)]
hifi_regions = [[(edges[i], edges[i+1]), (edges[j], edges[j+1])] for i in range(n) for j in range(n)]

# Parameters
n_clusters = 10          # how many clusters to attempt
total_points_range = (1, 100)  # points per cluster
cluster_radius_range = (0.05, 0.2)  # cluster radii
domain_min, domain_max = 0, 1        # domain bounds
points = []

for _ in range(n_clusters):
    # Random cluster center
    cx, cy = np.random.uniform(domain_min, domain_max, 2)

    # Random cluster radius
    r = np.random.uniform(*cluster_radius_range)

    # Random number of points in this cluster (introduces sparsity)
    n_points = np.random.randint(*total_points_range)

    # Generate points in a circle using polar coordinates
    theta = np.random.uniform(0, 2*np.pi, n_points)
    radius = r * np.sqrt(np.random.rand(n_points))  # uniform in circle
    xs = cx + radius * np.cos(theta)
    ys = cy + radius * np.sin(theta)

    # Clip points to stay inside [0,1]^2
    xs = np.clip(xs, domain_min, domain_max)
    ys = np.clip(ys, domain_min, domain_max)
    points.append(np.column_stack([xs, ys]))

# Final dataset
points = np.vstack(points)
print(f"Generated {points.shape[0]} points across {n_clusters} clusters.")

hifi_regions_num_points = []
for (xrange, yrange) in hifi_regions:
    x_min, x_max = xrange
    y_min, y_max = yrange
    mask = ((points[:,0] >= x_min) & (points[:,0] < x_max) &
            (points[:,1] >= y_min) & (points[:,1] < y_max))
    hifi_regions_num_points.append(np.sum(mask))

lofi_regions = [(0, 1)]
lofi_regions_num_points = [1000]

# Generate each subset of data
x_hifi_data, y_hifi_data = generate_hifi_data(hifi_regions, hifi_regions_num_points, seed = seed)

# Concatenate all the hifi data
x_hifi_data = torch.cat([x_hifi_data], dim=0)
y_hifi_data = torch.cat([y_hifi_data], dim=0)
x_lofi_data, y_lofi_data = generate_lofi_data(lofi_regions, lofi_regions_num_points)
x_test_data, y_test_data, x1_test_data_grid, x2_test_data_grid = generate_test_data(test_num=10000)

# Create scalers
scalers = create_scalers(x_lofi_data, y_lofi_data, x_hifi_data, y_hifi_data)

# Scale data
x_hifi_scaled_all, y_hifi_scaled_all = scale_data(x_hifi_data, y_hifi_data, scalers)
x_lofi_scaled, y_lofi_scaled = scale_data(x_lofi_data, y_lofi_data, scalers)
x_test_scaled, y_test_scaled = scale_data(x_test_data, y_test_data, scalers)

# total number of samples
n = x_hifi_scaled_all.shape[0]
# create shuffled indices
split_ratio = 0.7
perm = np.random.permutation(n)
# compute split index
split_index = int(split_ratio * n)
# apply shuffled split
train_idx = perm[:split_index]
valid_idx = perm[split_index:]
x_hifi_scaled = x_hifi_scaled_all[train_idx]
y_hifi_scaled = y_hifi_scaled_all[train_idx]
x_valid_scaled = x_hifi_scaled_all[valid_idx]
y_valid_scaled = y_hifi_scaled_all[valid_idx]

# Concatenating the data
hifi_train_data = (x_hifi_scaled, y_hifi_scaled)
lofi_train_data = (x_lofi_scaled, y_lofi_scaled)
valid_data = (x_valid_scaled, y_valid_scaled)
x_Global_Scaler, y_Global_Scaler = scalers

## Create Hi-Fi GP model
# # Initialise likelihood.
likelihood = gpytorch.likelihoods.GaussianLikelihood()
# # Create model 1 on Lo-Fi data
# baseGPmodel = BaseGPModel(x_hifi_scaled, y_hifi_scaled, likelihood)
#
# # Find optimal model hyperparameters
# baseGPmodel.train()
# likelihood.train()
# # Use the adam optimizer
# optimizer = torch.optim.Adam(baseGPmodel.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters
# # "Loss" for GPs - the marginal log likelihood
# mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, baseGPmodel)
#
# ## GP model
# iter = 10000
# # Training loop
# for i in range(iter):
#     # Zero gradients from previous iteration
#     optimizer.zero_grad()
#     # Output from model
#     output = baseGPmodel(x_hifi_scaled)
#     # Calc loss and backprop gradients
#     loss = -mll(output, y_hifi_scaled)
#     loss.backward()
#     print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
#         i + 1, iter, loss.item(),
#         baseGPmodel.covar_module.base_kernel.lengthscale.item(),
#         baseGPmodel.likelihood.noise.item()
#     ))
#     optimizer.step()
#
# torch.save(baseGPmodel, 'base_GP_model')

## Create Hi-Fi NN model
input_size = x_hifi_scaled.shape[1]  # Number of features in low-fidelity data#
output_size = 1  # Assuming a single output for regression
# Initialize the neural network
nn_model = BaseNNModel(input_size, output_size)
# Define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.01)

# Train the neural network on low-fidelity data
num_epochs = 10000
for epoch in range(num_epochs):
    nn_model.train()
    optimizer.zero_grad()
    outputs = nn_model(x_hifi_scaled)
    loss = criterion(outputs, y_hifi_scaled.unsqueeze(-1))  # Ensure y_lofi_scaled is 2D
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained neural network model
torch.save(nn_model, 'base_NN_model')