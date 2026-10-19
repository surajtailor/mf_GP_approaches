import torch
from data_generation import generate_lofi_data, generate_hifi_data, generate_test_data, create_scalers, scale_data
import random
import matplotlib.pyplot as plt
import numpy as np
import gpytorch

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
hifi_regions = [[(edges[i], edges[i+1]), (edges[j], edges[j+1])]
                for i in range(n) for j in range(n)]

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

############# METHOD ####################
# Simplest form of GP model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(),
        )
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# # Initialise likelihood.
likelihood = gpytorch.likelihoods.GaussianLikelihood()
#
# Create model 1 on Lo-Fi data
model_1 = ExactGPModel(x_lofi_scaled, y_lofi_scaled, likelihood)

## Find optimal model hyperparameters
model_1.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model_1.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
# Need to setoutput scale here and fix it.
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model_1)
iter = 3000
for i in range(iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model_1(x_lofi_scaled)
    # Calc loss and backprop gradients
    loss = -mll(output, y_lofi_scaled)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, iter, loss.item(),
        model_1.covar_module.base_kernel.lengthscale.item(),
        model_1.likelihood.noise.item()
    ))
    optimizer.step()

# Set model to eval mode
model_1.eval()
likelihood.eval()
# Make on the hifi inputs in model_1
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_preds_1 = likelihood(model_1(x_hifi_scaled))

# Ensure observed_preds_1.mean is 2D
mu1 = observed_preds_1.mean.unsqueeze(-1)  # shape [n, 1] if needed
# Concatenate hi-fi inputs with low-fidelity predictions
X2 = torch.cat([x_hifi_scaled, mu1], dim=1)  # shape [n, d+1]
# Save model
torch.save(model_1, 'perdikaris_2017_GP_model_1')

# Train model 2
# Initialise likilihood.
# I think I leave this as is.
model_2 = ExactGPModel(X2, y_hifi_scaled, likelihood)
d = x_hifi_data.shape[1]  # number of original input dimensions
model_2.covar_module = gpytorch.kernels.ScaleKernel(
    gpytorch.kernels.RBFKernel(ard_num_dims=1, active_dims=torch.tensor([d])) *
    gpytorch.kernels.RBFKernel(ard_num_dims=d, active_dims=torch.arange(d)) +
    gpytorch.kernels.RBFKernel(ard_num_dims=d, active_dims=torch.arange(d)))

# Find optimal model hyperparameters
model_2.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model_2.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model_2)
iter = 3000
for i in range(iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model_2(X2)
    # Calc loss and backprop gradients
    loss = -mll(output, y_hifi_scaled)
    loss.backward()
    # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
    #     i + 1, iter, loss.item(),
    #     model_2.covar_module.base_kernel.lengthscale.item(),
    #     model_2.likelihood.noise.item()
    # ))
    optimizer.step()

## Make predictions
model_2.eval()
likelihood.eval()

# Save the models
torch.save(model_2, 'perdikaris_2017_GP_model_2')

model_1 = torch.load('perdikaris_2017_GP_model_1')
model_2 = torch.load('perdikaris_2017_GP_model_2')

# Make evaluations on model_1 then concatenate mu predictions with
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_preds_2 = likelihood(model_1(x_test_scaled))
    m1 = observed_preds_2.mean.unsqueeze(-1)  # shape [n, 1] if needed
    # Concatenate test inputs with low-fidelity predictions
    X_test = torch.cat([x_test_scaled, m1], dim=1)  # shape [n, d+1]
    # Make predictions
    observed_preds_2 = likelihood(model_2(X_test))
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x_test_scaled[:, 0], x_test_scaled[:, 1], observed_preds_2.mean.numpy(), c='b', s=scatter_size)
    #ax.scatter(x_hifi_data[:, 0], x_hifi_data[:, 1], y_hifi_data.numpy(), c='r', s=scatter_size)
    ax.scatter(x_test_scaled[:, 0], x_test_scaled[:, 1], y_test_scaled.numpy(), c='g', s=scatter_size)
    ax.set_xlabel('X1', fontsize=axis_fontsize)
    ax.set_ylabel('X2', fontsize=axis_fontsize)
    ax.set_zlabel('Y', fontsize=axis_fontsize)
    ax.set_title('Test and Hi-Fi Data Perdikaris', fontsize=title_fontsize)
    ax.legend(fontsize=legend_fontsize)
    plt.show()

    # f, ax = plt.subplots(1, 1, figsize=(4, 3))
    # # Get upper and lower confidence bounds
    # lower, upper = observed_preds_2.confidence_region()
    # # Plot lofi training data as black stars
    # ax.plot(x_lofi_data.numpy(), y_lofi_data.numpy(), 'k*')
    # # Plot hifi training data as red stars
    # ax.plot(x_hifi_data.numpy(), y_hifi_data.numpy(), 'r*')
    # # Plot predictive means as blue line
    # ax.plot(x_test_data.numpy(), observed_preds_2.mean.numpy(), 'b')
    # # Shade between the lower and upper confidence bounds
    # ax.fill_between(x_test_data.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    # ax.set_ylim([-3, 3])
    # ax.legend(['Observed Data', 'Mean', 'Confidence'])
    # plt.show()

mse_nn = torch.mean((observed_preds_2.mean - y_test_scaled.numpy()) ** 2)
print(f'MSE of Neural Network on Test Data: {mse_nn.item():.4f}')