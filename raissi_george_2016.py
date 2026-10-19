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

# NN output size
nn_output_size = 4

################# METHOD ####################
class MultiFidelityKernel(gpytorch.kernels.Kernel):
    def __init__(self, base_kernel, delta_kernel, **kwargs):
        super().__init__(has_lengthscale=False)
        self.base_kernel = base_kernel
        self.delta_kernel = delta_kernel
        self.register_parameter(name="raw_rho", parameter=torch.nn.Parameter(torch.tensor(0.0)))

    @property
    def rho(self):
        return self.raw_rho

    @rho.setter
    def rho(self, value):
        self.raw_rho.data = torch.as_tensor(value)

    def forward(self, x1, x2, diag = False , **params):
        # Assume x1, x2 are [N, D+1] where last dim is fidelity flag (0 or 1)
        X1, f1 = x1[:, :-1], x1[:, -1]
        X2, f2 = x2[:, :-1], x2[:, -1]
        k_l = self.base_kernel(X1, X2)
        k_d = self.delta_kernel(X1, X2)

        f1 = f1.view(-1, 1)
        f2 = f2.view(1, -1)

        # add the output scale in the first term??
        # Base correlation structure
        k = k_l*((f1 == 0)&(f2 == 0)).float()  # Lo-Fi to Lo-Fi
        k += self.rho * k_l*((f1 == 0)&(f2 == 1)).float()  # Lo-Fi to Hi-Fi
        k += self.rho * k_l*((f1 == 1)&(f2 == 0)).float()  # Hi-Fi to Lo-Fi
        k += self.rho**2 * k_l*((f1 == 1)&(f2 == 1)).float() + k_d*((f1 == 1)&(f2 == 1)).float()  # Hi-Fi to Hi-Fi

        if diag:
            k = k[0]
        return k

# Simplest form of GP model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, base_kernel, delta_kernel, nn_output_size):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            MultiFidelityKernel(base_kernel, delta_kernel))
        self.neural_network = NN(input_size=(train_x.shape[1]-1), output_size=nn_output_size)
    def forward(self, x):
        x_nn = self.neural_network(x[:, :-1])  # Exclude the last column (fidelity flag) for NN input
        # concatenate back with fidelity flag
        x_nn = torch.cat([x_nn, x[:, -1:]], dim=1)  # Keep the fidelity flag
        mean_x = self.mean_module(x_nn)
        covar_x = self.covar_module(x_nn)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class NN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(NN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# x labeled
x_labeled = torch.cat([
    torch.cat([x_lofi_scaled, torch.zeros(x_lofi_scaled.shape[0], 1)], dim=1),
    torch.cat([x_hifi_scaled, torch.ones(x_hifi_scaled.shape[0], 1)], dim=1)])

# y combined
y_combined = torch.cat([y_lofi_scaled, y_hifi_scaled])

# Initialise likelihood.
likelihood = gpytorch.likelihoods.GaussianLikelihood()

# Create model 1 on Lo-Fi data
# On the base kernel, we set our output_scale
model = ExactGPModel(
    x_labeled, y_combined, likelihood,
    base_kernel=gpytorch.kernels.RBFKernel(ard_num_dims=nn_output_size),
    delta_kernel=gpytorch.kernels.RBFKernel(ard_num_dims=nn_output_size),
    nn_output_size=nn_output_size
)

# Find optimal model hyperparameters
model.train()
likelihood.train()
# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Includes GaussianLikelihood parameters
# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

iter = 20000
# Training loop
for i in range(iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(x_labeled)
    # Calc loss and backprop gradients
    loss = -mll(output, y_combined)
    loss.backward()
    print(f"Iter {i + 1} - Loss: {loss.item():.3f} - Rho: {model.covar_module.base_kernel.rho.item():.3f}")
    optimizer.step()

torch.save(model, 'raissi_george_2016_GP_2')

# # Set model to eval mode
# model.eval()
# likelihood.eval()
# x_star = torch.cat([x_test_scaled, torch.ones(x_test_scaled.shape[0], 1)], dim=1)  # Mark as high-fidelity
# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     preds = likelihood(model(x_star))
#     mean = preds.mean.unsqueeze(-1)  # shape [n, 1] if needed
#     #var = preds.variance
#     # Create a 3D plot
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(projection='3d')
#     ax.scatter(x_test_scaled[:, 0], x_test_scaled[:, 1], mean.squeeze().numpy(), c='b', s=scatter_size)
#     ax.scatter(x_test_scaled[:, 0], x_test_scaled[:, 1], y_test_scaled.numpy(), c='g', s=scatter_size)
#     ax.set_xlabel('X1', fontsize=axis_fontsize)
#     ax.set_ylabel('X2', fontsize=axis_fontsize)
#     ax.set_zlabel('Y', fontsize=axis_fontsize)
#     ax.set_title('Test and Hi-Fi Data George', fontsize=title_fontsize)
#     ax.legend(fontsize=legend_fontsize)
#     plt.show()