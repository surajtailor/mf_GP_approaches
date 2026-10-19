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
hifi_regions = [[(edges[i], edges[i+1]), (edges[j], edges[j+1])]for i in range(n) for j in range(n)]

# Parameters
n_clusters = 10 # how many clusters to attempt
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

# ############# METHOD ####################
# # Step 1: Train a neural network on the low-fidelity data
# # Define the neural network model
class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def representation(self, x):
        """First-layer representation only, for use as a GP feature extractor."""
        return self.fc1(x)

# # # Define input and output sizes
# input_size = x_lofi_scaled.shape[1]  # Number of features in low-fidelity data#
# output_size = 1  # Assuming a single output for regression
# # Initialize the neural network
# nn_model = NeuralNetwork(input_size, output_size)
# # Define loss function and optimizer
# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)
#
# # Train the neural network on low-fidelity data
# num_epochs = 50000
# for epoch in range(num_epochs):
#     nn_model.train()
#     optimizer.zero_grad()
#     outputs = nn_model(x_lofi_scaled)
#     loss = criterion(outputs, y_lofi_scaled.unsqueeze(-1))  # Ensure y_lofi_scaled is 2D
#     loss.backward()
#     optimizer.step()
#     if (epoch + 1) % 10 == 0:
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
#
# # Save the trained neural network model
# torch.save(nn_model, 'dehghani_2018_nn_model_step_1')
#
# # --- Freeze the NN so its first-layer representation is fixed for GP training ---
# nn_model.eval()
# for param in nn_model.parameters():
#     param.requires_grad = False
#
# # Extract frozen first-layer representations of the hi-fi inputs
# with torch.no_grad():
#     x_hifi_repr = nn_model.representation(x_hifi_scaled)
# #
# # # Step 2: Train a Gaussian Process on the NN's representation of the high-fidelity data
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1])
        )
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
#
likelihood = gpytorch.likelihoods.GaussianLikelihood()
# gp_model = ExactGPModel(x_hifi_repr, y_hifi_scaled, likelihood)
#
# gp_model.train()
# likelihood.train()
# optimizer = torch.optim.Adam(gp_model.parameters(), lr=0.001)
# mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)
# iter = 50000
# for i in range(iter):
#     optimizer.zero_grad()
#     output = gp_model(x_hifi_repr)
#     loss = -mll(output, y_hifi_scaled)
#     loss.backward()
#     print('Iter %d/%d - Loss: %.3f   noise: %.3f' % (
#         i + 1, iter, loss.item(),
#         gp_model.likelihood.noise.item()
#     ))
#     optimizer.step()
#
# # Save GP model
# torch.save(gp_model, 'dehghani_2018_gp_model_step_2')

nn_model = torch.load('dehghani_2018_nn_model_step_1')
gp_model = torch.load('dehghani_2018_gp_model_step_2')

# Step 3: Evaluate validation points on the GP and use the uncertainty to weight the learning of the neural network loss function
def evaluate_gp(gp_model, nn_model, x_test):
    gp_model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        x_test_repr = nn_model.representation(x_test)  # map through frozen 1st layer first
        gp_predictions = gp_model(x_test_repr)
        gp_mean = gp_predictions.mean
        gp_variance = gp_predictions.variance
    return gp_mean, gp_variance
#
# # Evaluate using a concatenation of the Hi-Fi and Lo-Fi training points
hi_fi_lo_fi_concat = torch.cat([x_hifi_scaled, x_lofi_scaled], dim=0)
gp_mean, gp_variance = evaluate_gp(gp_model, nn_model, hi_fi_lo_fi_concat)

# --- Unfreeze the NN before retraining it in Step 4 ---
for param in nn_model.parameters():
    param.requires_grad = True

from torch.utils.data import TensorDataset, DataLoader

# Step 4: Use the GP predictions to weight the neural network loss function and retrain (batch-wise)
def weighted_loss(nn_model, x_train, gp_mean, gp_variance, num_epochs=50000, beta=1, batch_size=64):
    nn_model.train()
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)

    # Bundle inputs + GP targets so each batch stays aligned
    dataset = TensorDataset(x_train, gp_mean, gp_variance)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for x_batch, mean_batch, var_batch in loader:
            optimizer.zero_grad()
            outputs = nn_model(x_batch)  # full forward pass, not .representation()
            weights = torch.exp(-beta * var_batch)
            loss = torch.mean(weights * (outputs - mean_batch.unsqueeze(-1)) ** 2)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Avg Weighted Loss: {avg_loss:.4f}')

# Retrain the neural network with weighted loss
weighted_loss(nn_model, hi_fi_lo_fi_concat, gp_mean, gp_variance)

# Save the retrained neural network model
torch.save(nn_model, 'dehghani_2018_nn_model_step_3_beta_1')
#
# # Step 5: Evaluate the final model on the test set
def evaluate_final_model(nn_model, x_test):
    nn_model.eval()
    with torch.no_grad():
        predictions = nn_model(x_test)
    return predictions

# Load the step 1 neural network model
# step_1_nn = torch.load('dehghani_2018_nn_model_step_1')
# nn_model = torch.load('dehghani_2018_nn_model_step_3')
# # Evaluate the final model
# final_predictions = evaluate_final_model(nn_model, x_test_scaled)
# original_predictions = evaluate_final_model(step_1_nn, x_test_scaled)
#
# # Compare MSE of GP on Hi-Fi data with MSE of nn_model
# #mse_gp = torch.mean((gp_mean - y_test_scaled.unsqueeze(-1)) ** 2)
# mse_nn = torch.mean((final_predictions - y_test_scaled.unsqueeze(-1)) ** 2)
# #print(f'MSE of GP on Hi-Fi data: {mse_gp.item():.4f}')
# print(f'MSE of Neural Network on Test Data: {mse_nn.item():.4f}')
#
# # Plot the predictions in a 3d plot
# def plot_predictions(x_test, y_test, predictions):
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(projection='3d')
#     ax.scatter(x_test[:, 0], x_test[:, 1], y_test, c='r', s=scatter_size, label='True Values')
#     #ax.scatter(x_test[:, 0], x_test[:, 1], y_test, c='r', s=scatter_size, label='True Values')
#     ax.scatter(x_test[:, 0], x_test[:, 1], predictions.squeeze().numpy(), c='b', s=scatter_size, label='Predictions')
#     #ax.scatter(x_test[:, 0], x_test[:, 1], original_predictions.squeeze().numpy(), c='g', s=scatter_size, label='Step 1 Predictions')
#     #ax.scatter(x_test[:, 0], x_test[:, 1], gp_mean.squeeze().numpy(), c='black', s=scatter_size, label='GP Predictions')
#     ax.set_xlabel('X1', fontsize=axis_fontsize)
#     ax.set_ylabel('X2', fontsize=axis_fontsize)
#     ax.set_zlabel('Y', fontsize=axis_fontsize)
#     ax.set_title('3D Plot of Predictions vs True Values Dehghani', fontsize = title_fontsize)
#     ax.legend(fontsize=legend_fontsize)
#     plt.show()
#
# plot_predictions(x_test_scaled, y_test_scaled, final_predictions)