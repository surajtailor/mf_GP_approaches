import torch
from statsmodels.nonparametric.kde import kernel_switch
from data_generation import generate_lofi_data, generate_hifi_data, generate_test_data, create_scalers, scale_data
import random
import matplotlib.pyplot as plt
import numpy as np
import gpytorch
from utility import unconcatenate_mu, unconcatenate_var
from models.dehghani_model import DehghaniNN
from models.perdikaris_model import PerdikarisGPModel
from models.raissi_model import RaissiGPModel, RaissiFeatureNN, MultiFidelityKernel
from models.base_models import BaseGPModel, BaseNNModel
from models.net import NetAnyFunctional2D
import sys
sys.path.append(r"C:\Users\Suraj\Documents\01_git_repositories\01_phd_projects\GP_MF_Methods")


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

# Trick pickle into finding it under the name/module it was saved with
sys.modules['__main__'].NeuralNetwork = DehghaniNN
dehghani_NN_model_1 = torch.load("dehghani_2018_nn_model_step_1")
dehghani_NN_model = torch.load("dehghani_2018_nn_model_step_3_beta_1")

sys.modules['__main__'].ExactGPModel = PerdikarisGPModel
perdikaris_model_1 = torch.load("perdikaris_2017_GP_model_1")
perdikaris_model_2 = torch.load("perdikaris_2017_GP_model_2")

# reassign ExactGPModel to the Raissi/George version before this load
sys.modules['__main__'].ExactGPModel = RaissiGPModel
sys.modules['__main__'].NN = RaissiFeatureNN
sys.modules['__main__'].MultiFidelityKernel = MultiFidelityKernel
raissi_GP_model = torch.load("raissi_george_2016_GP_4")

# Kernel based approach
sys.modules['__main__'].NetAnyFunctional2D = NetAnyFunctional2D
kernel_GP_model = NetAnyFunctional2D(input_size=2, num_layers=6, hidden_size=20, device='cuda',
                           functional='polynomial', poly_degree= 2, fourier_degree=None, poly_l_degree=None,
                           custom=False, activation_func='elu', interpolate_H=True, spline_type="CT").to(device)  # reconstruct with the same architecture args
kernel_GP_model.load_state_dict(torch.load("kernel_GP.pt"))
kernel_GP_model._create_interpolate_H(hifi_train_data)

# Base models
base_GP_model = torch.load("base_GP_model")
base_NN_model = torch.load("base_NN_model")

# Initialise the likelihood for the GP models
likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

### George GP model
raissi_GP_model.eval()
likelihood.eval()
x_star = torch.cat([x_test_scaled, torch.ones(x_test_scaled.shape[0], 1)], dim=1)  # Mark as high-fidelity
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    preds_raissi = likelihood(raissi_GP_model(x_star))
    mean_pred_raissi = preds_raissi.mean.unsqueeze(-1) # shape [n, 1] if needed
    var_pred_raissi = preds_raissi.variance.unsqueeze(-1) # shape [n, 1] if needed
    #unscale
    mean_pred_unscaled_raissi = y_Global_Scaler.inverse_transform(mean_pred_raissi)
    var_pred_unscaled_raissi = y_Global_Scaler.inverse_transform_variance(var_pred_raissi)

### Perdikaris GP model
# Set model to eval mode
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    preds_perdi_1 = likelihood(perdikaris_model_1(x_test_scaled))
    m1 = preds_perdi_1.mean.unsqueeze(-1)  # shape [n, 1] if needed
    X_test = torch.cat([x_test_scaled, m1], dim=1)  # shape [n, d+1]
    preds_perdi_2 = likelihood(perdikaris_model_2(X_test))
    mean_pred_perdi = preds_perdi_2.mean.unsqueeze(-1)  # shape [n, 1] if needed
    var_pred_perdi = preds_perdi_2.variance.unsqueeze(-1)  # shape [n, 1] if needed
    #Unscale
    mean_pred_unscaled_perdi = y_Global_Scaler.inverse_transform(mean_pred_perdi)
    var_pred_unscaled_perdi = y_Global_Scaler.inverse_transform_variance(var_pred_perdi)
    m1_pred_unscaled_perdi = y_Global_Scaler.inverse_transform(m1)

## Deghani NN model
dehghani_NN_model.eval()
dehghani_NN_model_1.eval()
with torch.no_grad():
    mean_pred_deghani = dehghani_NN_model(x_test_scaled)
    mean_pred_deghani_1 = dehghani_NN_model_1(x_test_scaled)
    #unscale
    mean_pred_unscaled_dehghani = y_Global_Scaler.inverse_transform(mean_pred_deghani)
    mean_pred_unscaled_dehghani_1 = y_Global_Scaler.inverse_transform(mean_pred_deghani_1)

## Kernel GP model
kernel_GP_model.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    mu, var = kernel_GP_model(x_test_scaled)
    mu_H, mu_L, mu_D, H = unconcatenate_mu(mu)
    var_H, var_L, var_D = unconcatenate_var(var)

    # Unscale stuff
    mu_H = y_Global_Scaler.inverse_transform(mu_H)
    mu_L = y_Global_Scaler.inverse_transform(mu_L)
    mu_D = y_Global_Scaler.inverse_transform_mu_D(mu_D)
    H = y_Global_Scaler.inverse_transform(H)
    var_H = y_Global_Scaler.inverse_transform_variance(var_H)

    # Detach tensors before conversion to NumPy
    mu_H = mu_H.detach()
    mu_L = mu_L.detach()
    mu_D = mu_D.detach()
    H = H.detach()
    var_H = var_H.detach()

## Base GP
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    base_GP_model.eval()
    preds_base_GP = likelihood(base_GP_model(x_test_scaled))
    mean_pred_base_GP = preds_base_GP.mean.unsqueeze(-1)  # shape [n, 1] if needed
    var_pred_base_GP = preds_base_GP.variance.unsqueeze(-1)  # shape [n, 1] if needed
    # Unscale
    mean_pred_unscaled_base_GP = y_Global_Scaler.inverse_transform(mean_pred_base_GP)
    var_pred_unscaled_base_GP = y_Global_Scaler.inverse_transform_variance(var_pred_base_GP)

## Base NN
base_NN_model.eval()
with torch.no_grad():
    mean_pred_base_NN = base_NN_model(x_test_scaled)
    mean_pred_unscaled_base_NN = y_Global_Scaler.inverse_transform(mean_pred_base_NN)

## Calculate the MSE
mse_perdi = torch.mean((mean_pred_unscaled_perdi.squeeze() - y_test_data)**2)
mse_raissi = torch.mean((mean_pred_unscaled_raissi.squeeze() - y_test_data) ** 2)
mse_dehghani = torch.mean((mean_pred_unscaled_dehghani.squeeze() - y_test_data)**2)
mse_dehghani_1 = torch.mean((mean_pred_unscaled_dehghani_1.squeeze() - y_test_data)**2)
mse_perdi_m1 = torch.mean((m1_pred_unscaled_perdi.squeeze() - y_test_data)**2)
mse_base_GP =  torch.mean((mean_pred_unscaled_base_GP.squeeze() - y_test_data)**2)
mse_base_NN = torch.mean((mean_pred_unscaled_base_NN.squeeze() - y_test_data)**2)

# Print the MSEs
print(f"MSE for Base GP model: {mse_base_GP.item():.6f}")
print(f"MSE for Base NN model: {mse_base_NN.item():.6f}")
print(f"MSE for Perdikaris GP model (first stage): {mse_perdi_m1.item():.6f}")
print(f"MSE for Perdikaris GP model: {mse_perdi.item():.6f}")
print(f"MSE for Raissi GP model: {mse_raissi.item():.6f}")
print(f"MSE for Dehghani NN model: {mse_dehghani.item():.6f}")
print(f"MSE for Dehghani NN model (first stage): {mse_dehghani_1.item():.6f}")
print(f"MSE for Kernel GP model: {torch.mean((mu_H - y_test_data)**2).item():.6f}")

# Plot each of the models predictions against the true test data create a plotting function 3d plot
def plot_predictions(x_test_scaled, y_test_scaled, mean_pred_unscaled, title):
    plt.figure(figsize=(8, 6))
    ax = plt.axes(projection='3d')
    ax.scatter(x_test_scaled[:, 0], x_test_scaled[:, 1], y_test_scaled, c='g', s=scatter_size, label='Ground Truth')
    ax.scatter(x_test_scaled[:, 0], x_test_scaled[:, 1], mean_pred_unscaled.squeeze(), c='b', s=scatter_size, label='Predictions')
    ax.set_xlabel('$x_1$', fontsize=axis_fontsize)
    ax.set_ylabel('$x_2$', fontsize=axis_fontsize)
    #ax.set_title(title, fontsize=title_fontsize)
    #ax.legend(fontsize=legend_fontsize)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=500)

# Plot the predictions for each model
plot_predictions(x_test_data, y_test_data, mean_pred_unscaled_base_GP, 'Base GP Model Predictions')
plot_predictions(x_test_data, y_test_data, mean_pred_unscaled_base_NN, 'Base NN Model Predictions')
plot_predictions(x_test_data, y_test_data, mean_pred_unscaled_perdi, 'Perdikaris GP Model Predictions')
plot_predictions(x_test_data, y_test_data, mean_pred_unscaled_raissi, 'Raissi GP Model Predictions')
plot_predictions(x_test_data, y_test_data, mean_pred_unscaled_dehghani, 'Dehghani NN Model Predictions')
plot_predictions(x_test_data, y_test_data, mean_pred_unscaled_dehghani_1, 'Dehghani NN Model Predictions (First Stage)')
plot_predictions(x_test_data, y_test_data, mu_H, 'Kernel GP Model Predictions')

def plot_uncertainty(X1, X2, var_flat, x_hifi, title, dpi=1000):
    """
    Plot predictive uncertainty (variance) over a 2D input grid, with
    Hi-Fi training points overlaid.

    Parameters
    ----------
    X1, X2 : 2D meshgrid arrays (numpy), shape (n_grid, n_grid)
    var_flat : flat per-point variance tensor/array, shape (n_grid*n_grid,) or (n_grid*n_grid, 1)
    x_hifi : Hi-Fi input points, shape (N, 2) — tensor or numpy array
    title : str, used for plot title and filename
    """
    if torch.is_tensor(var_flat):
        var_flat = var_flat.detach().cpu().numpy()
    if torch.is_tensor(x_hifi):
        x_hifi = x_hifi.detach().cpu().numpy()

    if title == "Kernel GP":
        var_flat = np.clip(var_flat, a_min=1e-10, a_max=3)

    # Reshape flat predictions back into the meshgrid shape
    uncertainty_grid = var_flat.squeeze().reshape(X1.shape)

    x1_hifi, x2_hifi = x_hifi[:, 0], x_hifi[:, 1]

    plt.figure(figsize=(6, 5))
    im = plt.contourf(X1, X2, uncertainty_grid, levels=100, cmap='viridis')
    plt.colorbar(im)
    plt.scatter(x1_hifi, x2_hifi, c='red', s=16, label='HiFi Data', alpha=0.6, edgecolors='k')
    #plt.title(f"Uncertainty $\\sigma^2_{{Y_H}}(x)$ — {title}", fontsize=12)
    plt.xlabel('$x_1$', fontsize=12)
    plt.ylabel('$x_2$', fontsize=12)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"uncertainty_plot_{title.replace(' ', '_')}.png", dpi=dpi)
    plt.close()


uncertainty_data = {
    'Kernel GP': var_H,
    'Perdikaris GP': var_pred_unscaled_perdi,
    'Raissi GP': var_pred_unscaled_raissi,
    'Base GP': var_pred_unscaled_base_GP,
}

x_hifi_train_unscaled = x_hifi_data[train_idx]

for model_name, var_flat in uncertainty_data.items():
    plot_uncertainty(x1_test_data_grid, x2_test_data_grid, var_flat, x_hifi_train_unscaled, model_name)