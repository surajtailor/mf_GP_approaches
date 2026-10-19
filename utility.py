import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from torch.utils.data import Dataset, DataLoader

class Scaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)

    def transform(self, x):
        if self.mean is None or self.std is None:
            raise ValueError("The scaler has not been fitted yet. Call 'fit' with data before transforming.")
        return (x - self.mean) / self.std

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        if self.mean is None or self.std is None:
            raise ValueError("The scaler has not been fitted yet. Call 'fit' with data before inverse transforming.")
        return x * self.std + self.mean

    def inverse_transform_mu_D(self, x):
        if self.mean is None or self.std is None:
            raise ValueError("The scaler has not been fitted yet. Call 'fit' with data before inverse transforming.")
        return x * self.std

    def inverse_transform_variance(self, x):
        return x * self.std ** 2

class GaussianKDE:
    def __init__(self, data, H, beta=1):
        # calculate the covariance matrix
        self.data = data
        self.H = H
        self.beta = beta
        assert self.is_psd(self.H), "The covariance matrix H is not positive semi-definite."

    def is_psd(self, mat):
        return bool((mat == mat.T).all() and (torch.linalg.eigvals(mat).real >= 0).all())

    def K(self, x):
        # calculate the scaling coeff to shorten the equation
        coeff = torch.linalg.det(self.H) ** (-0.5)
        # calculate the exponential term
        exp = torch.exp(-0.5 * torch.sum((x @ torch.inverse(self.H)) * x, dim=1))
        # pi term
        pi = torch.tensor(2 * torch.pi) ** (-x.shape[1] / 2)
        # return the kernel value
        return coeff * exp * pi

    def KDE(self, x):
        # prepare the grid for output values
        output = torch.zeros_like(x[:, 0])
        # process each sample
        for sample in self.data:
            output += self.K(x - sample.reshape(1, -1))
        # return the average
        output /= len(self.data)
        return output

    def translated_sigmoid(self, x, gamma=1.0):
        """ Sigmoid Transformation """
        beta_scaled = torch.exp(gamma * self.beta)
        sigma_x = 1 / (1 + torch.exp(-beta_scaled * x))
        sigma_0 = 1 / (1 + torch.exp(torch.tensor(0.0)))
        return (sigma_x - sigma_0) / (1 - sigma_0)

    def transformed_output(self, x):
        """ Apply KDE and transformation """
        x = self.KDE(x)
        return self.translated_sigmoid(x)

    def __call__(self, x):
        return self.KDE(x)

def sample_bins_k_fold(x_data, z_data, x_data_scaled, z_data_scaled, bin_width, domain, train_ratio, K, enforce_unique_bins=False, device="cpu"):
    x_data = x_data.to(device)
    z_data = z_data.to(device)

    # Define bin edges
    x_bins = torch.arange(domain[0], domain[1] + bin_width[0], bin_width[0], device=device)

    # Digitize data points into bins
    bin_indices = torch.bucketize(x_data[:, 0], x_bins) - 1

    # Create dictionary mapping (bin_x) -> list of indices
    bin_dict = {}
    for i, bx in enumerate(bin_indices):
        key = bx.item()
        if key not in bin_dict:
            bin_dict[key] = []
        bin_dict[key].append(i)

    # Convert dict values to tensors
    for key in bin_dict:
        bin_dict[key] = torch.tensor(bin_dict[key], device=device)

    # Ensure all possible bins are represented in `bin_dict`
    valid_bins = list(bin_dict.keys())
    for b in range(len(x_bins)):  # Iterate over all possible bins
        if b not in bin_dict:
            bin_dict[b] = torch.tensor([], device=device)  # Empty tensor for missing bins

    total_bins = len(valid_bins)

    # Determine number of bins to sample per fold
    train_samp_num = int(train_ratio * total_bins)

    if enforce_unique_bins and train_samp_num * K > total_bins:
        raise ValueError(
            f"Not enough bins ({total_bins}) to allocate {K} unique folds with {train_samp_num} bins each."
        )

    # List to store K different train/val splits + selected bins
    k_folds = []
    k_folds_scaled = []
    selected_bins_list = []
    used_bins = set() if enforce_unique_bins else None  # Track bins only if enforcing uniqueness

    for fold in range(K):
        if enforce_unique_bins:
            # Get available bins (not used in previous folds)
            remaining_bins = [b for b in valid_bins if b not in used_bins]

            # Ensure there are enough bins left
            if len(remaining_bins) < train_samp_num:
                print(f"Skipping fold {fold + 1} due to not enough bins left.")
                continue  # Skip this fold if not enough bins remain

            # Sample only from available (non-empty, unused) bins
            sampled_bins = random.sample(remaining_bins, train_samp_num)
            used_bins.update(sampled_bins)  # Mark bins as used
        else:
            # Allow reuse of bins across folds
            sampled_bins = random.sample(valid_bins, min(train_samp_num, len(valid_bins)))

        selected_bins_list.append(sampled_bins)

        # Gather indices from sampled bins
        selected_indices_list = [bin_dict[b] for b in sampled_bins if b in bin_dict and bin_dict[b].numel() > 0]

        if not selected_indices_list:  # Check if the list is empty before concatenation
            print(f"Skipping fold {fold + 1} due to empty sampled bins: {sampled_bins}")
            continue

        selected_indices = torch.cat(selected_indices_list)

        # Create a mask for all selected indices
        mask = torch.ones(len(x_data), dtype=torch.bool, device=device)
        mask[selected_indices] = False  # Mark training data as False

        # Get remaining indices for validation
        val_indices = torch.arange(len(x_data), device=device)[mask]

        # Store split
        k_folds.append({
            "train": (x_data[selected_indices], z_data[selected_indices]),
            "val": (x_data[val_indices], z_data[val_indices]),
        })

        # Store split for scaled data
        k_folds_scaled.append({
            "train": (x_data_scaled[selected_indices], z_data_scaled[selected_indices]),
            "val": (x_data_scaled[val_indices], z_data_scaled[val_indices]),
        })

    return selected_bins_list, k_folds, k_folds_scaled

def plot_k_folds(k_folds):
    """Plots K different train-validation splits."""
    K = len(k_folds)
    fig, axes = plt.subplots(1, K, figsize=(5 * K, 5))

    if K == 1:
        axes = [axes]

    for i, fold in enumerate(k_folds):
        x_train, y_train = fold["train"]
        x_val, y_val = fold["val"]

        axes[i].scatter(x_train, y_train, color="blue", label="Train", alpha=0.6)
        axes[i].scatter(x_val, y_val, color="red", label="Remaining", alpha=0.6)
        axes[i].set_title(f"Fold {i + 1}")
        axes[i].set_xlabel("X")
        axes[i].set_ylabel("Y")
        axes[i].legend()
    plt.tight_layout()
    plt.show()

def unconcatenate_mu(x):
    mu_H = x[:, 0]
    mu_L = x[:, 1]
    mu_D = x[:, 2]
    H = x[:, 3]
    return mu_H, mu_L, mu_D, H

def unconcatenate_var(x):
    var_H = x[:, 0]
    var_L = x[:, 1]
    var_D = x[:, 2]
    return var_H, var_L, var_D

def beta_calculator(alpha, mode):
    beta = (alpha + 1) * mode
    return beta

class IndexedDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], idx  # return index too

def plot_3d_data(x_data, y_data, title="Data Plot"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_data[:, 0], x_data[:, 1], y_data, s=8)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    ax.set_title(title)
    plt.show()

def split_into_domain_bins_1D(x_data, z_data, x_data_scaled, z_data_scaled, n_bins, domain, device="cpu"):
    """
    Splits (x_data, z_data) into bins based on equal-width splits along x₁ only.

    Args:
        x_data (torch.Tensor): Input data (N, 2).
        z_data (torch.Tensor): Output data (N, targets).
        x_data_scaled (torch.Tensor): Scaled input data.
        z_data_scaled (torch.Tensor): Scaled output data.
        n_bins (int): Number of bins along x₁.
        domain (tuple): (min_x1, max_x1) domain for x₁ dimension.
        device (str): Device to put output tensors on.

    Returns:
        bins (list): List of (x_bin, z_bin) tuples.
        bins_scaled (list): List of (x_bin_scaled, z_bin_scaled) tuples.
    """

    x_data = x_data.to(device)
    z_data = z_data.to(device)
    x_data_scaled = x_data_scaled.to(device)
    z_data_scaled = z_data_scaled.to(device)

    # Define bin edges along x₁
    bin_edges = torch.linspace(domain[0], domain[1], n_bins + 1, device=device)

    # Digitize x₁ values into bins
    bin_indices = torch.bucketize(x_data[:, 1], bin_edges, right=False) - 1  # -1 to make bins 0-based

    # Clamp any points exactly at domain[1] to last bin
    bin_indices = torch.clamp(bin_indices, min=0, max=n_bins - 1)

    bins = []
    bins_scaled = []

    for i in range(n_bins):
        idx = (bin_indices == i)
        bins.append((x_data[idx], z_data[idx]))
        bins_scaled.append((x_data_scaled[idx], z_data_scaled[idx]))

    return bins, bins_scaled

def split_into_domain_bins_2D(x_data, z_data, x_data_scaled, z_data_scaled, n_bins, domain, device="cpu"):
    """
    Splits (x_data, z_data) into bins based on equal-width splits along x₁ and x₂.

    Args:
        x_data (torch.Tensor): Input data (N, 2).
        z_data (torch.Tensor): Output data (N, targets).
        x_data_scaled (torch.Tensor): Scaled input data.
        z_data_scaled (torch.Tensor): Scaled output data.
        n_bins (int): Number of bins along each axis (x₁ and x₂).
        domain (tuple): (min_x1, max_x1, min_x2, max_x2).
        device (str): Device to put output tensors on.

    Returns:
        bins (list): List of (x_bin, z_bin) tuples.
        bins_scaled (list): List of (x_bin_scaled, z_bin_scaled) tuples.
    """

    x_data = x_data.to(device)
    z_data = z_data.to(device)
    x_data_scaled = x_data_scaled.to(device)
    z_data_scaled = z_data_scaled.to(device)

    min_x1, max_x1, min_x2, max_x2 = domain

    # Define bin edges along x₁ and x₂
    x1_edges = torch.linspace(min_x1, max_x1, n_bins + 1, device=device)
    x2_edges = torch.linspace(min_x2, max_x2, n_bins + 1, device=device)

    # Digitize x₁ and x₂ values into bins
    x1_bin_indices = torch.bucketize(x_data[:, 0], x1_edges, right=False) - 1
    x2_bin_indices = torch.bucketize(x_data[:, 1], x2_edges, right=False) - 1

    # Clamp indices (handle boundary points exactly at max_x1, max_x2)
    x1_bin_indices = torch.clamp(x1_bin_indices, min=0, max=n_bins - 1)
    x2_bin_indices = torch.clamp(x2_bin_indices, min=0, max=n_bins - 1)

    bins = []
    bins_scaled = []

    for i in range(n_bins):
        for j in range(n_bins):
            idx = (x1_bin_indices == i) & (x2_bin_indices == j)
            bins.append((x_data[idx], z_data[idx]))
            bins_scaled.append((x_data_scaled[idx], z_data_scaled[idx]))

    return bins, bins_scaled

def sample_k_sets_without_replacement(x_data, z_data, x_data_scaled, z_data_scaled,n, k, device="cpu", seed=None):
    """
    Samples `k` non-overlapping sets of `n` points each from the data.

    Args:
        x_data (torch.Tensor): Raw input data (N, D).
        z_data (torch.Tensor): Raw output data.
        x_data_scaled (torch.Tensor): Scaled input data.
        z_data_scaled (torch.Tensor): Scaled output data.
        n (int): Number of samples per set.
        k (int): Number of non-overlapping sets.
        device (str): Torch device.
        seed (int, optional): Seed for reproducibility.

    Returns:
        sets (list): List of (x, z) tuples (unscaled).
        sets_scaled (list): List of (x_scaled, z_scaled) tuples.
    """
    x_data = x_data.to(device)
    z_data = z_data.to(device)
    x_data_scaled = x_data_scaled.to(device)
    z_data_scaled = z_data_scaled.to(device)

    total_points = x_data.shape[0]
    assert n * k <= total_points, "Not enough data to sample k non-overlapping sets."

    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    permuted_indices = torch.randperm(total_points, generator=generator)

    sets = []
    sets_scaled = []

    for i in range(k):
        idx = permuted_indices[i * n : (i + 1) * n]
        sets.append((x_data[idx], z_data[idx]))
        sets_scaled.append((x_data_scaled[idx], z_data_scaled[idx]))

    return sets, sets_scaled

# Plot the KDE output 3d plot
def plot_kde_3d(kde, x_test_scaled):
    kde_output = 100*kde.transformed_output(x_test_scaled).squeeze()
    # Convert to numpy for plotting
    x = x_test_scaled[:, 0].numpy()
    y = x_test_scaled[:, 1].numpy()
    z = kde_output.numpy()
    # Create a grid for surface plot
    from scipy.interpolate import griddata
    import numpy as np
    grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), 100),np.linspace(y.min(), y.max(), 100))
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
    # Plot surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none', alpha=0.9)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    plt.title("$KDE(x)$")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        # rotate the plot
    ax.view_init(elev=30, azim=30)
    plt.show()