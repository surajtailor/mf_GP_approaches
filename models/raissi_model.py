import torch
import gpytorch

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
class RaissiGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, base_kernel, delta_kernel, nn_output_size):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            MultiFidelityKernel(base_kernel, delta_kernel))
        self.neural_network = RaissiFeatureNN(input_size=(train_x.shape[1]-1), output_size=nn_output_size)
    def forward(self, x):
        x_nn = self.neural_network(x[:, :-1])  # Exclude the last column (fidelity flag) for NN input
        # concatenate back with fidelity flag
        x_nn = torch.cat([x_nn, x[:, -1:]], dim=1)  # Keep the fidelity flag
        mean_x = self.mean_module(x_nn)
        covar_x = self.covar_module(x_nn)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class RaissiFeatureNN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(RaissiFeatureNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x