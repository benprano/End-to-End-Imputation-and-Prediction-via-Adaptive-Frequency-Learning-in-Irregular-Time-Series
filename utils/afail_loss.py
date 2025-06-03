import torch
from torch import nn


class Afail(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(0.005, requires_grad=True))

    def forward(self, sampled_data, sampled_imputed_x, data_freqs, outputs, labels, criterion):
        # Calculate mean absolute error (MAE)
        loss_imp = torch.mean(torch.abs(sampled_data - sampled_imputed_x))

        # Normalize frequencies
        normalized_freqs = data_freqs / torch.max(data_freqs)

        # Apply dynamic weighting using exponential decay
        dynamic_weights = torch.exp(-self.alpha * (1 - normalized_freqs))

        # Apply weighted loss calculation
        weighted_loss_imp = loss_imp * dynamic_weights
        weighted_loss_imp = torch.sum(weighted_loss_imp) / torch.sum(dynamic_weights)

        # Calculate prediction loss
        prediction_loss = criterion(outputs.squeeze(-1), labels.squeeze(-1))

        # Combine imputation loss and prediction loss
        total_loss = prediction_loss + weighted_loss_imp
        return weighted_loss_imp, total_loss