import torch
from torch.utils.data import Dataset

# Removed `convert_feature_to_prompt_text` as it's LLM-specific.
# Numerical features will be directly processed.

class MLPCSVTrafficDataset(Dataset):
    """
    A PyTorch Dataset for traffic data, designed to load pre-processed
    numerical features for MLP models.
    """
    def __init__(self, data_list):
        """
        Initializes the dataset with a list of pre-processed data samples.
        Each sample in `data_list` should be a dictionary containing:
        - 'features': a torch.Tensor of numerical feature vectors
        - 'labels': a torch.Tensor (long) for local (base) class index
        - 'global_labels': a torch.Tensor (long) for global class index
        """
        self.data_list = data_list

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the given index.
        """
        return self.data_list[idx]
    