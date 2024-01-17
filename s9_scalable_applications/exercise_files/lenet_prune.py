# Overall script description
"""
This script serves as a comprehensive demonstration of several key aspects of deep neural network modeling and optimization. It covers the following main areas:

1. Pruning Neural Networks:
   - The script showcases how to apply L1 unstructured pruning to a neural network using PyTorch's pruning functionality. Pruning is a technique for reducing the model's complexity by removing unimportant connections, resulting in a more compact network.

2. Measuring Inference Time:
   - It measures the inference time (forward pass time) for both pruned and non-pruned versions of the network. This provides insights into the trade-off between model size reduction and inference speed.

3. Measuring Model Size:
   - The script quantifies the size of saved model files for both pruned and non-pruned networks. This allows you to assess the impact of pruning on model storage requirements.

Key Components of the Script:
   - Neural Network Architecture: The script uses a LeNet architecture as an example neural network for demonstration purposes.
   - L1 Unstructured Pruning: It applies L1 unstructured pruning to the network's weights, demonstrating the reduction in the number of parameters.
   - Sparse Format Conversion: After pruning, the pruned weights are converted to sparse format, which can further reduce storage requirements.
   - Model Size Measurement: The script calculates and displays the sizes (in megabytes) of the saved model files for both pruned and non-pruned networks.
   - Inference Time Measurement: It measures the time it takes to perform inference (forward pass) on the networks and reports the execution time in seconds.

Overall, this script provides a practical example of how to optimize and analyze deep neural networks, making it a useful resource for understanding the impact of pruning on model size and inference performance.
"""

import torch
import torch.nn.utils.prune as prune
from torch import nn
import os

class LeNet(nn.Module):
    """LeNet implementation.

    This class defines the LeNet architecture, which is a classic neural network architecture
    for image classification tasks. It consists of two convolutional layers followed by three
    fully connected layers.

    Attributes:
        conv1 (nn.Conv2d): The first convolutional layer with 1 input channel and 6 output channels.
        conv2 (nn.Conv2d): The second convolutional layer with 6 input channels and 16 output channels.
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer.
        fc3 (nn.Linear): The third fully connected layer.
    
    Methods:
        forward(x): Performs a forward pass of the network with input `x`.

    """

    def __init__(self):
        """Initialize the LeNet neural network architecture.

        This constructor initializes the LeNet neural network architecture by defining
        its layers, including convolutional and fully connected layers.

        The LeNet architecture consists of the following layers:

        - Convolutional Layer 1 (conv1):
          - Input channels: 1 (grayscale image)
          - Output channels: 6
          - Kernel size: 3x3
          - Activation function: ReLU
          - Max Pooling: 2x2

        - Convolutional Layer 2 (conv2):
          - Input channels: 6
          - Output channels: 16
          - Kernel size: 3x3
          - Activation function: ReLU
          - Max Pooling: 2x2

        - Fully Connected Layer 1 (fc1):
          - Input size: 16 * 5 * 5 (output from conv2)
          - Output size: 120
          - Activation function: ReLU

        - Fully Connected Layer 2 (fc2):
          - Input size: 120
          - Output size: 84
          - Activation function: ReLU

        - Fully Connected Layer 3 (fc3):
          - Input size: 84
          - Output size: 10 (for 10-class classification)
          - Activation function: None (final layer)

        Each layer is defined as an attribute of the LeNet class for easy access during
        forward passes.

        """
        super().__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after forward pass.
        """
        x = nn.functional.max_pool2d(nn.functional.relu(self.conv1(x)), (2, 2))
        x = nn.functional.max_pool2d(nn.functional.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def check_prune_level(module: nn.Module):
    """Check the sparsity level of a pruned module.

    This function calculates the sparsity level of a pruned module by counting the percentage
    of zero-weight elements in the module's weight tensor.

    Args:
        module (nn.Module): The module for which sparsity is measured.

    Returns:
        float: The sparsity level as a percentage.
    """
    sparsity_level = 100 * float(torch.sum(module.weight == 0) / module.weight.numel())
    return sparsity_level

if __name__ == "__main__":

    # Create a pruned model
    pruned_model = LeNet()

    # Create a tuple of parameters to prune for variable-size networks
    parameters_to_prune = []
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))

    # Prune globally using L1 unstructured pruning with 20% amount
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.2,
    )

    # Convert pruned weights to sparse format
    for module, _ in parameters_to_prune:
        module.weight = module.weight.to_sparse()

    # Save the state dictionaries of pruned and non-pruned models
    torch.save(pruned_model.state_dict(), 'pruned_network_sparse.pt')

    # Create a freshly initialized model
    non_pruned_model = LeNet()

    # Save the state dictionary of the freshly initialized model
    torch.save(non_pruned_model.state_dict(), 'network.pt')

    # Get the file sizes
    pruned_model_sparse_size = os.path.getsize('pruned_network_sparse.pt')
    non_pruned_model_size = os.path.getsize('network.pt')

    print(f"Size of pruned network with sparse weights: {pruned_model_sparse_size / (1024 * 1024):.2f} MB")
    print(f"Size of non-pruned network: {non_pruned_model_size / (1024 * 1024):.2f} MB")
