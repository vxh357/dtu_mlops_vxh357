"""Training a Variational Autoencoder (VAE) on the MNIST dataset.

This script trains a Gaussian MLP Encoder and Decoder on the MNIST dataset.
It uses the PyTorch library for building and training the model.

References:
- Original code adapted from: https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb
"""

import logging
import os

import hydra
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model import Decoder, Encoder, Model
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="default_config.yaml")
def train(config: OmegaConf) -> None:
    """Train the VAE model on the MNIST dataset.

    Args:
        config (OmegaConf): Configuration object containing parameters for training.

    The function sets up the MNIST dataset, defines the model, and trains it.
    It saves the trained model, and generates sample reconstructions and new samples.
    """
    log.info(f"Configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config.experiment
    torch.manual_seed(hparams["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data loading
    mnist_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST(hparams["dataset_path"], transform=mnist_transform, train=True, download=True)
    test_dataset = MNIST(hparams["dataset_path"], transform=mnist_transform, train=False, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=hparams["batch_size"], shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=hparams["batch_size"], shuffle=False)

    # Model setup
    encoder = Encoder(input_dim=hparams["x_dim"], hidden_dim=hparams["hidden_dim"], latent_dim=hparams["latent_dim"])
    decoder = Decoder(latent_dim=hparams["latent_dim"], hidden_dim=hparams["hidden_dim"], output_dim=hparams["x_dim"])
    model = Model(encoder=encoder, decoder=decoder).to(device)

    from torch.optim import Adam

    def loss_function(x: torch.Tensor, x_hat: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Compute the loss function for VAE.

        Args:
            x (torch.Tensor): Original input data.
            x_hat (torch.Tensor): Reconstructed data.
            mean (torch.Tensor): Mean from the latent space.
            log_var (torch.Tensor): Log variance from the latent space.

        Returns:
            torch.Tensor: The computed loss value.
        """
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss + kld

    optimizer = Adam(model.parameters(), lr=hparams["lr"])

    # Training loop
    log.info("Start training VAE...")
    model.train()
    for epoch in range(hparams["n_epochs"]):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(hparams["batch_size"], hparams["x_dim"])
            x = x.to(device)

            optimizer.zero_grad()
            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()
            loss.backward()
            optimizer.step()
        log.info(f"Epoch {epoch+1} complete! Average Loss: {overall_loss / (batch_idx * hparams['batch_size'])}")
    log.info("Training Finished")

    # Save the trained model
    torch.save(model, f"{os.getcwd()}/trained_model.pt")

    # Generate and save reconstructions and samples
    _generate_and_save_images(model, test_loader, decoder, device, hparams)


def _generate_and_save_images(model: Model, test_loader: DataLoader, decoder: Decoder, device: str, hparams: dict) -> None:
    """Generate and save reconstructed and new sample images.

    Args:
        model (Model): The trained VAE model.
        test_loader (DataLoader): DataLoader for the test dataset.
        decoder (Decoder): Decoder part of the VAE.
        device (str): Device to use for computations.
        hparams (dict): Hyperparameters for model and data processing.
    """
    # Generate reconstructions
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            x = x.view(hparams["batch_size"], hparams["x_dim"])
            x = x.to(device)
            x_hat, _, _ = model(x)
            break

    save_image(x.view(hparams["batch_size"], 1, 28, 28), "orig_data.png")
    save_image(x_hat.view(hparams["batch_size"], 1, 28, 28), "reconstructions.png")

    # Generate samples
    with torch.no_grad():
        noise = torch.randn(hparams["batch_size"], hparams["latent_dim"]).to(device)
        generated_images = decoder(noise)

    save_image(generated_images.view(hparams["batch_size"], 1, 28, 28), "generated_sample.png")


if __name__ == "__main__":
    train()