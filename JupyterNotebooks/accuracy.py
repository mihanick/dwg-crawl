import numpy as np
import torch
from calc_loss import KLDivLoss, ReconstructionLoss

def CalculateLoaderAccuracy(encoder, decoder, loader, device):
    with torch.no_grad():
        encoder.eval()
        decoder.eval()

        kl_losses = []
        reconstruction_losses = []

        for _, batch in enumerate(loader):
            data = batch[0].to(device).transpose(0, 1)
            mask = batch[1].to(device).transpose(0, 1)

            z, mu, sigma_hat = encoder(data)

            z_stack = z.unsqueeze(0).expand(data.shape[0]- 1, -1, -1)

            inputs = torch.cat([data[:-1], z_stack], 2)

            dist, q_logits, _ = decoder(inputs, z, None)

            kl_loss = KLDivLoss()(sigma_hat, mu)
            reconstruction_loss = ReconstructionLoss()(mask, data[1:], dist, q_logits)

            kl_losses.append(kl_loss.item())
            reconstruction_losses.append(reconstruction_loss.item())

        return np.mean(kl_losses), np.mean(reconstruction_losses)

