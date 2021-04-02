'''
Main program for model training and evaluation in python console
version 4. For SketchRNN and straight sequence of entities
'''
# https://code.visualstudio.com/docs/remote/wsl-tutorial
import time
import torch
from torch import nn

from model import DecoderRNN, EncoderRNN
from calc_loss import ReconstructionLoss, KLDivLoss
from dataset import DwgDataset
from accuracy import CalculateLoaderAccuracy


def run(batch_size=32, pickle_file='test_dataset_cluster_labeled.pickle', lr=0.008, epochs=15):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    dwg_dataset = DwgDataset(pickle_file=pickle_file, batch_size=batch_size)

    train_loader = dwg_dataset.train_loader
    val_loader   = dwg_dataset.val_loader
    test_loader  = dwg_dataset.test_loader

    ent_features = dwg_dataset.entities.ent_features
    dim_features = dwg_dataset.entities.dim_features

    max_seq_length = dwg_dataset.max_seq_length

    enc_hidden_size    = 256
    dec_hidden_size    = 512

    d_z                = 128
    n_distributions    = 20
    kl_div_loss_weight = 0.5
    grad_clip          = 1.0
    temperature        = 0.4

    encoder = EncoderRNN(d_z, enc_hidden_size).to(device)
    decoder = DecoderRNN(d_z, dec_hidden_size, n_distributions).to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

    start = time.time()
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        
        encoder.train()
        decoder.train()

        for i, batch in enumerate(train_loader):
            data = batch[0].to(device).transpose(0, 1)
            mask = batch[1].to(device).transpose(0, 1)

            z, mu, sigma_hat = encoder(data)

            z_stack = z.unsqueeze(0).expand(data.shape[0]- 1, -1, -1)

            inputs = torch.cat([data[:-1], z_stack], 2)

            dist, q_logits, _ = decoder(inputs, z, None)

            kl_loss = KLDivLoss()(sigma_hat, mu)
            reconstruction_loss = ReconstructionLoss()(mask, data[1:], dist, q_logits)

            loss = reconstruction_loss + kl_loss * kl_div_loss_weight

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
            nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)

            optimizer.step()

            print('[{:4.0f}-{:4.0f} @ {:5.1f} sec] RLoss: {:5.5f} KL Loss: {:1.4f}'
                        .format(
                            epoch,
                            i,
                            time.time() - start,
                            reconstruction_loss.item(),
                            kl_loss.item()
                        ))
        
        # validation
        val_kl, val_rl = CalculateLoaderAccuracy(encoder, decoder, val_loader)
        print('Epoch [{}] validation losses kl:{:1.4f} rl:{:1.4f}'.format(epoch, val_kl, val_rl))

        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        # save model
        torch.save(encoder.state_dict(), 'DimEncoder.model')
        torch.save(decoder.state_dict(), 'DimDecoder.model')

    # Calculate test accuracy
    test_kl, test_rl = CalculateLoaderAccuracy(encoder, decoder, test_loader)
    print('Test losses kl:{:1.4f} rl:{:1.4f}'.format(epoch, test_kl, test_rl))
