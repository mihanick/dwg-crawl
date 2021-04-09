'''
Contains tests for input and output of models and also for loss functions
'''


#https://code.visualstudio.com/docs/python/testing
import torch
from dataset import DwgDataset
# from model import DecoderRNN, EncoderRNN, Sampler
from sketch_rnn import DecoderRNN, EncoderRNN, Trainer

from plot_graphics import images_from_batch, save_batch_images
from calc_loss import BivariateGaussianMixture


def test_data_drawing():
    dwg_dataset = DwgDataset('test_dataset_cluster_labeled.pickle', batch_size = 32, limit_seq_len=200)

    for i, batch in enumerate(dwg_dataset.train_loader):
        
        data = batch[0].transpose(0, 1)
        mask = batch[1].transpose(0, 1)
        
        save_batch_images(data)
        break

# test_data_drawing()

def generate_sequence(trainer, batch):
    trainer.encoder.eval()
    trainer.decoder.eval()

    batch = batch[0].to(trainer.device).transpose(0, 1)
    lengths = batch[1].to(trainer.device).transpose(0, 1)

    # batch, lenghts = make_batch(batch_size)
    z, mu, sigma = trainer.encoder(batch)

    # expand z in order it to be able to concatenate with inputs
    z_stack = torch.stack([z] * (trainer.max_seq_length))

    # inputs is the concatination of z and batch_inputs:
    inputs = torch.cat([batch, z_stack], 2)

    pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, _, _ = trainer.decoder(inputs, z)

    # prepare targets:
    mask, dx, dy, p = trainer.make_target(batch)

    #[max_seq_len, batch_size, M]->[max_seq_len, batch_size, 1] (i.e. mean by M)
    dx = torch.mean(dx, dim=-1).unsqueeze(-1)
    dy = torch.mean(dy, dim=-1).unsqueeze(-1)
    return torch.cat([dx, dy, p], dim=-1).transpose(0, 1)

def test_model_generation():
    dwg_dataset = DwgDataset('test_dataset_cluster_labeled.pickle', batch_size = 32, limit_seq_len=200)
    trainer = Trainer(dwg_dataset)

    for b in dwg_dataset.train_loader:
        s = generate_sequence(trainer=trainer, batch=b)
        print(s.shape, b[0].shape)
        

# test_model_generation()
