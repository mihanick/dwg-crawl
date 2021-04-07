'''
Contains tests for input and output of models and also for loss functions
'''

#https://code.visualstudio.com/docs/python/testing
import torch
from dataset import DwgDataset
from model import DecoderRNN, EncoderRNN

from plot_graphics import images_from_batch, save_batch_images
from calc_loss import BivariateGaussianMixture


class Sampler:
    def __init__(self, encoder:EncoderRNN, decoder:DecoderRNN):
        self.decoder = decoder
        self.encoder = encoder

    def sample(self, data: torch.Tensor, temperature: float):
        longest_seq_len = data.shape[0]
        z, _, _ = self.encoder(data)
        s = data.new_tensor([0, 0, 1, 0, 0])
        seq = [s]

        state = None

        z_stack = z.unsqueeze(0).expand(data.shape[0]- 1, -1, -1)
        inputs = torch.cat([data[:-1], z_stack], 2)

        with torch.no_grad():
            for i in range(longest_seq_len):
                dist, q_logits, state = self.decoder(inputs, z, state)
                s = self._sample_step(dist, q_logits, temperature)

                seq.append(s)

                if s[4] == 1:
                    break
        # print(seq)
        seq = torch.stack(seq)

        return seq
    
    @staticmethod
    def _sample_step(dist: 'BivariateGaussianMixture', q_logits: torch.Tensor, temperature: float):
        dist.set_temperature(temperature)
        pi, mix = dist.get_distribution()
        idx = pi.sample()[0, 0]
        q = torch.distributions.Categorical(logits=q_logits / temperature)
        q_idx = q.sample()[0, 0]
        xy = mix.sample()[0, 0, idx]

        stroke = q_logits.new_zeros(5)
        stroke[:2] = xy
        stroke[q_idx + 2] = 1

        return stroke

def test_data_drawing():
    from dataset import DwgDataset
    dwg_dataset = DwgDataset('test_dataset_cluster_labeled.pickle', batch_size = 32, limit_seq_len=200)

    for i, batch in enumerate(dwg_dataset.train_loader):
        
        data = batch[0].transpose(0, 1)
        mask = batch[1].transpose(0, 1)
        
        save_batch_images(data)
        break

# test_data_drawing()

def test_model_generation():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    from dataset import DwgDataset
    dwg_dataset = DwgDataset('test_dataset_cluster_labeled.pickle', batch_size = 32, limit_seq_len=200)

    stroke_features = dwg_dataset.entities.stroke_features
    max_seq_length = dwg_dataset.entities.max_seq_length

    enc_hidden_size    = 256
    dec_hidden_size    = 512

    d_z=128
    n_distributions    = 20

    from model import DecoderRNN, EncoderRNN
    encoder = EncoderRNN(d_z, enc_hidden_size, stroke_features=stroke_features)
    decoder = DecoderRNN(d_z, dec_hidden_size, n_distributions, stroke_features=stroke_features)
    encoder.to(device)
    decoder.to(device)

    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    encoder.load_state_dict(torch.load('DimEncoder.model', map_location=device))
    decoder.load_state_dict(torch.load('DimDecoder.model', map_location=device))

    decoder.eval()
    encoder.eval()
    sampler = Sampler(encoder, decoder)

    for i, batch in enumerate(dwg_dataset.train_loader):
        
        data = batch[0].to(device).transpose(0, 1)
        mask = batch[1].to(device).transpose(0, 1)
        
        # draw_batch(data)

        seq = sampler.sample(data, 0.4)
        print(seq)
        break

test_model_generation()
