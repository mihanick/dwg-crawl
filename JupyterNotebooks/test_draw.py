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

def predict_one_stroke_for_batch(trainer, batch):
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

    batch_of_one_strokes = trainer.sample_next_state(pi, q, mu_x, mu_y, sigma_x, sigma_y, rho_xy)
    return batch_of_one_strokes

def generate_image(trainer):
    trainer.encoder.eval()
    trainer.decoder.eval()

    # batch of one with sos seq
    # 400 32 5
    seq = torch.zeros(trainer.max_seq_length, trainer.batch_size, trainer.stroke_features)
    seq[0, :, 2] = 1
    seq[-1, :, 4] = 1

    for i in range(1, trainer.max_seq_length):
        #seed single batch with sos

        z, _, _ = trainer.encoder(seq)

        # expand z in order it to be able to concatenate with inputs
        z_stack = torch.stack([z] * (trainer.max_seq_length))

        # inputs is the concatination of z and batch_inputs:
        inputs = torch.cat([seq, z_stack], 2)

        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, _, _ = trainer.decoder(inputs, z)

        a_stroke = trainer.sample_next_state(pi, q, mu_x, mu_y, sigma_x, sigma_y, rho_xy)

        seq[i, :] = a_stroke

    return seq

def test_model_generation():
    dwg_dataset = DwgDataset('test_dataset_cluster_labeled.pickle', batch_size = 32, limit_seq_len=200)
    trainer = Trainer(dwg_dataset)

    for b in dwg_dataset.train_loader:
        s = predict_one_stroke_for_batch(trainer=trainer, batch=b)
        print(s)
        
#test_model_generation()

def test_generate_image():
    dwg_dataset = DwgDataset('test_dataset_cluster_labeled.pickle', batch_size=1, limit_seq_len=200)
    trainer = Trainer(dwg_dataset)

    trainer.encoder.load_state_dict(torch.load('DimEncoder.model', map_location=trainer.device))
    trainer.decoder.load_state_dict(torch.load('DimDecoder.model', map_location=trainer.device))
 
    gen_bat = generate_image(trainer)
    generated_images = images_from_batch(gen_bat.transpose(0, 1))

    i=0
    for img in generated_images:
        img.savePng('img_g/generated' + str(i) + '.png')
        i+=1

test_generate_image()
