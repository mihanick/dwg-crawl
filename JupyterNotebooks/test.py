'''
Contains tests for input and output of models and also for loss functions
'''


#https://code.visualstudio.com/docs/python/testing
import torch
from dataset import DwgDataset
from plot_graphics import images_from_batch, save_batch_images
from sketch_rnn import Trainer, pretty_print


def test_data_drawing():
    dwg_dataset = DwgDataset('test_dataset_cluster_labeled.pickle', batch_size = 32, limit_seq_len=200)

    for i, batch in enumerate(dwg_dataset.train_loader):
        
        data = batch[0].transpose(0, 1)
        mask = batch[1].transpose(0, 1)
        
        save_batch_images(data)
        break

# test_data_drawing()

def test_predict_stroke_by_stroke():
    dwg_dataset = DwgDataset('test_dataset_cluster_labeled_.pickle', batch_size=1, limit_seq_len=100)
    trainer = Trainer(dwg_dataset)

    trainer.encoder.load_state_dict(torch.load('DimEncoder.model', map_location=trainer.device))
    trainer.decoder.load_state_dict(torch.load('DimDecoder.model', map_location=trainer.device))
    trainer.encoder.eval()
    trainer.decoder.eval()

    for b in dwg_dataset.train_loader:
        
        input_seq = b[0].to(trainer.device).transpose(0, 1)

        seq_so_far = torch.zeros_like(input_seq)
        
        for i in range(1, trainer.max_seq_length):
            input_stroke = input_seq[i]
            seq_so_far[:, :i+1] = input_seq[:, :i+1]
            z, _, _ = trainer.encoder(seq_so_far)
            #pretty_print(z)
            # expand z in order it to be able to concatenate with inputs
            z_stack = torch.stack([z] * (trainer.max_seq_length))

            # inputs is the concatination of z and batch_inputs:
            inputs = torch.cat([seq_so_far, z_stack], 2)

            pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, _, _ = trainer.decoder(inputs, z)

            generated_stroke = trainer.sample_next_state(pi, q, mu_x, mu_y, sigma_x, sigma_y, rho_xy)

            iss = input_stroke[0]
            gss = generated_stroke[0]
            if iss[4] != 1:
                pretty_print("input    :", iss[2:])
                pretty_print("generated:", gss[2:])
                print(" ")

# test_predict_stroke_by_stroke()

def test_random_decoder_input():
    dwg_dataset = DwgDataset('test_dataset_cluster_labeled.pickle', batch_size=16, limit_seq_len=100)
    trainer = Trainer(dwg_dataset)

    trainer.encoder.load_state_dict(torch.load('DimEncoder.model', map_location=trainer.device))
    trainer.decoder.load_state_dict(torch.load('DimDecoder.model', map_location=trainer.device))
    trainer.encoder.eval()
    trainer.decoder.eval()

    seq = torch.zeros(trainer.max_seq_length, trainer.batch_size, trainer.stroke_features)

    #seed single batch with sos
    seq[0, :, 2] = 1

    for i in range(1, trainer.max_seq_length):
       
        z = torch.randn(trainer.batch_size, trainer.Nz)
        # expand z in order it to be able to concatenate with inputs
        z_stack = torch.stack([z] * (trainer.max_seq_length))

        # inputs is the concatination of z and batch_inputs:
        inputs = torch.cat([seq, z_stack], 2)

        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, _, _ = trainer.decoder(inputs, z)

        # clip q generator
        # q[0, 0, 2] = 1

        a_stroke = trainer.sample_next_state(pi, q, mu_x, mu_y, sigma_x, sigma_y, rho_xy)
        # print(a_stroke.cpu().numpy())
        seq[i, :] = a_stroke

    generated_images = images_from_batch(seq.transpose(0, 1))
    pretty_print(seq[seq[:,:,4]==0])
    i=0
    for img in generated_images:
        img.savePng('img_g/generated' + str(i) + '.png')
        i+=1

#test_random_decoder_input()