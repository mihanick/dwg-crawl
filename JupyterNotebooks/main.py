# https://code.visualstudio.com/docs/remote/wsl-tutorial
import torch
from dataset import DwgDataset

# https://stackoverflow.com/questions/20309456/call-a-function-from-another-file
from train import train_model


batch_size = 2
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

dwg_dataset = DwgDataset(pickle_file = 'test_dataset.pickle', batch_size = batch_size)

train_loader = dwg_dataset.train_loader
val_loader   = dwg_dataset.val_loader
test_loader  = dwg_dataset.test_loader
 
ent_features = dwg_dataset.entities.ent_features
dim_features = dwg_dataset.entities.dim_features

from model import RnnDecoder, RnnEncoder
rnn_encoder = RnnEncoder(ent_features, 1024).to(device)
rnn_decoder = RnnDecoder(1024, dim_features).to(device)


from chamfer_distance_loss import MyChamferDistance

loss = MyChamferDistance()

lr = 1e-4
epochs = 13
decoder_optimizer = torch.optim.Adam(rnn_decoder.parameters(), lr=lr)
encoder_optimizer = torch.optim.Adam(rnn_encoder.parameters(), lr=lr)


train_model(
    encoder      = rnn_encoder, 
    decoder      = rnn_decoder, 
    train_loader = train_loader,
    val_loader   = val_loader,
    loss         = loss,
    decoder_opt  = decoder_optimizer,
    encoder_opt  = encoder_optimizer,
    epochs       = epochs)