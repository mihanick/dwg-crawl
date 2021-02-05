#https://code.visualstudio.com/docs/python/testing
from dataset import DwgDataset
from model import RnnDecoder, RnnEncoder
from os import listdir


def test_input_output_model():
    '''
    runs all samples throug dataset
    '''
    data = DwgDataset('./test_dataset.pickle')

    encoder = RnnEncoder(9,32)
    decoder = RnnDecoder(32,6)

    i = 0
    for (x, y) in iter(data.train_loader):
        outs, learned = encoder(x)
        decoded = decoder(learned, outs)

        i += 1
