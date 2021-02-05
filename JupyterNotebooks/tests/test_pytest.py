#https://code.visualstudio.com/docs/python/testing
from dataset import DwgDataset
from model import RnnDecoder, RnnEncoder
from chamfer_distance_loss import MyChamferDistance
import torch

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

def test_chamfer_distance():
    loss = MyChamferDistance()

    a = [torch.randn([3,6]), torch.randn([2,6])]
    b = [torch.randn([4,6]), torch.randn([1,6])]
    
    loss_v = loss(a, b)
    
    # It can just throw, value 0 doesn't matter
    assert loss_v.item() != 0 , 'Random set chamfer distance is zero'

    # проверка расстояний до пустых множеств
    a = [torch.randn([0,6]), torch.randn([2,6])]
    b = [torch.randn([4,6]), torch.randn([0,6])]

    loss_v = loss(a, b)
    assert loss_v.item() != 0, 'Zero chamfer distance to an empty set'