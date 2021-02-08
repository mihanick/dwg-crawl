#https://code.visualstudio.com/docs/python/testing
from dataset import DwgDataset
from model import RnnDecoder, RnnEncoder
from chamfer_distance_loss import MyChamferDistance, MyNumberLoss
from train import calculate_accuracy

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
        decoded = decoder(outs, learned)

        i += 1

def test_debug_ch_dist():
    '''
    analyze calculation of chamfer loss and accuracy
    '''
    data = DwgDataset('./test_dataset.pickle')

    encoder = RnnEncoder(9,32)
    decoder = RnnDecoder(32,6)
    loss = MyChamferDistance()

    i = 0
    for (x, y) in iter(data.train_loader):
        outs, learned = encoder(x)
        decoded = decoder(outs, learned)


        lv = loss(decoded, y)
        print ('loss', lv)

        acc = calculate_accuracy(decoded, y)
        print('acc', acc)

        for batch_no in range(len(y)):
            print('actual number', y[batch_no].shape[0])
            print('predicted number', decoded[batch_no].shape[0])

            print('loss', lv[batch_no])

            print('accuracy', acc[batch_no])
        i += 1
        print(i, '------------------------------')



def test_chamfer_distance():
    # loss = MyChamferDistance()
    loss = MyNumberLoss()
    a = [torch.randn([3,6]), torch.randn([2,6])]
    b = [torch.randn([4,6]), torch.randn([1,6])]
    
    loss_v = loss(a,b)
    
    # It can just throw, value 0 doesn't matter
    assert loss_v.item() != 0 , 'Random set chamfer distance is zero'

    loss_v.backward()

    # проверка расстояний до пустых множеств
    a = [torch.randn([0, 6]), torch.randn([2, 6])]
    b = [torch.randn([4, 6]), torch.randn([0, 6])]

    loss_v = loss(a, b)
    assert loss_v.item() != 0, 'Does not calculate chamfer distance to an empty set'

    # RuntimeError: leaf variable has been moved into the graph interior
    # https://stackoverflow.com/questions/64272718/understanding-the-reason-behind-runtimeerror-leaf-variable-has-been-moved-into

    loss_v.backward()