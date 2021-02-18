'''
Contains tests for input and output of models and also for loss functions
'''

#https://code.visualstudio.com/docs/python/testing
import torch
from chamfer_distance_loss import MyChamferDistance
from dataset import DwgDataset
from model import RnnDecoder, RnnEncoder
from train import calculate_accuracy

def test_input_output_model():
    '''
    runs all samples throug dataset
    '''

    loss = MyChamferDistance()

    data = DwgDataset('./test_dataset.pickle')

    encoder = RnnEncoder(9, 32)
    decoder = RnnDecoder(32, 6)

    i = 0
    for (x, _y) in iter(data.train_loader):
        outs, learned = encoder(x)
        _decoded = decoder(outs, learned)
        loss_v = loss(_decoded, _y)
        loss_v.backward()
        acc = calculate_accuracy(_decoded, _y)
        i += 1

def test_debug_ch_dist():
    '''
    analyze calculation of chamfer loss and accuracy
    '''
    data = DwgDataset('./test_dataset.pickle')

    encoder = RnnEncoder(9, 32)
    decoder = RnnDecoder(32, 6)
    loss = MyChamferDistance()
    # loss = MyMSELoss(6)

    i = 0
    for (x, y) in iter(data.train_loader):
        outs, learned = encoder(x)
        decoded = decoder(outs, learned)

        loss_v = loss(decoded, y)
        print('loss', loss_v)

        acc = calculate_accuracy(decoded, y)
        # print('acc', acc)

        for batch_no in range(len(y)):
            print('actual number', y[batch_no].shape[0])
            print('predicted number', decoded[batch_no].shape[0])

            print('loss', loss_v[batch_no])

            print('accuracy', acc[batch_no])
        i += 1
        print(i, '------------------------------')


def _check_loss(loss_layer, print_loss=False):
    loss = loss_layer
    _a = [torch.randn([3, 6]), torch.randn([2, 6])]
    _b = [torch.randn([4, 6]), torch.randn([1, 6])]

    loss_v = loss(_a, _b)

    print('check random sets')
    print(loss_v)

    loss_v.backward()
    if print_loss:
        print(loss_v)

    acc = calculate_accuracy(_a, _b)
    print(acc)

    print("check empty sets")
    _a = [torch.randn([0, 6]), torch.randn([2, 6])]
    _b = [torch.randn([4, 6]), torch.randn([0, 6])]

    loss_v = loss(_a, _b)
    print(loss_v)

    # RuntimeError: leaf variable has been moved into the graph interior
    # https://stackoverflow.com/questions/64272718/understanding-the-reason-behind-runtimeerror-leaf-variable-has-been-moved-into

    loss_v.backward()
    acc = calculate_accuracy(_a, _b)
    print(acc)

def test_loss_acc():
    '''
    tests chamfer distance loss class
    '''
    loss_layer = MyChamferDistance()
    _check_loss(loss_layer)
