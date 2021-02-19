'''
Contains tests for input and output of models and also for loss functions
'''

#https://code.visualstudio.com/docs/python/testing
import torch
from torch import nn
from chamfer_distance_loss import MyChamferDistance, ChamferDistance2
from dataset import DwgDataset
from model import RnnDecoder, RnnEncoder, UglyModel, Padder
from train import calculate_accuracy, calculate_accuracy2, count_relevant

def _ugly_model_run(loss):
    '''
    runs all samples throug dataset
    '''
    
    data = DwgDataset('./test_dataset.pickle', batch_size=2)
    max_seq_length = data.max_seq_length
    input_padder = Padder(max_seq_length)
    output_padder = Padder(max_seq_length)

    model = UglyModel(9, 6, max_seq_length)

    i = 0
    for (x, y) in iter(data.train_loader):
        x_padded = input_padder(x)
        output = model(x_padded)

        y_padded = output_padder(y)
        loss_v = loss(output_padder(output), y_padded)
        loss_v.backward()
        acc = calculate_accuracy2(output, y_padded)
        
        
        print(i, '------------------------------')
        print('actual number',  count_relevant(y_padded))
        print('predicted number', count_relevant(output))
        print('loss', loss_v)
        print('accuracy', acc)

        i += 1

def test_ugly_model_mse():
    '''
    runs all samples throug dataset with mse
    '''
    loss = nn.MSELoss()
    _ugly_model_run(loss)

def test_ugly_model_ch_dist():
    '''
    runs all samples throug dataset with ch_dist
    '''
    loss = ChamferDistance2()
    _ugly_model_run(loss)


def test_single_random_input_ugly_model():
    loss = nn.MSELoss()

    max_seq_length = 4
    input_padder = Padder(max_seq_length, requires_grad=False)
    output_padder = Padder(max_seq_length, requires_grad=False)

    out = [torch.randn([3, 6], requires_grad=True), torch.randn([max_seq_length, 6], requires_grad=True)]
    y = [torch.randn([max_seq_length, 6]), torch.randn([1, 6])]
    output_p = input_padder(out)
    y_p = output_padder(y)

    loss_v = loss(output_p, y_p)
    print('check random sets')
    print(loss_v)
    loss_v.backward()

    acc = calculate_accuracy2(output_p, y_p)

    print("check empty sets")

    out = [torch.randn([0, 6], requires_grad=True), torch.randn([max_seq_length, 6], requires_grad=True)]
    y = [torch.randn([max_seq_length, 6]), torch.randn([0, 6])]
    output_p = input_padder(out)
    y_p = output_padder(y)

    loss_v = loss(output_p, y_p)
    print(loss_v)
    loss_v.backward()

    acc = calculate_accuracy2(output_p, y_p)
    print(acc)

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
