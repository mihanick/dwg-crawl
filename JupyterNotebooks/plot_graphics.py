'''
Train functions to run from console or jupyter
'''
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

# https://pypi.org/project/drawSvg/
# !pip install drawsvg
# Works on linux only!
# sudo apt-get install libcairo2
import drawSvg as draw

def plot_history(loss_history, train_history, val_history):
    '''
    Plots learning history in jupyter
    '''
    plt.ylabel('Accuracy @ epoch')

    train, = plt.plot(train_history)
    train.set_label("train")

    validation, = plt.plot(val_history)
    validation.set_label("validation")

    loss, = plt.plot(np.log10(loss_history))
    loss.set_label("loss")

    plt.legend()
    plt.show()

def generate_file(group, verbose = False, entities_limit = 1e9, save_file=True):
    # print(group.info())
    
    # skip small drawings
    if (len(group)<1):
        return
    
    fileid = group.iloc[0]['FileId']
    if len(fileid) == 0:
        return
    
    filename = 'img/' + fileid + '.png'
    d = draw.Drawing(800, 200, origin=(0,0), displayInline = False)
    
    entscount = 0
    for row_index, row in group.iterrows():
        if verbose:
            print(row)
        if row['ClassName'] == 'AcDbLine':
            # print('StartPoint.X', row['StartPoint.X'])
            # print('StartPoint.Y', row['StartPoint.Y'])
            # print('EndPoint.X', row['EndPoint.X'])
            # print('EndPoint.Y', row['EndPoint.Y'])
            
            d.append(
                draw.Lines(
                    row['StartPoint.X'],
                    row['StartPoint.Y'],
                    row['EndPoint.X'],
                    row['EndPoint.Y'],
                    close = False,
                    fill='#eeee00',
                    stroke = 'black'))
            entscount = entscount + 1
        # https://github.com/cduck/drawSvg/blob/master/drawSvg/elements.py
        if row['ClassName'] == 'AcDbText':

            d.append(
                draw.Text(
                    row['TextString'],
                    6,
                    row['Position.X'],
                    row['Position.Y'],
                    center = False
                )
            )
            entscount = entscount + 1
        if row['ClassName'] == 'AcDbRotatedDimension':

            dim = draw.Lines(
                    row['XLine1Point.X'],
                    row['XLine1Point.Y'],
                    row['XLine2Point.X'],
                    row['XLine2Point.Y'],
                    close = False,
                    fill='#eeee00',
                    stroke = 'blue',
                    stroke_width = '1'
            )
            
            # https://github.com/cduck/drawSvg
            # dim.appendTitle(row['DimensionText'])
            d.append(dim)
            entscount = entscount + 1    
        if entscount > entities_limit:
            break
            
    print('id:',fileid,'entities:', entscount)        
    #https://pypi.org/project/drawSvg/
    d.setPixelScale(2)
    r = d.rasterize()
    
    if save_file:
        d.savePng(filename)
    #d.saveSvg(filename+'.svg')
    return d    