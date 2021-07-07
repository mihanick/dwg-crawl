'''
Train functions to run from console or jupyter
'''

import collections
from math import inf, sqrt
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio
import glob
import math

from processing import query_collection_to_dataframe, expand_columns, normalize

# https://pypi.org/project/drawSvg/
# !pip install drawsvg
# Works on linux only!
# sudo apt-get install libcairo2
try:
    import drawSvg as draw
except:
    print('Could not import drawSvg')
    
def plot_history(train_ls, train_lp, train_lkl, val_ls, val_lp, val_lkl):
    '''
    Plots learning history in jupyter
    '''
    #plt.ylabel('Loss @ epoch')

    # https://stackabuse.com/matplotlib-plot-multiple-line-plots-same-and-different-scales/
    fig, ax = plt.subplots()
    ax.plot(train_ls, label="train_ls", color='orange')
    ax.plot(val_ls, label="val_ls", color='red')
    
    ax.plot(train_lp, label="train_lp", color='lime')
    ax.plot(val_lp, label="val_lp", color='green')
    
    ax.tick_params(axis='y')
    ax.set_ylabel('Stroke loss, pen loss @ epoch')
    
    ax2 = ax.twinx()
    ax2.plot(train_lkl, label="train_kl", color='cyan')
    ax2.plot(val_lkl, label="val_kl", color='blue')
     
    ax2.tick_params(axis='y')
    ax2.set_ylabel('Kullback Leiber loss @ epoch')

    ax.legend(loc="upper left")
    ax2.legend(loc='upper right')
    plt.show()

def generate_file(group, verbose=False, save_file=False, draw_dimensions=False, draw_texts=False, main_stroke='2'):
    # print(group.info())
    
    fileid = group.iloc[0]['GroupId']
    if len(fileid) == 0:
        return
    
    file_name = 'img/' + fileid + '.png'
    d = draw.Drawing(512, 512, origin=(0, 0), displayInline=False)
    
    ents_count = 0
    for _, row in group.iterrows():
        if verbose:
            print(row)
        if row['ClassName'] == 'Line':
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
                    close=False,
                    fill='#eeee00',
                    stroke='black',
                    stroke_width=main_stroke))
            ents_count = ents_count + 1
        if row['ClassName'] == 'Arc':
            cx = row['Center.X']
            cy = row['Center.Y']
            sx = row['StartPoint.X']
            sy = row['StartPoint.Y']
            ex = row['EndPoint.X']
            ey = row['EndPoint.Y']
            v_start = [sx - cx, sy - cy]
            v_end = [ex - cx, ey - cy]
            r = sqrt( v_start[0] * v_start[0]  + v_start[1] * v_start[1])
            if v_start[0] == 0:
                start_angle = math.degrees(math.atan(np.sign(v_start[1]) * math.inf))
            else:
                start_angle = math.degrees(math.atan2(v_start[1] , v_start[0]))
            if v_end[0] == 0:
                end_angle = math.degrees(math.atan(np.sign(v_end[1]) * math.inf))
            else:
                end_angle = math.degrees(math.atan2(v_end[1] , v_end[0]))

            if start_angle < 0:
                start_angle += 360
            if end_angle < 0:
                end_angle += 360

            if (v_start[0]* v_end[1] -  v_start[1]*v_end[0]) < 0:
                (end_angle,start_angle) = (start_angle, end_angle)
            # print(start_angle, end_angle, v_start[0]* v_end[1] -  v_start[1]*v_end[0])
            
            d.append(
                draw.Arc(
                    cx=cx,
                    cy=cy,
                    r=r,
                    startDeg=start_angle,
                    endDeg=end_angle,
                    stroke='black',
                    stroke_width=main_stroke,
                    fill='none')
            )
            ents_count = ents_count + 1

        if row['ClassName'] == 'Circle':
            cx = row['Center.X']
            cy = row['Center.Y']
            r = row['Radius']

            d.append(
                draw.Circle(
                    cx=cx,
                    cy=cy,
                    r=r,
                    stroke='black',
                    stroke_width=main_stroke,
                    fill='none')
            )
            ents_count = ents_count + 1

        # https://github.com/cduck/drawSvg/blob/master/drawSvg/elements.py
        if row['ClassName'] == 'Text' and draw_texts:
            d.append(
                draw.Text(
                    row['TextString'],
                    20,
                    row['Position.X'],
                    row['Position.Y'],
                    center=False
                )
            )
            ents_count = ents_count + 1

        if row['ClassName'] == 'AlignedDimension' and draw_dimensions:
            # https://github.com/cduck/drawSvg

            dim_x_coords = [row['XLine1Point.X'], row['XLine2Point.X'], row['DimLinePoint.X']] 
            dim_y_coords = [row['XLine1Point.Y'], row['XLine2Point.Y'], row['DimLinePoint.Y']] 

            x = min(dim_x_coords)
            y = min(dim_y_coords)
            width = max(dim_x_coords) - x
            height = max(dim_y_coords) - y

            d.append(
                draw.Rectangle(
                    x=x,
                    y=y,
                    width=width, 
                    height=height,
                    stroke='blue',
                    fill='none',
                    stroke_width='1'))
            
            d.append(
                draw.Text(
                    row['DimensionText'],
                    10,
                    row['DimLinePoint.X'],
                    row['DimLinePoint.Y'],
                    center=False,
                    stroke='blue'))

            ents_count = ents_count + 1

    print('id:', fileid, 'entities:', ents_count)
    #https://pypi.org/project/drawSvg/
    d.setPixelScale(1)
    r = d.rasterize()
    
    if save_file:
        d.savePng(file_name)
        # d.saveSvg('img/' + fileid + '.svg')
    return r, file_name

def images_from_batch(data, verbose=False):
    #print(data.shape)
    drawing_size = 200
    
    #data = data.transpose(0, 1)
    result = []

    for batch_no in range(data.shape[0]):
        d = draw.Drawing(2*drawing_size, drawing_size, displayInline=True)
        seq = data[batch_no]

        # TODO: sample this data already normalized
        min_coord = torch.min(seq[:, :2])
        max_coord = torch.max(seq[:, :2])
        scale = 1/(max_coord - min_coord)
        seq[:, :2] -= min_coord
        seq[:, :2] *= scale

        entscount = 0
        x = 0
        y = 0

        for stroke in seq:
            if verbose:
                print(stroke)            
            if stroke[4] == 1:
                continue

            dx = stroke[0].item() * drawing_size
            dy = stroke[1].item() * drawing_size

            if dx == 0 and dy == 0:
                continue

            cl = 'black'
            stroke_width = '1'
            if stroke[3] == 1: #is_dim
                cl='blue'
                stroke_width = '2'

            if stroke[2] == 1: #is_pen
                d.append(
                    draw.Lines(
                        x,
                        y,
                        x + dx,
                        y + dy,
                        close=False,
                        fill='#eeee00',
                        stroke=cl,
                        stroke_width=stroke_width))
                entscount = entscount + 1
            x = x + dx
            y = y + dy

        if verbose:
            print('entities:', entscount)        
        #https://pypi.org/project/drawSvg/
        d.setPixelScale(2)
        r = d.rasterize()
        #d.savePng('img_g/test' + str(batch_no) + '.png')
        # return r
        result.append(d)
        
    return result

def save_batch_images(data):
    vv = images_from_batch(data)
    for i, img in enumerate(vv):
        img.savePng('img_g/img'+str(i)+'.png')

def plot_generated_stroke_over_sequence(sequence, predicted_stroke=None, batch_number=0):
    sequence = sequence[:, batch_number, :].cpu().numpy()
    
    x = 0
    y = 0

    fig = plt.figure()
    for seq_stroke in sequence:
        if seq_stroke[4] == 1:
            continue
        dx = 100*seq_stroke[0]
        dy = -100*seq_stroke[1]
        if seq_stroke[2] > 0:
            plt.plot((x, x + dx), (y, y + dy), color='black')
        x+=dx
        y+=dy
    
    if predicted_stroke is not None:
        s = predicted_stroke.squeeze(0)
        dx = 100*s[0]
        dy = -100*s[1]
        if s[2] > 0:
            plt.plot((x, x + dx), (y, y + dy), color='red')
    
    return fig


def create_gif_from_frames(folder):
    '''
    creates gif file from png frames in folder
    frames should be named 'gen_frame*.png'
    png frames are erased in process
    
    uses
     imageio
     os
     glob
    '''

    anim_file = folder+ '/predictions.gif'

    with imageio.get_writer(anim_file, mode = 'I', fps=0.75) as writer:
        filenames = glob.glob(folder+'/gen_frame*.png')
        filenames = sorted(filenames)

        last = -1

        for i, filename in enumerate(filenames):
            #frame = 2*(i**0.5)
            #if (round(frame)> round(last)):
            #    last = frame
            #else:
            #    continue

            image = imageio.imread(filename)
            writer.append_data(image)
        return anim_file