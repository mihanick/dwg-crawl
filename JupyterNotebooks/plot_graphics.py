'''
Train functions to run from console or jupyter
'''

import collections
from math import inf, sqrt
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.npyio import save
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
    
def generate_file2(file_id, path=None, verbose=False, save_file=False, draw_dimensions=False, draw_texts=False, main_stroke='2'):
    return generate_file(
        group = query_collection_to_dataframe(fileId=file_id),
        path=path, 
        verbose=verbose,
        save_file=save_file,
        draw_dimensions=draw_dimensions,
        draw_texts=draw_texts,
        main_stroke=main_stroke)

def generate_file(group, path=None, verbose=False, save_file=False, draw_dimensions=False, draw_texts=False, main_stroke='2'):
    # print(group.info())
    
    fileid = group.iloc[0]['GroupId']
    if len(fileid) == 0:
        return
    
    file_name = 'img/' + fileid + '.png'
    if path:
        file_name = path

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
                end_angle = math.degrees(math.atan2(v_end[1], v_end[0]))

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

        if row['ClassName'] == 'AlignedDimension':
            # https://github.com/cduck/drawSvg

            dim_x_coords = [row['XLine1Point.X'], row['XLine2Point.X'], row['DimLinePoint.X']] 
            dim_y_coords = [row['XLine1Point.Y'], row['XLine2Point.Y'], row['DimLinePoint.Y']] 

            x = min(dim_x_coords)
            y = min(dim_y_coords)
            width = max(dim_x_coords) - x
            height = max(dim_y_coords) - y

            # We need to actually draw dimensions invisible, 
            # in order to generate images without repositioning
            dim_stroke = 'none'
            if draw_dimensions:
                dim_stroke = 'blue'

            d.append(
                draw.Rectangle(
                    x=x,
                    y=y,
                    width=width, 
                    height=height,
                    stroke=dim_stroke,
                    fill='none',
                    stroke_width='1'))
                    
            if draw_dimensions:
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

