import pymongo
import pandas as pd
import numpy as np
from pymongo import MongoClient

# https://pypi.org/project/drawSvg/
# !pip install drawsvg
# Works on linux only!
# sudo apt-get install libcairo2
import drawSvg as draw

def query_collection_to_dataframe(mongo_collection, fileId):
    query = {
        '$or':[
            {
                'ClassName' : 'AcDbLine',
                'EndPoint' : {'$ne' : None},
                'StartPoint' : {'$ne' : None},
                'FileId' : fileId
            },
            {
                'Position' : {'$ne' : None},
                'FileId' : fileId
            },
            {
                'ClassName' : 'AcDbRotatedDimension',
                'XLine1Point' : {'$ne' : None},
                'XLine2Point' : {'$ne' : None},
                'FileId' : fileId
            }
        ]
    }

    df = pd.DataFrame(list(mongo_collection.find(query)))
    return df

def generate_file(group, verbose = False, entities_limit = 1e9):
    fileid = group['FileId'].any()
    filename = fileid +'.png'
    d = draw.Drawing(800, 400, origin=(0,0), displayInline = False)
    
    # skip small drawings
    #if (len(group)<100):
    #    continue
    
    entscount = 0
    for row_index, row in group.iterrows():
        if verbose:
            print(row)
        if row['ClassName'] == 'AcDbLine':
            d.append(
                draw.Lines(
                    row['StartPoint']['X'],
                    row['StartPoint']['Y'],
                    row['EndPoint']['X'],
                    row['EndPoint']['Y'],
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
                    row['Position']['X'],
                    row['Position']['Y'],
                    center = False
                )
            )
            entscount = entscount + 1
        if row['ClassName'] == 'AcDbRotatedDimension':

            dim = draw.Lines(
                    row['XLine1Point']['X'],
                    row['XLine1Point']['Y'],
                    row['XLine2Point']['X'],
                    row['XLine2Point']['Y'],
                    close = False,
                    fill='#eeee00',
                    stroke = 'blue',
                    stroke_width = '1'
            )
            
            # https://github.com/cduck/drawSvg
            dim.appendTitle(row['DimensionText'])
            d.append(dim)
            entscount = entscount + 1    
        if entscount > entities_limit:
            break
            
    print('id:',fileid,'entities:', entscount)        
    #https://pypi.org/project/drawSvg/
    d.setPixelScale(2)
    r = d.rasterize()
    
    d.savePng(filename)
    #d.saveSvg(filename+'.svg')
    
def normalize(df, to_size = 100):
    cols = []
    for column in df.columns:
        m = ".X" in column or ".Y" in column
        if m:
            cols.append(column)

    coords = df[cols].fillna(0).to_numpy()
    
    # https://stackoverflow.com/questions/44471801/zero-size-array-to-reduction-operation-maximum-which-has-no-identity
    if (not np.any(coords)):
        return df
    
    # print(coords)
    
    diff = np.max(coords) - np.min(coords)
    min_coord = np.min(coords)

    scale = to_size/diff

    # print(min_coord, scale)
    v = (coords - min_coord)*scale
    df[cols] = v
        
    return  df

def generate_file2(group, verbose = False, entities_limit = 1e9):
    # print(group.info())
    
    # skip small drawings
    if (len(group)<1):
        return
    
    fileid = group.iloc[0]['FileId']
    if len(fileid) == 0:
        return
    
    filename = fileid +'.png'
    d = draw.Drawing(800, 400, origin=(0,0), displayInline = False)
    
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
            dim.appendTitle(row['DimensionText'])
            d.append(dim)
            entscount = entscount + 1    
        if entscount > entities_limit:
            break
            
    print('id:',fileid,'entities:', entscount)        
    #https://pypi.org/project/drawSvg/
    d.setPixelScale(2)
    r = d.rasterize()
    
    d.savePng(filename)
    #d.saveSvg(filename+'.svg')
    
# https://stackoverflow.com/questions/49081097/slice-pandas-dataframe-json-column-into-columns
def expand_columns(df, column_names):
    res = df
    for col_name in column_names:
        
        if col_name not in df.columns:
            continue
        
        # get rid of nans in column, so the type can be determined and parsed
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html 
        res1 = df.dropna(subset = [col_name])
        values = res1[col_name].values.tolist()
        indexes = res1[col_name].index
        # print(values, indexes)
        # as we dropped some rows to get rid of nans
        # we need to keep index so rows can be matched between list 
        # and source dataset
        res1 = pd.DataFrame(data= values, index = indexes)
        #print(res1)
        res1.columns = col_name +"."+ res1.columns

        # res = res.drop(col_name, axis = 1)
        
        # Keep index!!
        res = pd.concat([res, res1], axis = 1)
    return res