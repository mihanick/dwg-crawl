# from drawSvg.elements import DrawingBasicElement

# https://stackoverflow.com/questions/16249736/how-to-import-data-from-mongodb-to-pandas

import pymongo
import pandas as pd
import numpy as np
import math

from pymongo import MongoClient

def build_data(rebuild=False):
    pickle_file = 'test_dataset_groups.pickle'
    group_ids_file = 'ids.txt'
    result_ids = []

    if rebuild:
        client = MongoClient('mongodb://192.168.0.104:27017')
        db = client.geometry2
        objects = db.objects

        fileidsWithDims = list(objects.find().distinct('GroupId'))
        df = pd.DataFrame()

        
        for fileId in fileidsWithDims:
            data = query_collection_to_dataframe(objects, fileId)
            if data is not None:
                df = pd.concat([df, data])
                result_ids.append(fileId)

        df['ClassName'] = df['ClassName'].astype('category')
        df['GroupId'] = df['GroupId'].astype('category')

        df.to_pickle(pickle_file)
        with open(group_ids_file, 'w') as f:
            for id in result_ids:
                f.write(id+'\n')

    else:
        df = pd.read_pickle(pickle_file)
        with open(group_ids_file) as f:
            result_ids = f.read().splitlines()

    return df, result_ids

def query_collection_to_dataframe(mongo_collection=None, fileId=None, max_entities=25000, min_entities=8):
    '''
    Queries mongo collection to dataframe.
    Expands certain columns, like StartPoint to StartPoint.X, StartPoint.Y
    Scales each sample
    Returns pandas dataframe with given columns.

    If mongo_collection is None, connects new client to collection
    '''

    # If collection is not specified
    if mongo_collection is None:
        client = MongoClient('mongodb://192.168.1.49:27017')
        db = client.geometry2
        mongo_collection = db.objects

    # Just arbitrary drawing
    if fileId is None:
        fileId = '1317d221-8d9e-4e2e-b290-3be2a0aa67fb'

    # first we query mongo collection for lines, texts and dimensions
    query = {
        '$or':[
            {
                'ClassName' : 'Line',
                'EndPoint'  : {'$ne' : None},
                'StartPoint': {'$ne' : None},
                'GroupId'   : fileId
            },
            {
                'ClassName' : 'Arc',
                'EndPoint'  : {'$ne' : None},
                'StartPoint': {'$ne' : None},
                'Center'    : {'$ne' : None},
                'GroupId'   : fileId
            },
            {
                'Radius': {'$ne' : None},
                'Center'    : {'$ne' : None},
                'GroupId'   : fileId
            },
            {
                'ClassName': 'Text',
                'Position' : {'$ne' : None},
                'GroupId'  : fileId
            },
            {
                'ClassName'     : 'AlignedDimension',
                'XLine1Point'   : {'$ne' : None},
                'XLine2Point'   : {'$ne' : None},
                'DimensionText' : {'$ne':None},
                'GroupId'       : fileId
            }
        ]
    }

    # than we query collection for polylines 
    
    all_entities = list(mongo_collection.find(query))

    query = {
                'ClassName': 'Polyline',
                'GroupId' : fileId
            }
    polylines = list(mongo_collection.find(query))

    # and add each polyline segment as line
    for pline in polylines:
        line = pline
        line['ClassName'] = 'Line'

        for i, vertix in enumerate(pline['Vertices']):
            if i==0:
                continue
            line['StartPoint'] = pline['Vertices'][i - 1]
            line['EndPoint'] = vertix
            all_entities.append(line)

    # now we create dataframe
    if (len(all_entities) < min_entities or len(all_entities) > max_entities):
        return

    df = pd.DataFrame(all_entities)

    # We expand object point columns to point coordinates
    cols_to_expand = ['XLine1Point', 'XLine2Point', 'StartPoint', 'EndPoint', 'Position', 'DimLinePoint', 'Center']
    description_cols = ['GroupId', 'ClassName', 'TextString', 'DimensionText', 'Radius']
    df = expand_columns(df, cols_to_expand)

    # and return only dataframe with given columns
    dataframe_cols = []
    for col in cols_to_expand:
        dataframe_cols.append(col+'.X')
        dataframe_cols.append(col+'.Y')
    dataframe_cols += description_cols

    for col in dataframe_cols:
        if col not in df.columns:
            df[col] = np.nan

    # we normalize dataframe
    df = normalize(df)

    return df[dataframe_cols]

def normalize(df, to_size=512):
    xcols = []
    ycols = []
    for column in df.columns:
        if ".X" in column:
            xcols.append(column)
        if ".Y" in column:
            ycols.append(column)
    
    cols = xcols + ycols

    coords = df[cols]
    max_coord_x = df[xcols].max().max()
    max_coord_y = df[ycols].max().max()
    min_coord_x = df[xcols].min().min()
    min_coord_y = df[ycols].min().min()

    diff_x = max_coord_x - min_coord_x
    diff_y = max_coord_y - min_coord_y

    diff = max(diff_x, diff_y)

    # https://stackoverflow.com/questions/38134012/pandas-dataframe-fillna-only-some-columns-in-place
    #coords[xcols] = coords[xcols].fillna(min_coord_x)
    #coords[ycols] = coords[ycols].fillna(min_coord_y)

    # https://stackoverflow.com/questions/44471801/zero-size-array-to-reduction-operation-maximum-which-has-no-identity
    if (not np.any(coords.to_numpy())):
        return df
    
    # print(coords)
    scale = to_size/diff

    # print(min_coord, scale)
    df[xcols] = (coords[xcols] - min_coord_x) * scale
    df[ycols] = (coords[ycols] - min_coord_y) * scale
    df['Radius'] = df['Radius'] * scale
        
    return  df
    
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

def scale_ds(x):
    _x1 = x.fillna(0).to_numpy()
    
    mn = _x1.min()
    mx = _x1.max()

    assert math.isnan(mn) == False, 'min is NaN'
    assert math.isnan(mx) == False, 'max is NaN'

    scale = 1/(mx - mn)
    x = (x - mn) * scale
    return x, scale