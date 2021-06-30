import pymongo
import pandas as pd
import numpy as np
import math

from pymongo import MongoClient

def query_collection_to_dataframe(mongo_collection, fileId):
    query = {
        '$or':[
            {
                'ClassName' : 'Line',
                'EndPoint' : {'$ne' : None},
                'StartPoint' : {'$ne' : None},
                'GroupId' : fileId
            },
            {
                'ClassName': 'Text',
                'Position' : {'$ne' : None},
                'GroupId' : fileId
            },
            {
                'ClassName' : 'AlignedDimension',
                'XLine1Point' : {'$ne' : None},
                'XLine2Point' : {'$ne' : None},
                'GroupId' : fileId
            }
        ]
    }

    all_entities = list(mongo_collection.find(query))

    query = {
                'ClassName': 'Polyline',
                'GroupId' : fileId
            }

    polylines = list(mongo_collection.find(query))

    for pline in polylines:
        line = pline
        line['ClassName'] = 'Line'

        for i, vertix in enumerate(pline['Vertices']):
            if i==0:
                continue
            line['StartPoint'] = pline['Vertices'][i - 1]
            line['EndPoint'] = vertix
            all_entities.append(line)


    df = pd.DataFrame(all_entities)
    return df



def normalize(df, to_size=100):
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
    df[xcols] = (coords[xcols] - min_coord_x)*scale
    df[ycols] = (coords[ycols] - min_coord_y)*scale
        
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

def Col2Numpy(series, column_names):
    '''
    Splits json data {'ClassName':.., 'X':..,'Y':...,'Z':...} 
    of each column in column_names
    into columns 'column_name.X', 'column_name.Y',..
    '''
    
    # As each point columns is a json of {ClassName, X, Y, Z}
    # We break them into x, y, z columns and drop ClassName column
    
    result = pd.DataFrame()
    
    for col_name in column_names:
        # we don't split nan rows
        points = series[col_name].dropna(how="all")
        
        # get only row values and store index
        values = points.values.tolist()
        indexes = points.index
        
        # create sub-dataframe parsing json in values
        # and assign it stored index
        res = pd.DataFrame(data= values, index = indexes)
        
        # drop "Point3d" class name (if frame not empty)
        if len(res)>0:
            res = res.drop(columns=['ClassName'])

        # join columns with input dataframe
        result = pd.concat([result, res])
    return result

def scale_ds(x):
    _x1 = x.fillna(0).to_numpy()
    
    mn = _x1.min()
    mx = _x1.max()

    assert math.isnan(mn) == False, 'min is NaN'
    assert math.isnan(mx) == False, 'max is NaN'

    scale = 1/(mx - mn)
    x = (x - mn) * scale
    return x, scale