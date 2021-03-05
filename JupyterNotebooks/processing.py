import pymongo
import pandas as pd
import numpy as np
from pymongo import MongoClient

def generate_image_by_id(collection, fileId):
    data = query_collection_to_dataframe(collection, fileId)
    cols_to_expand = ['XLine1Point', 'XLine2Point','StartPoint','EndPoint','Position']
    data = expand_columns(data, cols_to_expand)
    data = normalize(data, to_size = 400)
    generate_file(data)
    
    filename = 'img/' + fileId + '.png'
    return filename  

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
        points = series[col_name].dropna()
        
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