# https://stackoverflow.com/questions/16249736/how-to-import-data-from-mongodb-to-pandas
from processing import build_data
from pymongo import MongoClient
from plot_graphics import generate_file

df, file_ids = build_data()

for file_id in file_ids:
    if file_id!='1f36e041-dc69-462b-ade6-dbfef3b41284' :
        continue
    drawing, file_name = generate_file(df[df['GroupId'] == file_id], verbose=False, save_file=True)
