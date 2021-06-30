import re
import pymongo
from pymongo import MongoClient

from plot_graphics import generate_image_by_id

client = MongoClient('mongodb://192.168.0.104:27017/')
db = client.geometry2
objects = db.objects
fileidsWithDims = objects.find({'ClassName':'AlignedDimension'}).distinct('GroupId')
# f_id = fileidsWithDims[1]
f_id = '179ecf2a-ca38-4cc0-a8b6-94048fecf551'
generate_image_by_id(objects, f_id)

exit

for f_id in fileidsWithDims:
    filename, d = generate_image_by_id(objects, f_id)
