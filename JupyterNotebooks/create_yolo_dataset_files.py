from pathlib import Path
from sys import path
import numpy as np

from processing import  build_data
from plot_graphics import generate_file
import pandas as pd

df, ids = build_data(rebuild=False)
test_fraction = 0.1
test_split_index = round(test_fraction * len(ids))

np.random.seed(42)
np.random.shuffle(ids)

train_images_path = Path("data/dwg/images/train")
train_images_path.mkdir(parents=True,exist_ok=True)
train_labels_path = Path("data/dwg/labels/train")
train_labels_path.mkdir(parents=True,exist_ok=True)
train_desc_file_path = "data/dwg/train.txt"


val_images_path = Path("data/dwg/images/val")
val_images_path.mkdir(parents=True,exist_ok=True)
val_labels_path = Path("data/dwg/labels/val")
val_labels_path.mkdir(parents=True,exist_ok=True)
val_desc_file_path = "data/dwg/val.txt"

with open(train_desc_file_path, "w") as train_desc_file:
        with open(val_desc_file_path, "w") as val_desc_file:
                for i, id in enumerate(ids):
                        desc_file = train_desc_file
                        image_folder = str(train_images_path)
                        label_folder = str(train_labels_path)
                        if i < test_split_index:
                                desc_file = val_desc_file
                                image_folder = str(val_images_path)
                                label_folder = str(val_labels_path)

                        image_file_name = "{}/{}.png".format(image_folder, id)
                        label_file_name = "{}/{}.txt".format(label_folder, id)
                        generate_file(
                                df[df['GroupId'] == id], 
                                path=image_file_name,
                                verbose=False, 
                                draw_dimensions=False, 
                                draw_texts=False, 
                                save_file=True,
                                main_stroke='1')

                        desc_file.write("{}\n".format(image_file_name))

                        with open(label_file_name, 'w') as label_file:
                                dims = df[(df['GroupId'] == id) & (df['ClassName'] == 'AlignedDimension')]
                                for _, dim_row in dims.iterrows():
                                        category = "0"

                                        # TODO: get image size from image or from generation
                                        img_size = 512

                                        dim_x_coords = [dim_row['XLine1Point.X'] / img_size, dim_row['XLine2Point.X'] / img_size, dim_row['DimLinePoint.X'] / img_size] 
                                        dim_y_coords = [dim_row['XLine1Point.Y'] / img_size, dim_row['XLine2Point.Y'] / img_size, dim_row['DimLinePoint.Y'] / img_size] 

                                        x = min(dim_x_coords)
                                        y = min(dim_y_coords)
                                        bb_width = max(dim_x_coords) - x
                                        bb_height = max(dim_y_coords) - y

                                        bb_center_x = x# + (bb_width / 2)
                                        bb_center_y = y# + (bb_height / 2)

                                        label_file.write("{} {} {} {} {} \n".format(
                                                category,
                                                bb_center_x,
                                                bb_center_y,
                                                bb_width,
                                                bb_height
                                        ))