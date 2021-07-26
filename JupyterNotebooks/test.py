from models import CalculatePrediction
from utils.datasets import ListDataset
import torch
from utils.utils import PlotImageAndPrediction

dataloader = torch.utils.data.DataLoader(
    ListDataset('./data/dwg/train.txt', max_objects=87), batch_size=4, shuffle=False
)

from utils.utils import PlotImageAndPrediction
from plot_graphics import generate_file2
from IPython import display
import os

for batch_i, (file_names, imgs, targets) in enumerate(dataloader):
    dets = CalculatePrediction(model=None, batch_of_images=imgs)
    for i, _img in enumerate(imgs):
        #file_id = os.path.splitext(os.path.split(file_names[i])[1])[0]
        #d, _ = generate_file2(file_id=file_id, draw_dimensions=True, draw_texts=True, save_file=False)
        #display.display(d)

        det = dets[i]
        trg = targets[i]
        display.display(PlotImageAndPrediction(image=_img, target=trg, detections=det))
        break
    
    break