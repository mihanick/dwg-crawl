# https://arxiv.org/pdf/1904.08921.pdf
# https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html

from dataset import DwgDataset
dwg_dataset = DwgDataset('test_dataset_cluster_labeled.pickle', batch_size = 4)
dim_features = dwg_dataset.entities.dim_features
ent_features = dwg_dataset.entities.ent_features

import drawSvg as draw

def draw_sample(x, y, verbose = False):
    print(x.shape,y.shape)
    d = draw.Drawing(800, 200, origin=(0,0), displayInline = False)
    x = x*100
    y = y*100
    entscount = 0
    for row in x:
        if verbose:
            print(row)
        
        d.append(
        draw.Lines(
            row[0].item(),
            row[1].item(),
            row[3].item(),
            row[4].item(),
            close = False,
            fill='#eeee00',
            stroke = 'black'))
        entscount = entscount + 1
    
    print(y, (y != 0).all())
    if (y != 0).all():
        dim = draw.Lines(
                y[0].item(),
                y[1].item(),
                y[3].item(),
                y[4].item(),
                close = False,
                fill='#eeee00',
                stroke = 'blue',
                stroke_width = '1'
        )

        d.append(dim)
        entscount = entscount + 1    
            
    print('entities:', entscount)        
    #https://pypi.org/project/drawSvg/
    d.setPixelScale(2)
    r = d.rasterize()
    
    return d    

from IPython.display import Image
from IPython.display import clear_output

for _x,_y in iter(dwg_dataset.train_loader):
    for i in range(len(_x)):
        x=_x[i]
        y=_y[i]
        graphics = draw_sample(x,y, verbose=False)
        display(graphics) 
        #clear_output(wait=True)   
