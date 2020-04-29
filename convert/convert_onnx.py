import torch
import yaml
from torch import nn
from backbone import EfficientDetBackbone
import numpy as np

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

device = torch.device('cuda')
params = Params(f'projects/coco.yml')
model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=0, onnx_export=True,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales)).to(device)

                    
model.backbone_net.model.set_swish(memory_efficient=False)

dummy_input = torch.randn((1,3,512,512), dtype=torch.float32).to(device)

model.load_state_dict(torch.load(f'weights/efficientdet-d0.pth'))

# opset_version can be changed to 10 or other number, based on your need
torch.onnx.export(model, dummy_input,
                  'convert/efficientdet-d0.onnx',
                  verbose=False,
                  input_names=['data'],
                  opset_version=11)
