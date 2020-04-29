# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Compile PyTorch Models
======================
**Author**: `Alex Wong <https://github.com/alexwong/>`_

This article is an introductory tutorial to deploy PyTorch models with Relay.

For us to begin with, PyTorch should be installed.
TorchVision is also required since we will be using it as our model zoo.

A quick solution is to install via pip

.. code-block:: bash

    pip install torch==1.4.0
    pip install torchvision==0.5.0

or please refer to official site
https://pytorch.org/get-started/locally/

PyTorch versions should be backwards compatible but should be used
with the proper TorchVision version.

Currently, TVM supports PyTorch 1.4 and 1.3. Other versions may
be unstable.
"""

import tvm
from tvm import relay

import numpy as np

from tvm.contrib import util
from tvm.contrib.download import download_testdata
# from tvm.relay.frontend.pytorch import get_graph_input_names

# PyTorch imports
import torch
import torchvision

from backbone import EfficientDetBackbone

obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']

target_device = 'x86.cuda'
path = '/media/hsw/E/work/github/Yet-Another-EfficientDet-Pytorch/weights/efficientdet-d0.pth'
device = torch.device('cuda')

######################################################################
# Load a pretrained PyTorch model
# -------------------------------
compound_coef = 0
model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list), onnx_export=True).to(device)
model.backbone_net.model.set_swish(False)

model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth'))
model.eval()

# # We grab the TorchScripted model via tracing
input_shape = [1, 3, 512, 512]
import cv2 
img = cv2.imread('/media/hsw/E/work/github/efficientdet-tf2/img.png').astype(np.float32) / 255.
img = cv2.resize(img, (512,512))
img = np.expand_dims(img.transpose(2,0,1), axis=0)

input_data = torch.from_numpy(img).to(device)
scripted_model = torch.jit.trace(model, input_data).eval()
print ('scripted model done.')

######################################################################
# Import the graph to Relay
# -------------------------
# Convert PyTorch graph to Relay graph.
# input_name = get_graph_input_names(scripted_model)[0]  # only one input
# shape_dict = {input_name: input_shape}

input_name = 'data'
shape_list = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(scripted_model,
                                          shape_list)
print ('relay frontend from_pytorch done.')

######################################################################
# Relay Build
# -----------
# Compile the graph to llvm target with given input specification.
if target_device == 'x86.cuda':
    target = tvm.target.cuda(model='1080ti',options="-libs=cudnn, cublas")
    target_host = 'llvm'
else:
    target = tvm.target.cuda(model='1080ti',options="-libs=cudnn, cublas")
    target_host = 'llvm'

with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(mod,
                                     target=target,
                                     target_host=target_host,
                                     params=params)
######################################################################
#build the relay model
# compile kernels with history best records
print("Compile...")

######################################################################
#save the relay model
temp = util.tempdir()
path_lib = temp.relpath("%s.%s.so" % (path, device))

lib.export_library(path_lib)
#lib.export_library(path_lib, tvm.contrib.cc.create_shared, cc="aarch64-linux-gnu-g++")
print("path: ", temp.relpath("%s"%path))
with open(temp.relpath("%s.%s.json" % (path, device)), "w") as fo:
    fo.write(graph)
with open(temp.relpath("%s.%s.params" % (path, device)), "wb") as fo:
    fo.write(relay.save_param_dict(params))


print("------convert done!!!------")


#####################################################################
#####################################################################
# # Execute the portable graph on TVM
# # ---------------------------------
# # Now we can try deploying the compiled model on target.
from tvm.contrib import graph_runtime
import time
# ctx = tvm.cpu(0)
ctx = tvm.gpu(0)
dtype = 'float32'
m = graph_runtime.create(graph, lib, ctx)
# Set inputs
i = 0
# test tvm inference
while i < 100:
    start = time.time()
    m.set_input(input_name, tvm.nd.array(img.astype(dtype)))
    m.set_input(**params)
    # Execute
    m.run()
    # Get outputs
    tvm_output = m.get_output(0)
    print (time.time() - start)
    i += 1

# #####################################################################
# a = list(range(1,10))
# inputs = torch.from_numpy(input_img).to(device)
# with torch.no_grad():
#     torch_regression, torch_class = model(inputs)
#     # torch_regression, torch_class = torch_regression.cpu().numpy(), torch_class.cpu().numpy()
# tvm_regression, tvm_class = m.get_output(0).asnumpy(), m.get_output(1).asnumpy()

# torch_regression = torch.from_numpy(tvm_regression).to(device)
# torch_class = torch.from_numpy(tvm_class).to(device)

# from efficientdet.utils import BBoxTransform, ClipBoxes, Anchors
# def display(preds, imgs, imshow=True, imwrite=False, obj_list=obj_list):
#     for i in range(len(imgs)):
#         if len(preds[i]['rois']) == 0:
#             continue

#         for j in range(len(preds[i]['rois'])):
#             (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
#             cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
#             obj = obj_list[preds[i]['class_ids'][j]]
#             score = float(preds[i]['scores'][j])

#             cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
#                         (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                         (255, 255, 0), 1)

#         if imshow:
#             cv2.imshow('img', imgs[i])
#             cv2.waitKey(0)

#         if imwrite:
#             cv2.imwrite(f'test/from_tvm.jpg', imgs[i])

# anchors_func = Anchors()
# anchors = anchors_func(inputs, inputs.dtype)

# regressBoxes = BBoxTransform()
# clipBoxes = ClipBoxes()
# threshold = 0.2
# iou_threshold = 0.2
# out = postprocess(inputs,
#                     anchors, torch_regression, torch_class,
#                     regressBoxes, clipBoxes,
#                     threshold, iou_threshold)

# out = invert_affine(framed_metas, out)
# display(out, ori_imgs, imshow=False, imwrite=True, obj_list=obj_list)

#####################################################################