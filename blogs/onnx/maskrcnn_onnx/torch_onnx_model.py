#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file       :torch_onnx_model.py
@Description:create a model & export to *.onnx file
@Date       :2021/01/15
@Author     :Luo Siyou
@version    :1.0
'''
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn


img = Image.open('demo.jpg')
img = img.resize((500,667))
img.save('demo_resize.jpg')

transform = transforms.Compose([
    transforms.ToTensor(),])
input_imgs = transform(img).unsqueeze(0)
print(f"shape of input_imgs: {input_imgs.shape}")


# device = torch.device('cuda:0')
device = torch.device('cpu')

model = maskrcnn_resnet50_fpn(pretrained=True)
model.to(device).eval()

predictions = model(input_imgs.to(device))

# input_randn = torch.randn(1, 3, 667, 500,device=device) # batch_size = 1

x = input_imgs
# x = input_randn
print(f"shape of input x: {x.shape}")


torch.onnx.export(
    model.to(device),
    x,
    'model.onnx',
    opset_version=11,
    input_names=["input"],		# 输入名
    output_names=["boxes","labels","scores","masks"],	# 输出名
    dynamic_axes={  "input":{0:"batch_size"},	# 批处理变量
                    "boxes":{0:"batch_size"},
                    "labels":{0:"batch_size"},
                    "scores":{0:"batch_size"},
                    "masks":{0:"batch_size"}
                    },
    verbose=True  # set True to print model graph
    )

import onnx
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
# print("model graph: \n",onnx.helper.printable_graph(onnx_model.graph))
