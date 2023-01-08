#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file       :onnxruntime_load_model.py
@Description:load model from *.onnx file & inference
@Date       :2021/01/15
@Author     :Luo Siyou
@version    :1.0
'''
import copy
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import onnxruntime


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


img = Image.open('demo.jpg')
img = img.resize((500,667))
img.save('demo_resize.jpg')

transform = transforms.Compose([
    transforms.ToTensor(),])
input_imgs = transform(img).unsqueeze(0)
print(f"shape of input_imgs: {input_imgs.shape}")


# device = torch.device('cuda:0')
device = torch.device('cpu')

x = input_imgs
ort_session = onnxruntime.InferenceSession("model.onnx")
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
# print(ort_inputs)
ort_output = ort_session.run(None,input_feed=ort_inputs)
for i in range(len(ort_output)-1):
    print(f"output node {i}: \n",ort_output[i])


import cv2
n_predictions = ort_output[1].shape[0]

img_result = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
masks_predict = ort_output[3]
print(f'shape of masks_predict: {masks_predict.shape}') # (14, 1, 667, 500)
for i in range(n_predictions):
    score = ort_output[2][i]
    if score > 0.3:
        label = ort_output[1][i]
        boxes = ort_output[0][i]
        masks = ort_output[3][i][0]
        mask = np.zeros_like(masks,dtype=np.uint8)
        mask[masks>0.5] = 255
        mask_mat = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
        img_result2 = cv2.addWeighted(img_result,1,mask_mat,0.2,0)
        img_result = img_result2
        print(score,label,boxes)
        cv2.rectangle(img_result,(boxes[0],boxes[1]),(boxes[2],boxes[3]),(0,255,0))
        cv2.putText(img_result,
            'L{}: {:.3f}'.format(label,score),
            (boxes[0],boxes[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            1)
cv2.imshow('img result',img_result)
cv2.waitKey(0)
