#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file       :torch_script_model.py
@Description:create a model & export to *.pt file
@Date       :2021/01/15
@Author     :Luo Siyou
@version    :1.0
'''
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
import copy

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

model = maskrcnn_resnet50_fpn(pretrained=True)
model.to(device).eval()

predictions = model(input_imgs.to(device))
n_predictions = predictions[0]['labels'].shape[0]

import cv2
img_result = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)

masks_predict = to_numpy(predictions[0]['masks'])
print(f'shape of masks_predict: {masks_predict.shape}') # (13, 1, 667, 500)
for i in range(n_predictions):
    if predictions[0]['scores'][i] > 0.3:
        score = to_numpy(predictions[0]['scores'][i])
        label = to_numpy(predictions[0]['labels'][i])
        boxes = to_numpy(predictions[0]['boxes'][i])
        masks = masks_predict[i][0]
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

traced_model = torch.jit.script(model)
torch.jit.save(traced_model, 'model.pt')