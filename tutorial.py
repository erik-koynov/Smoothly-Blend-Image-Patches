import torch
from smooth_image_blending.smooth_tiled_predictions import predict_img_with_smooth_windowing
from smooth_image_blending.padding import AllAroundPadding
import numpy as np
import sys
sys.path.append('../backboned-unet/backboned_unet')
from backboned_unet.base_classes import BaseModel
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

class DummyModel(nn.Module, BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, input_):
        return input_

def predict_same(small_img_patches):
    return torch.Tensor(small_img_patches)

model = DummyModel()
padding = AllAroundPadding(pad_repeat=2)
dummy_image = np.random.randint(0, 255, (1000, 1000))
dummy_image = np.array(Image.open("images/6100_1_3.jpg"))
dtype = dummy_image.dtype
plt.imshow(dummy_image)
plt.show()

print(dummy_image.shape)
print(dummy_image[0])
prediction = predict_img_with_smooth_windowing(dummy_image, 128, 64,
                                  padding, lambda x: predict_same(x), batch_size=4)
print(prediction[0])
print(dummy_image.mean())
print(prediction.mean())
print("mean absolute reconstruction error: ", np.mean(np.abs(prediction-dummy_image)))
plt.imshow(prediction.astype(dtype))
plt.show()
