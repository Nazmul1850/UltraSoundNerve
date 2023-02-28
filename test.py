import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import cv2
import os
import numpy as np
from data_loader import process
from rsaModel import RA_Net
# from keras import backend as K

class ContextEncoder(nn.Module):
    def __init__(self):
        super(ContextEncoder, self).__init__()

        #encoder blocks
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)

        #decoder blocks
        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1)

        self.relu = nn.LeakyReLU(0.2, inplace=True)
    

    def forward(self, input):
        input = self.relu(self.conv1(input))
        input = self.relu(self.conv2(input))
        input = self.relu(self.conv3(input))
        input = self.relu(self.conv4(input))

        input = self.relu(self.deconv1(input))
        input = self.relu(self.deconv2(input))
        input = self.relu(self.deconv3(input))
        input = self.deconv4(input)

        return input
    

def dice_cofficient(y_true, y_pred, smooth=1):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    return (2. * intersection + smooth) / (union + smooth)

def test(model_path, image, device):
    model = RA_Net(n_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location = torch.device(device)))
    impainted_image = None
    with torch.no_grad():
        model.eval()
        impainted_image = model(image)
    
    return impainted_image

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# path = r'train'
# image = process(os.path.join(path,'15_51.tif'))
# print(image.shape)
# transform = transforms.ToTensor()
# image = transform(image)
# print(image.shape)
# new_image = test('nerve_model_path.path', image, device)
# new_image = new_image.numpy()
# # 0-255 pixel range
# print(new_image)
# # new_image = (new_image - new_image.min()) / (new_image.max() - new_image.min())
# new_image = (new_image*255).astype(np.uint8)

# new_image = cv2.resize(new_image[0], (580,420), interpolation =cv2.INTER_AREA)
# cv2.imwrite('mask_resized.jpg', new_image)
# _, new_image = cv2.threshold(new_image,170,255,cv2.THRESH_BINARY)
# # new_image = cv2.cvtColor(new_image, cv2.COLOR_GRAY2BGR)
# print(new_image.shape)
# print(new_image)
# cv2.imwrite('mask_binary.jpg', new_image)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
path = r'train'
image = process(os.path.join(path,'15_51.tif'))
image_mask = process(os.path.join(path,'15_51_mask.tif'))
print(image.shape)
print(image_mask.shape)
transform = transforms.ToTensor()
image = transform(image)
image_mask = transform(image_mask)
print(image.shape)
image = torch.unsqueeze(image, dim=1)
image_mask = torch.unsqueeze(image_mask, dim=1)
print(image.shape)
print(image_mask.shape)
new_images = test('nerve_model_path.path', image, device)
new_images = new_images.numpy()
image_mask = image_mask.numpy()
print(new_images.shape)
print(image_mask.shape)
dc = dice_cofficient(new_images, image_mask)
print('Dice Coefficient: ', dc)
# 0-255 pixel range
for new_image in new_images:
    print(new_image)
    # new_image = (new_image - new_image.min()) / (new_image.max() - new_image.min())
    new_image = (new_image*255).astype(np.uint8)

    new_image = cv2.resize(new_image[0], (580,420), interpolation =cv2.INTER_AREA)
    cv2.imwrite('mask_resized.jpg', new_image)
    _, new_image = cv2.threshold(new_image,170,255,cv2.THRESH_BINARY)
    # new_image = cv2.cvtColor(new_image, cv2.COLOR_GRAY2BGR)
    print(new_image.shape)
    print(new_image)
    cv2.imwrite('mask_binary.jpg', new_image)

