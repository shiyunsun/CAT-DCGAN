import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

class GAN_Dis(nn.Module):
    def __init__(self):
        super(GAN_Dis, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.0002),
            nn.Conv2d(64, 128, 4, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.0002),
            nn.Conv2d(128, 256, 4, 2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.0002),
            nn.Conv2d(256, 512, 4, 2, padding=1),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 1, 4, 1),    
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x.view(-1)


class GAN_Gen(nn.Module):
    def __init__(self):
        super(GAN_Gen, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 1024*4*4),
            nn.LeakyReLU(0.0002),
            nn.Unflatten(1, (1024, 4, 4)),
            nn.ConvTranspose2d(1024, 512, 4, 2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.0002),
            nn.ConvTranspose2d(512, 256, 4, 2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.0002),
            nn.ConvTranspose2d(256, 128, 4, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.0002),
            nn.ConvTranspose2d(128, 3, 4, 2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)
    
    def generate_image(self, save_path):
        with torch.no_grad():
            seed = torch.rand(100)
            image = self.forward(seed)
            image = F.to_pil_image(image[0])
            image.save(save_path)        