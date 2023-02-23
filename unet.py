import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F

#ToDO Fill in the __ values
class Unet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.conv11 = nn.Conv2d(3, 64, kernel_size=3, stride=1)
        self.bnd11 = nn.BatchNorm2d(64)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bnd12 = nn.BatchNorm2d(64)
        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.bnd21 = nn.BatchNorm2d(128)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, stride=1)
        self.bnd22 = nn.BatchNorm2d(128)
        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        self.bnd31 = nn.BatchNorm2d(256)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bnd32 = nn.BatchNorm2d(256)
        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, stride=1)
        self.bnd41 = nn.BatchNorm2d(512)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, stride=1)
        self.bnd42 = nn.BatchNorm2d(512)
        self.conv51 = nn.Conv2d(512, 1024, kernel_size=3, stride=1)
        self.bnd51 = nn.BatchNorm2d(1024)
        self.conv52 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1)
        self.bnd52 = nn.BatchNorm2d(1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2)
        self.conv61 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.bnd61 = nn.BatchNorm2d(512)
        self.conv62 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bnd62 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.conv71 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bnd71 = nn.BatchNorm2d(256)
        self.conv72 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bnd72 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.conv81 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bnd81 = nn.BatchNorm2d(128)
        self.conv82 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bnd82 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2)
        self.conv91 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=2)
        self.bnd91 = nn.BatchNorm2d(64)
        self.conv92 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2)
        self.bnd92 = nn.BatchNorm2d(64)
        self.classifier = nn.Conv2d(64, self.n_class, kernel_size=1)
        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)

#TODO Complete the forward pass
    def forward(self, x):
        x1 = self.bnd11(self.relu(self.conv11(x)))
        x1 = self.bnd12(self.relu(self.conv12(x1)))        
        
        x2 = self.bnd21(self.relu(self.conv21(x1)))
        x2 = self.bnd22(self.relu(self.conv22(x2)))        
        
        x3 = self.bnd31(self.relu(self.conv31(x2)))
        x3 = self.bnd32(self.relu(self.conv32(x3)))        
        
        x4 = self.bnd41(self.relu(self.conv41(x3)))
        x4 = self.bnd42(self.relu(self.conv42(x4)))        

        x5 = self.bnd51(self.relu(self.conv51(x4)))
        x5 = self.bnd52(self.relu(self.conv52(x5)))        

        # Complete the forward function for the rest of the decoder
        y1 = self.deconv1(x5)
        dw, dh = x4.size()[2]-y1.size()[2], x4.size()[3]-y1.size()[3]
        y1 = F.pad(y1, [dw//2, dw-dw//2, dh//2, dh-dh//2])
        y1 = torch.cat([y1, x4], dim=1)
        y1 = self.bnd61(self.relu(self.conv61(y1)))
        y1 = self.bnd62(self.relu(self.conv62(y1)))        
        
        y2 = self.deconv2(y1)
        dw, dh = x3.size()[2]-y2.size()[2], x3.size()[3]-y2.size()[3]
        y2 = F.pad(y2, [dw//2, dw-dw//2, dh//2, dh-dh//2])
        y2 = torch.cat([y2, x3], dim=1)
        y2 = self.bnd71(self.relu(self.conv71(y2)))
        y2 = self.bnd72(self.relu(self.conv72(y2)))        
        
        y3 = self.deconv3(y2)
        dw, dh = x2.size()[2]-y3.size()[2], x2.size()[3]-y3.size()[3]
        y3 = F.pad(y3, [dw//2, dw-dw//2, dh//2, dh-dh//2])
        y3 = torch.cat([y3, x2], dim=1)
        y3 = self.bnd81(self.relu(self.conv81(y3)))
        y3 = self.bnd82(self.relu(self.conv82(y3)))        

        y4 = self.deconv4(y3)
        dw, dh = x1.size()[2]-y4.size()[2], x1.size()[3]-y4.size()[3]
        y4 = F.pad(y4, [dw//2, dw-dw//2, dh//2, dh-dh//2])
        y4 = torch.cat([y4, x1], dim=1)
        y4 = self.bnd91(self.relu(self.conv91(y4)))
        y4 = self.bnd92(self.relu(self.conv92(y4)))        

        score = self.classifier(y4)
        return score  # size=(N, n_class, H, W)

model = Unet(21).cuda()
summary(model, (3, 224, 224))