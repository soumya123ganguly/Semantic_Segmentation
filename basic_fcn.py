import torch.nn as nn

#ToDO Fill in the __ values
class FCN(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.conv1 = nn.Conv2d(3, __, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd1 = nn.BatchNorm2d(__)
        self.conv2 = nn.Conv2d(32, __, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd2 = nn.BatchNorm2d(__)
        self.conv3 = nn.Conv2d(64, __, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd3 = nn.BatchNorm2d(__)
        self.conv4 = nn.Conv2d(128, __, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd4 = nn.BatchNorm2d(__)
        self.conv5 = nn.Conv2d(256, __, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd5 = nn.BatchNorm2d(__)
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, __, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(__)
        self.deconv2 = nn.ConvTranspose2d(512, __, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(__)
        self.deconv3 = nn.ConvTranspose2d(256, __, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(__)
        self.deconv4 = nn.ConvTranspose2d(128, __, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(__)
        self.deconv5 = nn.ConvTranspose2d(64, __, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(__)
        self.classifier = nn.Conv2d(__, self.n_class, kernel_size=1)

#TODO Complete the forward pass
    def forward(self, x):
        x1 = self.bnd1(self.relu(self.conv1(x)))
        # Complete the forward function for the rest of the encoder

        y1 = self.bn1(self.relu(self.deconv1(__)))
        # Complete the forward function for the rest of the decoder

        score = self.classifier(__)

        return score  # size=(N, n_class, H, W)
