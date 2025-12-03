#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyTorch implementation of REDNet architecture
Based on REDNet_ch1.prototxt and REDNet_ch3.prototxt

Reference:
  Xiao-Jiao Mao, Chunhua Shen, Yu-Bin Yang.
  Image Restoration Using Very Deep Convolutional Encoder-Decoder Networks with Symmetric Skip Connections
  NIPS 2016 (https://arxiv.org/pdf/1606.08921.pdf)
"""
import torch.nn as nn


class REDNet(nn.Module):
    """
    RED-Net encoder-decoder with symmetric skip connections.
    Architecture: 15 conv layers + 15 deconv layers with skip connections
    """
    def __init__(self, num_channels=1, num_features=128):
        """
        Args:
            num_channels: Input channels (1 for grayscale, 3 for RGB)
            num_features: Number of feature maps in hidden layers (128)
        """
        super(REDNet, self).__init__()

        # Encoder (15 conv layers)
        self.conv1 = nn.Conv2d(num_channels, num_features, 3, padding=1)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv3 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv4 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv5 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv6 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv7 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv8 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv9 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv10 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv11 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv12 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv13 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv14 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv15 = nn.Conv2d(num_features, num_features, 3, padding=1)

        # Decoder (15 deconv layers, last one outputs num_channels)
        self.deconv1 = nn.ConvTranspose2d(num_features, num_features, 3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(num_features, num_features, 3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(num_features, num_features, 3, padding=1)
        self.deconv4 = nn.ConvTranspose2d(num_features, num_features, 3, padding=1)
        self.deconv5 = nn.ConvTranspose2d(num_features, num_features, 3, padding=1)
        self.deconv6 = nn.ConvTranspose2d(num_features, num_features, 3, padding=1)
        self.deconv7 = nn.ConvTranspose2d(num_features, num_features, 3, padding=1)
        self.deconv8 = nn.ConvTranspose2d(num_features, num_features, 3, padding=1)
        self.deconv9 = nn.ConvTranspose2d(num_features, num_features, 3, padding=1)
        self.deconv10 = nn.ConvTranspose2d(num_features, num_features, 3, padding=1)
        self.deconv11 = nn.ConvTranspose2d(num_features, num_features, 3, padding=1)
        self.deconv12 = nn.ConvTranspose2d(num_features, num_features, 3, padding=1)
        self.deconv13 = nn.ConvTranspose2d(num_features, num_features, 3, padding=1)
        self.deconv14 = nn.ConvTranspose2d(num_features, num_features, 3, padding=1)
        self.deconv15 = nn.ConvTranspose2d(num_features, num_channels, 3, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Encoder with skip connections stored
        c1 = self.relu(self.conv1(x))
        c2 = self.relu(self.conv2(c1))
        c3 = self.relu(self.conv3(c2))
        c4 = self.relu(self.conv4(c3))
        c5 = self.relu(self.conv5(c4))
        c6 = self.relu(self.conv6(c5))
        c7 = self.relu(self.conv7(c6))
        c8 = self.relu(self.conv8(c7))
        c9 = self.relu(self.conv9(c8))
        c10 = self.relu(self.conv10(c9))
        c11 = self.relu(self.conv11(c10))
        c12 = self.relu(self.conv12(c11))
        c13 = self.relu(self.conv13(c12))
        c14 = self.relu(self.conv14(c13))
        c15 = self.relu(self.conv15(c14))

        # Decoder with symmetric skip connections
        # deconv1 -> skip with c14
        d1 = self.relu(self.deconv1(c15))
        d1a = self.relu(d1 + c14)

        # deconv2
        d2 = self.relu(self.deconv2(d1a))

        # deconv3 -> skip with c12
        d3 = self.relu(self.deconv3(d2))
        d3a = self.relu(d3 + c12)

        # deconv4
        d4 = self.relu(self.deconv4(d3a))

        # deconv5 -> skip with c10
        d5 = self.relu(self.deconv5(d4))
        d5a = self.relu(d5 + c10)

        # deconv6
        d6 = self.relu(self.deconv6(d5a))

        # deconv7 -> skip with c8
        d7 = self.relu(self.deconv7(d6))
        d7a = self.relu(d7 + c8)

        # deconv8
        d8 = self.relu(self.deconv8(d7a))

        # deconv9 -> skip with c6
        d9 = self.relu(self.deconv9(d8))
        d9a = self.relu(d9 + c6)

        # deconv10
        d10 = self.relu(self.deconv10(d9a))

        # deconv11 -> skip with c4
        d11 = self.relu(self.deconv11(d10))
        d11a = self.relu(d11 + c4)

        # deconv12
        d12 = self.relu(self.deconv12(d11a))

        # deconv13 -> skip with c2
        d13 = self.relu(self.deconv13(d12))
        d13a = self.relu(d13 + c2)

        # deconv14
        d14 = self.relu(self.deconv14(d13a))

        # deconv15 (final layer, no ReLU)
        d15 = self.deconv15(d14)

        # final skip connection with input
        return d15 + x
