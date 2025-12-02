import torch

import deblurganv2
import dncnn
import mair
import rednet
import restormer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = deblurganv2.get_model('../weights/DeblurGANv2/fpn_inception.h5', device=device)
model = dncnn.get_model('../weights/DnCNN/dncnn_color_blind.pth', n_channels=3, nb=20, device=device)
model = mair.get_model('mair/realDenoising/options/test_MaIR_MotionDeblur.yml', '..')
model = rednet.get_model('../weights/REDNet/50.pt', device=device)
model = restormer.get_model('restormer/options/Deblurring_Restormer.yml', device=device)
