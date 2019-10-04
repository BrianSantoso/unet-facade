import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
import time
import copy

'''
	Helpful Links:

	Conv2D layers: https://pytorch.org/docs/stable/nn.html#conv2d
	Upsampling layers: https://pytorch.org/docs/stable/_modules/torch/nn/modules/upsampling.html

'''

def convrelu(in_channels, out_channels, kernel_size, padding):
	return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
		nn.ReLU(inplace=True)
	)

class ResNetUNet(nn.Module):
	def __init__(self, num_classes):
		super().__init__()

		self.base_model = models.resnet18(pretrained=True)
		self.base_layers = list(self.base_model.children())

		self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2 x.W/2)TODO:?
		self.layer0_1x1 = convrelu(64, 64, 1, 0)
		self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
		self.layer1_1x1 = convrelu(64, 64, 1, 0)
		self.layer2 = self.base_layers[5] # size=(N, 128, x.H/8, x.W/8)
		self.layer2_1x1 = convrelu(128, 128, 1, 0)
		self.layer3 = self.base_layers[6] # size=(N, 256, x.H/16, x.W/16)
		self.layer3_1x1 = convrelu(256, 256, 1, 0)
		self.layer4 = self.base_layers[7] # size=(N, 512, x.H/32, x.W/32)
		self.layer4_1x1 = convrelu(512, 512, 1, 0)

		self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

		self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
		self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
		self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
		self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

		self.conv_original_size0 = convrelu(3, 64, 3, 1)
		self.conv_original_size1 = convrelu(64, 64, 3, 1)
		self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

		self.conv_last = nn.Conv2d(64, num_classes, 1)

	def forward(self, input):
		x_original = self.conv_original_size0(input) # N, 64, H, W
		x_original = self.conv_original_size1(x_original) # N, 64, H, W

		layer0 = self.layer0(input) # N, 64, H/2, W/2
		layer1 = self.layer1(layer0) # N, 64, H/4, W/4
		layer2 = self.layer2(layer1) # N, 128, H/8, W/8
		layer3 = self.layer3(layer2) # N, 256, H/16, W/16
		layer4 = self.layer4(layer3) # N, 512, H/32, W/32

		layer4 = self.layer4_1x1(layer4) # N, 512, H/32, W/32
		x = self.upsample(layer4) # N, 512, H/16, W/16
		layer3 = self.layer3_1x1(layer3)  # COPY LAYER 3: N, 256, H/16, W/16
		x = torch.cat([x, layer3], dim=1)  #  N, 512 + 256, H/16, W/16
		x = self.conv_up3(x) # N, 512, H/16?, W/16?

		x = self.upsample(x) # N, 512, H/8, W/8
		layer2 = self.layer2_1x1(layer2) # COPY LAYER 2: N, 128, H/8, W/8
		x = torch.cat([x, layer2], dim=1) # N, 512 + 128, H/8, W/8
		x = self.conv_up2(x) # N, 256,, H/8, W/8

		x = self.upsample(x) # N, 256, H/4, W/4
		layer1 = self.layer1_1x1(layer1) # COPY LAYER 1: N, 64, H/4, W/4
		x = torch.cat([x, layer1], dim=1) # N, 256 + 64, H/4, W/4
		x = self.conv_up1(x) # N, 256, H/4, W/4

		x = self.upsample(x) # N, 256, H/2, W/2
		layer0 = self.layer0_1x1(layer0) # COPY LAYER 0: N, 64, H/2, W/2
		x = torch.cat([x, layer0], dim=1) # N, 256 + 64, H/2, W/2
		x = self.conv_up0(x) # N, 128, H/2, W/2

		x = self.upsample(x) # N, 128, H, W
		x = torch.cat([x, x_original], dim=1) # N, 128 + 64, H, W
		x = self.conv_original_size2(x) # N, 64, H, W

		out = self.conv_last(x) # N, num_classes, H, Ws

		return out

	def load(self, PATH='checkpoint.pth'):
		try:
			self.load_state_dict(torch.load(PATH))
		except FileNotFoundError:
			print('Error while trying to load model. No file found for', PATH)

	def save(self, PATH='checkpoint.pth'):
		torch.save(self.state_dict(), PATH)



def train_model(model, optimizer, scheduler, num_epochs, dataloaders, device):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'checkpoint.pth')

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred.double(), target.double())

    pred = F.sigmoid(pred.double())
    dice = dice_loss(pred.double(), target.double())

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))