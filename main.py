import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from collections import defaultdict
import torch.nn.functional as F
import numpy as np

import chicken
from resnet_unet import ResNetUNet, train_model, calc_loss, print_metrics, dice_loss
from data_loader import get_dataloaders, tensor_to_np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)

NUM_CLASSES = 4
DIM = 192
NUM_POINTS = 1

MODEL_PATH = 'checkpoint.pth'
NUM_EPOCHS = 1000
LEARNING_RATE = 0.01
LEARNING_RATE_DECAY = 1.0
BATCH_SIZE = 1



dataloaders, image_datsets = get_dataloaders(width_height=DIM,
							  num_points=NUM_POINTS,
							  batch_size=BATCH_SIZE)
model = ResNetUNet(NUM_CLASSES).to(device)



if NUM_EPOCHS > 0:

	model.load(MODEL_PATH)

	optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=LEARNING_RATE_DECAY)

	model = train_model(model, optimizer_ft, exp_lr_scheduler, NUM_EPOCHS, dataloaders, device)

else:

	model.load(MODEL_PATH)
	model.eval()

	data_iter = iter(dataloaders['val'])
	inputs, labels = next(data_iter)
	inputs = inputs.to(device)
	labels = labels.to(device)

	predictions = model(inputs)

	metrics = defaultdict(float)
	epoch_samples = inputs.size(0)
	loss = calc_loss(predictions, labels, metrics)
	print_metrics(metrics, epoch_samples, 'val')

	predictions = F.sigmoid(predictions)





	import helper
	input_images = [tensor_to_np(x) for x in inputs]
	mask_images = [helper.masks_to_colorimg(x) for x in labels]
	prediction_images = [helper.masks_to_colorimg(x) for x in predictions]
	chicken.display_all(input_images + mask_images + prediction_images)



	def pred_to_channels(single_pred):

	    channel_images = []
	    # z = np.zeros(pix.shape)

	    for pix in single_pred:
	        z = np.zeros(pix.shape)

	        channel = np.stack([pix, z, z], axis=2)
	        channel_images.append(channel)

	    return np.array(channel_images)

	channel_images = pred_to_channels(predictions[0])
	chicken.display_all(channel_images)


