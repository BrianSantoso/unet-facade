import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from collections import defaultdict
import torch.nn.functional as F
import numpy as np

import chicken
from resnet_unet import ResNetUNet, train_model, calc_loss, print_metrics, dice_loss
from data_loader import get_PT_dataloaders, get_dataloaders, tensor_to_np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)

NUM_CLASSES = 4
DIM = 192
NUM_POINTS = 1

MODEL_PATH = 'checkpoint_overfit.pth'
# MODEL_PATH = 'checkpoint_overfit.pth'
NUM_EPOCHS = 0
LEARNING_RATE = 0.01
LEARNING_RATE_DECAY = 0.1
LEARNING_RATE_DECAY_STEP = 1000
BATCH_SIZE = 1

# Use this UNTIL we can figure out an efficient label RGB -> N Channels transform
dataloaders, image_datasets = get_dataloaders(width_height=DIM,
							  num_points=NUM_POINTS,
							  batch_size=BATCH_SIZE)

# Use this once we figure out an efficient label RGB -> N Channels transform
# dataloaders, image_datsets = get_PT_dataloaders(width_height=DIM,
# 							  num_points=NUM_POINTS,
# 							  batch_size=BATCH_SIZE,
# 							  input_directory='inputs/',
# 							  label_directory='labels/')
model = ResNetUNet(NUM_CLASSES).to(device)

if NUM_EPOCHS > 0:

	model.load(MODEL_PATH)

	optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=LEARNING_RATE_DECAY_STEP, gamma=LEARNING_RATE_DECAY)

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
	input_images = helper.input_tensors_to_colorimg(inputs)
	mask_images = helper.label_tensors_to_colorimg(labels)
	prediction_images = helper.label_tensors_to_colorimg(predictions)
	chicken.display_all(input_images + mask_images + prediction_images)

	[helper.show_prediction_channels(p) for p in predictions]

	# chicken.save_to_as(input_images + prediction_images, directory='testoutput/', prefix='img', file_type='jpg')

	# def pred_to_channels(single_pred):

	#     channel_images = []
	#     # z = np.zeros(pix.shape)

	#     for pix in single_pred:
	#         z = np.zeros(pix.shape)

	#         channel = np.stack([pix, z, z], axis=2)
	#         channel_images.append(channel)

	#     return np.array(channel_images)

	# channel_images = pred_to_channels(predictions[0])
	# chicken.display_all(channel_images)


