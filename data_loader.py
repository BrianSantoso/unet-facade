import chicken
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from copy import deepcopy


class SegmentDataset(Dataset):
		def __init__(self, inputs, masks, transform=None):
			self.input_images = np.array(inputs)
			self.target_masks = np.array(masks)
			self.transform = transform

		def __len__(self):
			return len(self.input_images)

		def __getitem__(self, index):
			image = self.input_images[index]
			mask = self.target_masks[index]

			image = to_tensor(image)
			mask = to_tensor(mask)

			if self.transform:
				image = self.transform(image)

			return [image, mask]

def to_tensor(image):
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])
    return trans(image)

def simplify_masks(masks):
	# Merge mask classes
	BACKGROUND = np.array([0, 0, 170])
	WALL = np.array([0, 0, 255])
	DOOR = np.array([0, 170, 255])
	WINDOW = np.array([0, 85, 255])

	new_list = []
	for picture in masks:
		# new_picture = deepcopy(picture)
		width, height, _ = picture.shape
		new_picture = np.zeros((width, height, 4))
		for x in range(width):
			for y in range(height):
				rgb = picture[x][y]
				if np.allclose(rgb, WALL):
					new_picture[x][y][1] = 1
				elif np.allclose(rgb, DOOR):
					new_picture[x][y][2] = 1
				elif np.allclose(rgb, WINDOW):
					new_rgb = deepcopy(WINDOW)
					new_picture[x][y][3] = 1
				else:
					new_picture[x][y][0] = 1
		new_list.append(new_picture)
	return new_list

def load_data(width_height=192, num_points=1):

	print('Preparing input dataset...')

	inputs = chicken.get_images('data/inputs_overfit/')
	inputs = inputs + chicken.fliplr(inputs)
	inputs = inputs[:num_points]
	inputs = chicken.resize_and_smart_crop_square(inputs, width_height)
	inputs = np.array(inputs)

	print('Preparing mask dataset...')

	masks = chicken.get_images('data/masks_overfit/')
	masks = masks + chicken.fliplr(masks)
	masks = masks[:num_points]
	masks = chicken.resize_and_smart_crop_square(masks, width_height)
	masks = simplify_masks(masks)
	masks = np.array(masks)

	print('Data loaded')

	return inputs, masks

def get_dataloaders(width_height=192, num_points=1, batch_size=1):

	NUM_EVAL_IMAGES = 200
	inputs, masks = load_data(width_height, num_points)

	# train_set = SegmentDataset(inputs[:-NUM_EVAL_IMAGES], masks[:-NUM_EVAL_IMAGES], transform=None)
	train_set = SegmentDataset(inputs[-NUM_EVAL_IMAGES:], masks[-NUM_EVAL_IMAGES:], transform=None)
	val_set = SegmentDataset(inputs[-NUM_EVAL_IMAGES:], masks[-NUM_EVAL_IMAGES:], transform=None)

	image_datasets = {
		'train': train_set,
		'val': val_set
	}

	dataloaders = {
		'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
		'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0),
	}

	print('Dataloaders created')

	return dataloaders, image_datasets


def tensor_to_np(single_tensor):
	# C, H, W
	x = single_tensor.cpu().numpy()
	x = x.transpose(1, 2, 0) # H, W, C

	return x
