import chicken
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from copy import deepcopy
import matplotlib.pyplot as plt
import random

def to_tensor(image):
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])
    return trans(image)

def show_tensor(tensor_image):
	plt.imshow(tensor_image.permute(1, 2, 0))
	plt.show()

def load_data(width_height=192, num_points=1, input_directory='data/inputs/', mask_directory='data/masks/'):

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
						new_rgb = WINDOW
						new_picture[x][y][3] = 1
					else:
						new_picture[x][y][0] = 1
			new_list.append(new_picture)
		return new_list

	print('Preparing input dataset...')

	inputs = chicken.get_images(input_directory)
	inputs = inputs + chicken.fliplr(inputs)
	inputs = chicken.resize_and_smart_crop_square(inputs, width_height)
	inputs = inputs[:num_points]
	inputs = np.array(inputs)

	print('Preparing mask dataset...')

	masks = chicken.get_images(mask_directory)
	masks = masks + chicken.fliplr(masks)
	masks = chicken.resize_and_smart_crop_square(masks, width_height)
	masks = masks[:num_points]
	masks = simplify_masks(masks)
	masks = np.array(masks)

	print('Data loaded')

	return inputs, masks

def get_PT_dataloaders(width_height=192, num_points=1, batch_size=1, input_directory='inputs_overfit/', label_directory='labels_overfit/'):

	BACKGROUND = np.array([0, 0, 170])
	WALL = np.array([0, 0, 255])
	DOOR = np.array([0, 170, 255])
	WINDOW = np.array([0, 85, 255])

	classes = [
		BACKGROUND, # BACKGROUND
		WALL, # WALL
		DOOR, # DOOR
		WINDOW # WINDOW
	]
	classes = np.array(classes)


	def simplify_label(label):
		label = np.array(label)
		width, height, _ = label.shape
		one_hot_label = np.zeros((width, height, 4))

		for x in range(width):
			for y in range(height):
				rgb = label[x][y]
				if np.allclose(rgb, WALL):
					one_hot_label[x][y][1] = 1
				elif np.allclose(rgb, DOOR):
					one_hot_label[x][y][2] = 1
				elif np.allclose(rgb, WINDOW):
					one_hot_label[x][y][3] = 1
				else:
					one_hot_label[x][y][0] = 1
		return one_hot_label

	preprocess_transform = transforms.Compose([
									transforms.Resize(width_height),
									transforms.CenterCrop(width_height),
									transforms.ToTensor()
									]) # TODO: Normalize with ImageNet mean/std
	'''
	T.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])])
 	'''
	lazy_label_transform = transforms.Compose([
									transforms.ToPILImage(mode=None),
									transforms.Lambda(simplify_label),
									transforms.ToTensor()])

	flip_transform = transforms.Compose([
		transforms.ToPILImage(),
		transforms.RandomHorizontalFlip(p=1.0),
		transforms.ToTensor()])

	class SegmentDataset(Dataset):
		def __init__(self, inputs, masks, random_flip=True):
			self.input_images = inputs
			self.target_masks = masks

		def __len__(self):
			return len(self.input_images)

		def __getitem__(self, index):
			image = self.input_images[index]
			mask = self.target_masks[index]

			do_flip = random.random() > 0.5
			# do_flip = True
			if do_flip:
				# image = transforms.ToPILImage()(image)
				# image = transforms.RandomHorizontalFlip(p=1.0)(image)
				# image = transforms.ToTensor()(image)

				# mask = transforms.ToPILImage()(mask)
				# mask = transforms.RandomHorizontalFlip(p=1.0)(mask)
				# mask = transforms.ToTensor()(mask)

				image = flip_transform(image)
				mask = flip_transform(mask)

				# image = transforms.functional.to_pil_image(image)
				# image = transforms.functional.hflip(image)
				# image = transforms.functional.to_tensor(image)

				# mask = transforms.functional.to_pil_image(mask)
				# mask = transforms.functional.hflip(mask)
				# mask = transforms.functional.to_tensor(mask)

			mask = lazy_label_transform(mask)
				

			return [image, mask]
	
	# Remove pytorch integer labels
	print('Loading inputs...')
	inputs = list(zip(*datasets.ImageFolder(root=input_directory, transform=preprocess_transform)))[0]

	print('Loading labels...')
	labels = list(zip(*datasets.ImageFolder(root=label_directory, transform=preprocess_transform)))[0]

	# NUM_EVAL_IMAGES = len(inputs) // 10 # Designate 10% of dataset to val set
	NUM_EVAL_IMAGES = max(len(inputs) // 10, 10) # Designate 10% of dataset to val set

	if len(inputs) > NUM_EVAL_IMAGES:
		train_images = inputs[:-NUM_EVAL_IMAGES]
		train_labels = labels[:-NUM_EVAL_IMAGES]
		val_images = inputs[-NUM_EVAL_IMAGES:]
		val_labels = labels[-NUM_EVAL_IMAGES:]
	else:
		train_images = inputs
		train_labels = labels
		val_images = inputs
		val_labels = labels

	train_set = SegmentDataset(train_images, train_labels, random_flip=True)
	# train_set = SegmentDataset(inputs[-NUM_EVAL_IMAGES:], labels[-NUM_EVAL_IMAGES:], random_flip=True)
	val_set = SegmentDataset(val_images, val_labels, random_flip=True)

	image_datasets = {
		'train': train_set,
		'val': val_set
	}

	dataloaders = {
		'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
		'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0),
	}

	return dataloaders, image_datasets

def get_dataloaders(width_height=192, num_points=1, batch_size=1):

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

			if self.transform:
				image = self.transform(image)

			image = to_tensor(image)
			mask = to_tensor(mask)

			return [image, mask]

	
	inputs, masks = load_data(width_height, num_points)

	NUM_EVAL_IMAGES = max(len(inputs) // 10, 10) # Designate 10% of dataset to val set

	if len(inputs) > NUM_EVAL_IMAGES:
		train_images = inputs[:-NUM_EVAL_IMAGES]
		train_labels = masks[:-NUM_EVAL_IMAGES]
		val_images = inputs[-NUM_EVAL_IMAGES:]
		val_labels = masks[-NUM_EVAL_IMAGES:]
	else:
		train_images = inputs
		train_labels = masks
		val_images = inputs
		val_labels = masks

	train_set = SegmentDataset(train_images, train_labels, transform=None)
	val_set = SegmentDataset(val_images, val_labels, transform=None)

	# train_set = SegmentDataset(inputs[:-NUM_EVAL_IMAGES], masks[:-NUM_EVAL_IMAGES], transform=None)
	# # train_set = SegmentDataset(inputs[-NUM_EVAL_IMAGES:], masks[-NUM_EVAL_IMAGES:], transform=None)
	# val_set = SegmentDataset(inputs[-NUM_EVAL_IMAGES:], masks[-NUM_EVAL_IMAGES:], transform=None)

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

def check_largest_dimensions(input_directory='data/masks/'):
	inputs = chicken.get_images(input_directory)


	largest_w = 0
	largest_h = 0
	for image in inputs:
		shape = image.shape
		width = shape[0]
		height = shape[1]

		largest_w = max(width, largest_w)
		largest_h = max(height, largest_h)

	print(largest_w, largest_h)
