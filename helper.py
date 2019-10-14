import matplotlib.pyplot as plt
import numpy as np
from chicken import display_all


def plot_img_array(img_array, ncol=3):
    nrow = len(img_array) // ncol

    f, plots = plt.subplots(nrow, ncol, sharex='all', sharey='all', figsize=(ncol * 4, nrow * 4))

    for i in range(len(img_array)):
        plots[i // ncol, i % ncol]
        plots[i // ncol, i % ncol].imshow(img_array[i])

from functools import reduce
def plot_side_by_side(img_arrays):
    flatten_list = reduce(lambda x,y: x+y, zip(*img_arrays))

    plot_img_array(np.array(flatten_list), ncol=len(img_arrays))

import itertools
def plot_errors(results_dict, title):
    markers = itertools.cycle(('+', 'x', 'o'))

    plt.title('{}'.format(title))

    for label, result in sorted(results_dict.items()):
        plt.plot(result, marker=next(markers), label=label)
        plt.ylabel('dice_coef')
        plt.xlabel('epoch')
        plt.legend(loc=3, bbox_to_anchor=(1, 0))

    plt.show()

def masks_to_colorimg(masks):
    # colors = np.asarray([(201, 58, 64), (242, 207, 1), (0, 152, 75), (101, 172, 228),(56, 34, 132), (160, 194, 56)])
    masks = masks.cpu()
    colors = np.asarray([(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)])

    colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    channels, height, width = masks.shape

    for y in range(height):
        for x in range(width):
            # selected_colors = colors[masks[:,y,x] > 0.5]
            # # selected_colors = colors[True]

            # if len(selected_colors) > 0:
            #     colorimg[y,x,:] = np.mean(selected_colors, axis=0)

            colorimg[y,x,:] = colors[np.argmax(masks[:,y,x].detach().numpy())]
                # colorimg[y,x,:] = np.max(selected_colors, axis=0)

    return colorimg.astype(np.uint8)


def input_tensors_to_colorimg(inputs):
    from data_loader import tensor_to_np
    return [tensor_to_np(x) for x in inputs]

def label_tensors_to_colorimg(labels):
    return [masks_to_colorimg(x) for x in labels]

def show_prediction_channels(prediction_tensor):
    channel_images = []
    for chan in prediction_tensor:
        z = np.zeros(chan.shape)
        channel = np.stack([chan.cpu().detach().numpy(), z, z], axis=2)
        channel_images.append(channel)

    channels_images = np.array(channel_images)
    display_all(channel_images)

