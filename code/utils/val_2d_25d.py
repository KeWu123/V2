"""Volume validation for a three-slice 2.5D network."""

import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        return metric.binary.dc(pred, gt), metric.binary.hd95(pred, gt)
    return 0, 0


def test_single_volume(image, label, model, classes, patch_size=(256, 256)):
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    depth = image.shape[0]

    model.eval()
    for index in range(depth):
        indices = (max(index - 1, 0), index, min(index + 1, depth - 1))
        context = np.stack([image[z] for z in indices], axis=0)
        height, width = context.shape[1:]
        context = zoom(
            context,
            (1, patch_size[0] / height, patch_size[1] / width),
            order=0,
        )
        input_tensor = torch.from_numpy(context).unsqueeze(0).float().cuda()
        with torch.no_grad():
            output = model(input_tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
            output = torch.argmax(torch.softmax(output, dim=1), dim=1)
            output = output.squeeze(0).cpu().detach().numpy()
        prediction[index] = zoom(
            output,
            (height / patch_size[0], width / patch_size[1]),
            order=0,
        )

    return [
        calculate_metric_percase(prediction == class_index, label == class_index)
        for class_index in range(1, classes)
    ]
