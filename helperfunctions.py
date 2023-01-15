import numpy as np
from scipy import ndimage


def simple_stitch(image_map: dict[str, np.ndarray], 
                  image_positions: dict[str, tuple[int, int]],
                  output_shape: tuple[int, int],
                  projection_type: str) -> np.ndarray:
    """simple max projection between tiles"""
    shifted_images = []
    for image in image_map.keys():
        im_shifted = ndimage.affine_transform(
            input = image_map[image],
            matrix = np.eye(2),
            offset = -np.array([0,0]) - np.array(image_positions[image]),
            output_shape = output_shape
        )
        shifted_images.append(im_shifted)
    if projection_type == 'max':
        return np.max(np.stack(shifted_images, axis = 2), axis = 2)
    if projection_type == 'mean':
        return np.mean(np.stack(shifted_images, axis = 2), axis = 2)
    if projection_type == 'sum':
        return np.sum(np.stack(shifted_images, axis = 2), axis = 2)