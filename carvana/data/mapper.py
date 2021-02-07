import copy
import torch
import numpy as np
from detectron2.data import detection_utils as utils


def mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = np.stack(
        [utils.read_image(img, format="BGR") for img in dataset_dict["filename"]]
    )
    image = torch.from_numpy(image)
    annos = [
        # utils.transform_instance_annotations(annotation, [], image.shape[1:])
        annotation
        for annotation in dataset_dict.pop("annotations")
    ]
    # The model's input
    return {
        "image": image,
        "instances": utils.annotations_to_instances(
            annos, image.shape[2:], mask_format="bitmap"
        ),
    }
