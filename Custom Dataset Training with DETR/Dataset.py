import os

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A


class VisDroneDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transforms=None):
        super(VisDroneDataset, self).__init__()
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms
        self.images = os.listdir(images_dir)
        if "9999985_00000_d_0000020.jpg" in self.images:
            del self.images[self.images.index("9999985_00000_d_0000020.jpg")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = self.images_dir + "/" + image_name
        annotation_path = self.annotations_dir + "/" + image_name[:-3] + "txt"

        # Image
        image = Image.open(image_path)

        image = np.array(image, dtype=np.float32)
        image /= 255.0

        # Bounding boxes, labels, areas and scores
        boxes = []
        labels = []
        areas = []
        scores = []
        with open(annotation_path, "r") as f:

            for line in f.readlines():
                x, y, w, h, s, l = line.split(",")

                boxes.append([int(x), int(y), int(w), int(h)])
                labels.append(int(l))
                scores.append(int(s))
                area = int(w) * int(h)
                areas.append(area)

        # Transformations
        if self.transforms:
            sample = {
                'image': image,
                'bboxes': boxes,
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            boxes = sample['bboxes']
            labels = sample['labels']

        # Normalizing BBoxes
        _, h, w = image.shape
        boxes = A.augmentations.bbox_utils.normalize_bboxes(boxes, rows=h, cols=w)

        # Storing in targets dictionary
        targets = {
            "boxes": torch.tensor(boxes, dtype=torch.float),
            "labels": torch.tensor(labels, dtype=torch.long),
            # "scores": torch.tensor(scores),
            "area": torch.tensor(areas, dtype=torch.float),
        }

        return image, targets
