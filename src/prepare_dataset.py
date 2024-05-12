import os

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset

pos_tags = ["DHFDLAFH", "FJKAHFBC", "DHAJFVIF", "SHDCBDJEL"]


class DicomDataset(Dataset):
    def __init__(self, dir: str, pos_tags: list, transform: bool = True):
        self.dir = dir
        self.pos_tags = pos_tags
        self.file = self.read_csv()
        self.transform = transform

    def __len__(self):
        return self.file.shape[0]

    def read_csv(self) -> pd.DataFrame:
        """
        Create a pandas DataFrame that contains the location of the images
        and their label
        """
        data = list()
        for dir_, folder, files in os.walk(self.dir):
            # check if it's postive image folder
            if (
                dir_.split("/")[-1] in self.pos_tags
                or dir_.split("/")[-2] in self.pos_tags
            ):
                # save to file if it's a dcm image with the positive tag
                rows = [
                    (f"{dir_}/{file}", 1) for file in files if file.endswith(".dcm")
                ]
                data.extend(rows)
            # Other control examples
            else:
                # save to file with negative tag if it's also a dcm image
                rows = [
                    (f"{dir_}/{file}", 0) for file in files if file.endswith(".dcm")
                ]
                data.extend(rows)
        return pd.DataFrame(data, columns=["location", "label"])

    def read_dicom(self, file_name: str) -> np.ndarray:
        dicom = pydicom.read_file(file_name)
        data = dicom.pixel_array
        return data

    def transform_fn(self, data: np.ndarray) -> torch.Tensor:
        ROTATION = [
            None,
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_180,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        ]
        # random horizontal rotation
        choice = np.random.choice(ROTATION)
        if choice is not None:
            data = cv2.rotate(data, choice)
        # resize to (224, 224)
        data = cv2.resize(data, (224, 224))
        data = (data - data.mean()) / data.std()
        # normalise
        data = cv2.normalize(data, None, 0, 1, cv2.NORM_MINMAX)
        data = np.expand_dims(data, 0)
        return torch.from_numpy(data)

    def __getitem__(self, index: int) -> tuple:
        file_loc, target = self.file.iloc[index, 0], self.file.iloc[index, 1]
        data = self.read_dicom(file_loc)
        if self.transform:
            data = self.transform_fn(data)
        return data, target


if __name__ == "__main__":
    data_dir = "../data/images"
    # file_dir = "../data/dataset.csv"
    dataset = DicomDataset(data_dir, pos_tags)
    y_labels = []
    for x, y in dataset:
        # print(x, y)
        y_labels.append(y)
    print(
        f"No of y: {len(y_labels)}, \n \
          No of positive: {np.sum(y_labels)}"
    )
