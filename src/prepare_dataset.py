import logging
import os
from typing import Any, List

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset

logging.basicConfig(
    level=logging.INFO,
    # fmt: off
    format="%(asctime)s %(funcName)s[%(levelname)s]: %(message)s",
    # fmt: on
)
logger = logging.getLogger()
pos_tags = ["DHFDLAFH", "FJKAHFBC", "DHAJFVIF", "SHDCBDJEL"]


class DicomDataset(Dataset):
    def __init__(
        self,
        filename: str = "",
        dir_location: str = "",
        pos_tags: List[Any] = [],
        transform: bool = True,
    ):
        # Read from file if filename is provided
        if filename != "":
            self.file = self.read_csv(filename)
        else:
            # else read from the directory after confirming dir_loc and pos tags
            assert (dir_location != "") & (
                pos_tags != []
            ), "Please provide the images directory and pos tags"

            self.file = self.create_dataframe(dir_location, pos_tags)
        self.transform = transform

    def __len__(self):
        "Return the length of the dataset"
        return self.file.shape[0]

    def read_csv(self, filename: str) -> pd.DataFrame:
        "Read the file address and labels from csv file"
        return pd.read_csv(filename)

    def create_dataframe(self, dir: str, pos_tags: List[Any]) -> pd.DataFrame:
        """
        Create a pandas DataFrame that contains the location of the images
        and their label
        """
        data = list()
        for dir_, folder, files in os.walk(dir):
            # check if it's postive image folder
            if dir_.split("/")[-1] in pos_tags or dir_.split("/")[-2] in pos_tags:
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
        logger.info("All files read into Dataset and ready to use...")
        return pd.DataFrame(data, columns=["location", "label"])

    def read_dicom(self, file_name: str) -> np.ndarray:
        "Read the custom dicom file provided"
        dicom = pydicom.read_file(file_name)
        data = dicom.pixel_array
        logger.debug("Dicom file read successfully...")
        return data

    def transform_fn(self, data: np.ndarray) -> torch.Tensor:
        """
        Transform the data(image) by randomly rotate, resize to 224x224
        stardardise, normalise, convert to 1 filter image and to tensors
        """
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
        logger.debug("data transformed successfully")
        return torch.from_numpy(data)

    def __getitem__(self, index: int) -> tuple:
        "Return image data and label given the index"
        file_loc, target = self.file.iloc[index, 0], self.file.iloc[index, 1]
        data = self.read_dicom(file_loc)
        if self.transform:
            data = self.transform_fn(data)
        logger.debug(f"returning item {data.shape, target.shape}...")
        return data, target


if __name__ == "__main__":
    # data_dir = "../data/images"
    file_dir = "../data/temp_data.csv"
    dataset = DicomDataset(file_dir)
    y_labels = []
    for i, (x, y) in enumerate(dataset):
        # print(x, y)
        y_labels.append(y)
    print(
        f"No of y: {len(y_labels)}, \n \
          No of positive: {np.sum(y_labels)}"
    )
