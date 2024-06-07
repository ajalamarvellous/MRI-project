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
        file = pd.read_csv(filename)
        logger.info("Returning read file...")
        return file

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
        return data.type(torch.float), target


def stratify_split(labels: List, test_size: float, train_size: float = 0) -> List[Any]:
    """
    Split the dataset into 3 places, trainset, testset and validation set with each
    set containing the same proportion of positive examples

    Parameter(s)
    --------------
    file_dir : str
        location of the datasource
    test_size : float
        fraction of the dataset to keep as testset
    random_state : int
        random state initialiser to ensure reproducibility

    Return(s)
    ----------
    split_data : list
        index for the various split in the order
        [validation_set, test_set, train_set]
    """
    if train_size == 0:
        train_size = 1 - test_size  # + val_size)
    else:
        total = test_size + train_size
        assert (
            total == 1
        ), f"The data size specified is equals {total} not 1, please change"
    data_size = len(labels)
    logger.debug("data size: ", data_size)
    # get indices for the postive and negative examples
    pos_indices = [i for i, x in enumerate(labels) if x == 1]
    neg_indices = [i for i, x in enumerate(labels) if x == 0]
    # fraction of the positive values in the dataset
    pos_fraction = len(pos_indices) / data_size
    logger.debug(
        f"Len pos {len(pos_indices)}, len neg {len(neg_indices)}, pos fract {pos_fraction}"
    )
    result = []

    for size in [test_size, train_size]:
        # Calculate the number of values for the val set and positve examples in it
        size_ = int(np.floor(size * data_size))

        pos_val_size = int(np.floor(pos_fraction * size_))
        # randomly select the values from the separated indices
        pos_val = np.random.choice(pos_indices, pos_val_size)
        neg_val = np.random.choice(neg_indices, (size_ - pos_val_size))
        logger.debug(f"Pos size: {len(pos_val)}, neg size {len(neg_val)}")

        # remove the selected values from the pos_indices and neg_indices
        pos_indices = list(set(pos_indices).difference(set(pos_val)))
        neg_indices = list(set(neg_indices).difference(set(neg_val)))

        indices = list(pos_val) + list(neg_val)
        np.random.shuffle(indices)
        logger.debug(
            f"final size: {len(indices)}, fraction desired: {size}, \
              fraction given: {(len(indices)/data_size)}, fraction o"
        )
        result.append(indices)
    logger.info(
        f"split completed... \n test: {len(result[0])}, train: {len(result[1])}"
    )
    return (result, pos_fraction)


if __name__ == "__main__":
    # data_dir = "../data/images"
    file_dir = "../data/temp_data.csv"
    dataset = DicomDataset(file_dir)
    y_labels = []
    for i, (x, y) in enumerate(dataset):
        print(f"index: {i}, shape: {x.shape}")
        y_labels.append(y)
    print(
        f"No of y: {len(y_labels)}, \n \
          No of positive: {np.sum(y_labels)}"
    )
    # data_split = stratify_split(y_labels, val_size=0.3, test_size=0.2)
