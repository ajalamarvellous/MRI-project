import csv
import os
import random

import numpy as np
import torch

pos_tags = ["DHFDLAFH", "FJKAHFBC", "DHAJFVIF", "SHDCBDJEL"]


def get_csv(dir: str, file_loc: str, pos_tags: list = pos_tags) -> None:
    """
    Create a CSV file that matches the file address with the tags
    folders in pos_tags are positive examples of what we are test for while
    others are negative.

    Parameter(s)
    ----------------
    dir : str
        Home directory where the dataset is
    file_loc : str
        Location to save the file
    pos_tags : list
        List of folders containing pos examples

    Return(s)
    -------------
    None
    """
    with open(file_loc, "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["location", "label"])
        for dir_, folder, files in os.walk(dir):
            # check if it's postive image folder
            if dir_.split("/")[-1] in pos_tags or dir_.split("/")[-2] in pos_tags:
                # save to file if it's a dcm image with the positive tag
                rows = [
                    (f"{dir_}/{file}", 1) for file in files if file.endswith(".dcm")
                ]
                csv_writer.writerows(rows)
            # Other control examples
            else:
                # save to file with negative tag if it's also a dcm image
                rows = [
                    (f"{dir_}/{file}", 0) for file in files if file.endswith(".dcm")
                ]
                csv_writer.writerows(rows)


def get_ylabels(dataset):
    """
    Returns the y labels with the index representing their index in the
    dataset
    """
    y_labels = []
    for i, (x, y) in enumerate(dataset):
        y_labels.append(y)
    return y_labels


def seedall(seed):
    """
    Set consistent seed for random, numpy and torch to ensure
    reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    data_dir = "../data/images"
    file_dir = "../data/dataset.csv"
    get_csv(data_dir, file_dir, pos_tags=pos_tags)
