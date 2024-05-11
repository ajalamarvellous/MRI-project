import os
import csv

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

pos_tags = ["DHFDLAFH", "FJKAHFBC", "DHAJFVIF", "SHDCBDJEL"]

def get_csv(dir: str, file_loc: str, pos_tags: list=pos_tags) -> None:
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
            if (dir_.split("/")[-1] in pos_tags or dir_.split("/")[-2] in pos_tags):
                # save to file if it's a dcm image with the positive tag
                rows = [(f"{dir_}/{file}", 1) for file in files if file.endswith(".dcm")]
                csv_writer.writerows(rows) 
            # Other control examples
            else:
                # save to file with negative tag if it's also a dcm image
                rows = [(f"{dir_}/{file}", 0) for file in files if file.endswith(".dcm")]
                csv_writer.writerows(rows) 


def split_dataset(file_dir: str, test_size: float=0.3, 
                  val_size: float=0.2, random_state: int=42) -> tuple:
    """
    Split the dataset into 3 places, trainset, testset and validation set

    Parameter(s)
    --------------
    file_dir : str
        location of the datasource
    test_size : float
        fraction of the dataset to keep as testset
    val_size : float
        fraction of the dataset to keep as valset
    random_state : int
        random state initialiser to ensure reproducibility

    Return(s)
    ----------
    split_data : dict({str:tuple})
        split data in format "set:(X, y)
    """
    df = pd.read_csv(file_dir)
    X, X_val, y, y_val = train_test_split(
        df["location"], df["label"], stratify=df["label"], 
        test_size=val_size, shuffle=True, random_state=random_state
        )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size/(1-val_size), 
        shuffle=True, random_state=random_state
    )
    return {"train_set":(X_train, y_train), 
            "test_set":(X_test, y_test), 
            "val_set":(X_val, y_val)}

if __name__ == "__main__":
    data_dir = "../data"
    file_dir = "../data/dataset.csv"
    get_csv(data_dir, file_dir)
    data_split = split_dataset(file_dir)
    train_set = data_split['train_set']
    test_set = data_split['test_set']
    val_set = data_split['val_set']
    print(f"Train set: {train_set[0].shape, train_set[1].shape} \
            Test set: {test_set[0].shape, test_set[1].shape} \
            Val set : {val_set[0].shape, val_set[1].shape}")