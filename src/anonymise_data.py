import os
import shutil
from pathlib import Path

import yaml

DATA = Path("../data/images")


def load_yaml(folder: str | Path) -> dict:
    f = open(folder, "r")
    maps = yaml.safe_load(f)
    f.close()
    return maps


def anonymize(maps: dict, folder: str, home: Path = DATA) -> None:
    previous_address = home / folder
    new_address = home / maps[folder]
    shutil.move(previous_address, new_address)
    print(f"Data previously in folder {folder[:3]} now located in {new_address}")


def main():
    print(f"Current location: {Path(__file__)}")
    maps = load_yaml(DATA.parent / "anonymise.yaml")
    for folder in os.listdir(DATA.__str__()):
        if not (folder.endswith("yaml") or folder == ".DS_Store"):
            anonymize(maps, folder)


if __name__ == "__main__":
    main()
