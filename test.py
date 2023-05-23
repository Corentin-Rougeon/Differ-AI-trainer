import os
from pathlib import Path
from shutil import move
from os import mkdir

from setuptools._distutils.dir_util import copy_tree

class1 = "glacier"
class2 = "montagne"
n = 100

folder_path1 = Path(f'img/{class1}')
folder_path2 = Path(f'img/{class2}')

dirpath = "data/test"

folder_paths = list(folder_path1.iterdir())+list(folder_path2.iterdir())

print(folder_paths)

for c in [class1,class2]:
    if not os.path.isdir(f"{dirpath}/img/train/{c}"):
        mkdir(f"{dirpath}/img/train/{c}")
    if not os.path.isdir(f"{dirpath}/img/validation/{c}"):
        mkdir(f"{dirpath}/img/validation/{c}")

for idx, item in enumerate(folder_paths):
    if item.is_file() and (item.suffix == ".jpeg" or item.suffix == ".png"):
        move(item, f"{dirpath}/img/train/{item.parent.name}")
    else:
        os.remove(item)


copy_tree(f"{dirpath}/img/train",f"{dirpath}/img/validation")

os.rmdir(f"img/{class1}")
os.rmdir(f"img/{class2}")