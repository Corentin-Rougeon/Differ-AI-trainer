# -*- coding: utf-8 -*-
import json
import os
import concurrent.futures
from pathlib import Path
from shutil import move
from PIL import Image

from setuptools._distutils.dir_util import copy_tree

from scrapper.GoogleImageScraper import GoogleImageScraper
from scrapper.patch import webdriver_executable

def run_image_scraper(dirpath="",class1="",class2="",n=1):
    def worker_thread(search_key):
        image_scraper = GoogleImageScraper(
            webdriver_path,
            image_path,
            search_key,
            number_of_images,
            headless,
            min_resolution,
            max_resolution,
            max_missed)
        image_urls = image_scraper.find_image_urls()
        image_scraper.save_images(image_urls, keep_filenames)
        del image_scraper


    webdriver_path = os.path.normpath(os.path.join(os.getcwd(), 'scrapper/webdriver', webdriver_executable()))
    image_path = os.path.normpath(os.path.join(os.getcwd(), f"./img"))

    search_keys = list(set([class1, class2]))
    number_of_images = n
    headless = True
    min_resolution = (0, 0)
    max_resolution = (9999, 9999)
    max_missed = 40
    number_of_workers = 2
    keep_filenames = False

    with concurrent.futures.ThreadPoolExecutor(max_workers=number_of_workers) as executor:
        executor.map(worker_thread, search_keys)





    folder_path1 = Path(f'img/{class1}')
    folder_path2 = Path(f'img/{class2}')

    folder_paths = list(folder_path1.iterdir()) + list(folder_path2.iterdir())

    for c in [class1, class2]:
        if not os.path.isdir(f"{dirpath}/img/train/{c}"):
            os.mkdir(f"{dirpath}/img/train/{c}")
        if not os.path.isdir(f"{dirpath}/img/validation/{c}"):
            os.mkdir(f"{dirpath}/img/validation/{c}")

    imgC = 0

    for idx, item in enumerate(folder_paths):
        if item.is_file() and (item.suffix == ".jpeg"):
            img = Image.open(item)
            img = img.resize((150, 150))
            img.save(item)
            imgC += 1
            move(item, f"{dirpath}/img/train/{item.parent.name}")
        else:
            os.remove(item)

    copy_tree(f"{dirpath}/img/train", f"{dirpath}/img/validation")

    with open(f"{dirpath}/meta.json", "r+") as f:
        data = f.read()
        data = json.loads(data)

        data["classes"] = [class1,class2]
        data["img_count"] = imgC
        data["has_img"] = True

        f.seek(0)
        f.write(json.dumps(data, indent=4))
        f.truncate()

    os.rmdir(f"img/{class1}")
    os.rmdir(f"img/{class2}")
