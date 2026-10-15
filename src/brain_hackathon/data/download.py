import os
import zipfile

import wget

URL = "https://box.almazovcentre.ru/index.php/s/ZD0MIbCoKEBMziJ/download"


def cache_download(out_dir="./data"):
    out_path = os.path.join(out_dir, "data")
    if not os.path.isfile(out_path):
        print("Downloading data")
        wget.download(URL, out=out_dir)
        with zipfile.ZipFile(out_path, "r") as zip_ref:
            zip_ref.extractall(out_dir)
    return os.path.join(out_dir, "Epic and CSCR hospital Dataset")


def main():
    filename = cache_download()
    print(filename)
