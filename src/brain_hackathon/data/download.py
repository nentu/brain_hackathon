import os

import wget

URL = "https://box.almazovcentre.ru/index.php/s/ZD0MIbCoKEBMziJ/download"


def cache_download(out_dir="./data"):
    out_path = os.path.join(out_dir, "data")
    if not os.path.isfile(out_path):
        print("Downloading data")
        filename = wget.download(URL, out=out_dir)
    return out_path


def main():
    filename = cache_download()
    print(filename)
