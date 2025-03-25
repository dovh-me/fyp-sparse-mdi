import glob
import os

def remove_dir_contents(path):
    filePaths = glob.glob(f"{path}/**")

    for filePath in filePaths:
        os.remove(filePath)
