import os
from pathlib import Path
imgfolder=r"./Pics_Ready"
imagepath_list = list(sorted(Path(imgfolder).glob("*/*.jpg")))
print(imagepath_list)