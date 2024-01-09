import glob
from PIL import Image
import pretrainedmodels
from tqdm import tqdm
from fastai import *
from fastai.vision import *
import os 
import pandas as pd


def load_data(path_to_data):
   # setting the path for joining multiple files
   #files = os.path.join(path_to_data,"*.csv")
   # list of merged files returned
   #files = glob.glob(files)
   df = pd.read_csv(path_to_data)

   imagePatches = glob.glob("./data1_image/*/*.*", recursive=True)
   return df, imagePatches