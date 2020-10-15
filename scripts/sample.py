import config
#from model import resnet_rnn,baseline
#from data import encode_single_file,preprocess_label
import argparse 
from glob import glob
import numpy as np 
from sklearn.model_selection import KFold
import cv2
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers 
#from decode import ctc_to_text
#from tensorflow.keras.models import load_model
#from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
parser=argparse.ArgumentParser()
parser.add_argument('--in_dir',default=config.img_dir,help='The directory where the cropped pictures are stored')
args=parser.parse_args()

image_files=np.array(sorted(glob(f'{args.in_dir}*.jpg')))
labels=np.array([i.split('_')[-1].split('.')[0] for i in image_files])
characters=list(set(char for label in labels for char in label))

sizes=[cv2.imread(i).shape for i in image_files]
min_height=min([i[0] for i in sizes]) 
min_width=min([i[1] for i in sizes]) 

print(min_height,min_width)