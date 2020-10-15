import config
from model import resnet_rnn,baseline,vgg_extractor,tf2cv_extractor
from data import encode_single_file,preprocess_label
import argparse 
from glob import glob
import numpy as np 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
from decode import ctc_to_text
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.models import Model
import cv2 
parser=argparse.ArgumentParser()
parser.add_argument('--in_dir',default=config.img_dir,help='The directory where the cropped pictures are stored')
args=parser.parse_args()

image_files=np.array(sorted(glob(f'{args.in_dir}*.jpg')))
labels=[i.split('_')[-1].split('.')[0] for i in image_files]
label_lens=np.array([len(i) for i in labels])

reverse_tokenizer=layers.experimental.preprocessing.StringLookup(vocabulary=config.vocab,invert=True)


"""train_data=tf.data.Dataset.from_tensor_slices((image_files,labels,label_lens))
train_data=(train_data.map(
  encode_single_file,num_parallel_calls=tf.data.experimental.AUTOTUNE).
  padded_batch(1)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  )"""


#model=tf2cv_extractor(config.input_shape,n_classes=config.n_classes,model_name='bn_vgg13',hidden_size=512)
model=tf2cv_extractor(config.input_shape,config.n_classes,config.model_name,config.hidden_size) 
model.load_weights(config.model_dir+f'{config.model_name}_{1}.hdf5')
model=Model(inputs=[model.get_layer('image').input],outputs=[model.get_layer('dense2').output])

"""for batch in train_data.take(1):
  images=batch['image'].numpy()
  labels=batch['label'].numpy()
  results=[]
  for label in labels:
    label=tf.strings.reduce_join(reverse_tokenizer(label)).numpy().decode("utf-8")
    results.append(label)
  preds=model.predict(images)
  txt,res=ctc_to_text(preds,reverse_tokenizer)
  print(f'Original text: {results}')
  print(f'Predictions: {txt}')
  print(f'Indices: {res}')
  break """
idx=-1
data=encode_single_file(image_files[idx],labels[idx],label_lens[idx])
img,label,label_len=data['image'],data['label'],data['label_length']
img=img.numpy()[np.newaxis,:,:,:]
preds=model.predict(img)
txt,res=ctc_to_text(preds,reverse_tokenizer)
print(f'Original text: {labels[idx]}')
print(f'Predictions: {txt}')
print(f'Indices: {res}')


