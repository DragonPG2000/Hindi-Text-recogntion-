import config
from model import resnet_rnn,baseline,tf2cv_extractor,inception_extractor
from data import encode_single_file,data_augment
from cyclic_lr import CyclicLR
import argparse 
from glob import glob
import numpy as np 
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
from decode import ctc_to_text
from tensorflow.keras.models import load_model,Model 
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
parser=argparse.ArgumentParser()
parser.add_argument('--in_dir',default=config.img_dir,help='The directory where the cropped pictures are stored')
args=parser.parse_args()

image_files=np.array(sorted(glob(f'{args.in_dir}*.jpg')))
np.random.shuffle(image_files)
labels=np.array([i.split('_')[-1].split('.')[0] for i in image_files])
label_lens=np.array([len(i) for i in labels])
reverse_tokenizer=layers.experimental.preprocessing.StringLookup(vocabulary=config.vocab,invert=True)

splits=KFold().split(image_files,labels)
fold_accuracies=[]
for i,(train_idx,val_idx) in enumerate(splits):
  print(f'Fold {i+1}')
  train_data=tf.data.Dataset.from_tensor_slices((image_files[train_idx],labels[train_idx],label_lens[train_idx]))
  val_data=tf.data.Dataset.from_tensor_slices((image_files[val_idx],labels[val_idx],label_lens[val_idx]))
  train_data=(train_data.map(
  encode_single_file,num_parallel_calls=tf.data.experimental.AUTOTUNE).
  map(data_augment,num_parallel_calls=tf.data.experimental.AUTOTUNE).
  padded_batch(config.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  )
  val_data=(val_data.map(
  encode_single_file,num_parallel_calls=tf.data.experimental.AUTOTUNE).
  padded_batch(config.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  )
  model=tf2cv_extractor(config.input_shape,config.n_classes,config.model_name)
  checkpoint=ModelCheckpoint(config.model_dir+f'{config.model_name}_{i+1}.hdf5',save_weights_only=True,save_best_only=True) 
  early=EarlyStopping(patience=5)
  n_steps=len(image_files)//config.batch_size 
  cycle=CyclicLR(base_lr=config.base_lr,max_lr=config.max_lr,step_size=n_steps*2,scale_mode='exp_range')
  callbacks=[checkpoint,early,cycle]
  model.fit(train_data,epochs=config.epochs,validation_data=val_data,callbacks=callbacks)
  model.load_weights(config.model_dir+f'{config.model_name}_{1}.hdf5')
  model=Model(inputs=[model.get_layer('image').input],outputs=[model.get_layer('dense2').output])
  preds=model.predict(val_data)
  txt,res=ctc_to_text(preds,reverse_tokenizer)
  tp=(txt==labels[val_idx]).astype(bool)
  accuracy=sum(tp)/len(labels[val_idx])
  fold_accuracies.append(accuracy)
  print(f'Accuracy for fold {i+1} is {accuracy}')
avg_accuracy=sum(fold_accuracies)/len(fold_accuracies)
print(f'The 5 fold average accuracy is {avg_accuracy}')