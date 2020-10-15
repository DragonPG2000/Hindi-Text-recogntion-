import config 
from glob import glob 
from tensorflow.keras import layers 
import tensorflow as tf
from tensorflow import keras
image_files=sorted(glob(f'{config.img_dir}*.jpg'))
labels=[i.split('_')[-1].split('.')[0] for i in image_files]

characters=list(set(char for label in labels for char in label))
tokenizer=layers.experimental.preprocessing.StringLookup(vocabulary=characters,num_oov_indices=0,mask_token=None,)
tokenizer.set_vocabulary(config.vocab)
reverse_tokenizer=layers.experimental.preprocessing.StringLookup(vocabulary=tokenizer.get_vocabulary(),invert=True)

def random_invert_img(img,p=0.5):
    if tf.random.uniform([])<p:
        img=(255-img)
    return img 

def data_augment(img_data):
    img=img_data['image']
    #img=random_invert_img(img)
    #img=tf.image.random_saturation(img,0,3)
    img=tf.image.random_brightness(img,0.3)
    #img=tf.image.random_contrast(img,0.2,0.5)
    #img=tf.image.random_hue(img,0.2)
    img_data['image']=img 
    return img_data 




def encode_single_file(img_path,label,label_len):
    img=tf.io.read_file(img_path)
    img=tf.io.decode_jpeg(img)
    #img=tf.image.rgb_to_grayscale(img)
    img=tf.image.convert_image_dtype(img,tf.float32)
    img=tf.image.resize(img,[config.img_height,config.img_width])
    img=tf.transpose(img,[1,0,2])
    label=tokenizer(tf.strings.unicode_split(label,input_encoding='UTF-8')) # Be Sure to intialize a tokenizer
    #label=tf.expand_dims(label,axis=0)
    #label=keras.preprocessing.sequence.pad_sequences(label,maxlen=max_label_len)
    #label=tf.squeeze(label,axis=0)
    return {'image':img,'label':label,'label_length':label_len}


if __name__ == "__main__":
    encode_single_file(image_files[0])


