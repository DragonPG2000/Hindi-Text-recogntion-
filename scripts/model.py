import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.models import Model 
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import VGG16,InceptionV3 
from tf2cv.model_provider import get_model 
from ctc import CTCLayer
import config
def resnet_rnn(shape=(224,128,3),n_classes=96):
    inp=keras.layers.Input(shape,name='image')
    labels=keras.layers.Input((config.max_len),name='label')
    label_len=keras.layers.Input((1),name='label_length',dtype='int64')
    extractor=ResNet50(include_top=False,weights=None)
    extractor.trainable=True 
    x=extractor(inp)
    target_shape=(x.shape[1],x.shape[2]*x.shape[3])
    x=keras.layers.Reshape(target_shape)(x)
    x=keras.layers.LSTM(config.hidden_size,return_sequences=True,go_backwards=True)(x)
    x=keras.layers.LSTM(config.hidden_size,return_sequences=True,go_backwards=True)(x)
    x=keras.layers.Dense(n_classes,activation='softmax',name='dense2')(x)
    out=CTCLayer()(labels,x,label_len)
    model=Model(inputs=[inp,labels,label_len],outputs=[out])
    model.compile(optimizer=keras.optimizers.Adam(config.lr))
    return model

def downsample(inp,kernel_size=3,n_filters=256):
    x=keras.layers.Conv2D(filters=n_filters,kernel_size=kernel_size,padding='same',
                          activation='relu',kernel_initializer='he_normal')(inp)
    #x=keras.layers.BatchNormalization()(x)
    x=keras.layers.MaxPooling2D((2,2))(x)
    return x 

def baseline(shape=(224,128,3),n_classes=96):
    inp=keras.layers.Input(shape,name='image')
    labels=keras.layers.Input((config.max_len),name='label')
    label_len=keras.layers.Input((1),name='label_length',dtype='int64')
    x=downsample(inp,n_filters=32)
    x=downsample(x,n_filters=64)
    target_shape=(x.shape[1],x.shape[2]*x.shape[3])
    x=keras.layers.Reshape(target_shape)(x)
    x=keras.layers.Dense(64,activation='relu')(x)
    x=keras.layers.Dropout(0.4)(x)
    x=keras.layers.LSTM(128,return_sequences=True,go_backwards=True,dropout=0.25)(x)
    x=keras.layers.LSTM(64,return_sequences=True,go_backwards=True,dropout=0.25)(x)
    x=keras.layers.Dense(n_classes,activation='softmax',name='dense2')(x)
    out=CTCLayer()(labels,x,label_len)
    model=Model(inputs=[inp,labels,label_len],outputs=[out])
    model.compile(optimizer=keras.optimizers.Adam(config.lr))
    return model

def inception_extractor(shape=(224,128,3),n_classes=96,hidden_size=256):
    inp=keras.layers.Input(shape,name='image')
    labels=keras.layers.Input((config.max_len),name='label')
    label_len=keras.layers.Input((1),name='label_length',dtype='int64')
    extractor=InceptionV3(include_top=False,weights=None,input_shape=shape)
    extractor.trainable=True 
    x=extractor(inp)
    target_shape=(x.shape[1],x.shape[2]*x.shape[3])
    x=keras.layers.Reshape(target_shape)(x)
    x_prelim=keras.layers.LSTM(hidden_size,return_sequences=True,go_backwards=True)(x)
    x_lstm=keras.layers.LSTM(hidden_size,return_sequences=True,go_backwards=True)(x_prelim)
    x_gru=keras.layers.GRU(hidden_size,return_sequences=True,go_backwards=True)(x_prelim)
    x=keras.layers.Concatenate(axis=-1)([x_lstm,x_gru])
    x=keras.layers.Dense(n_classes,activation='softmax',name='dense2')(x)
    out=CTCLayer()(labels,x,label_len)
    model=Model(inputs=[inp,labels,label_len],outputs=[out])
    model.compile(optimizer=keras.optimizers.Adam(config.lr))
    return model

def tf2cv_extractor(shape=(224,128,3),n_classes=96,model_name='bn_vgg11',hidden_size=256):
    inp=keras.layers.Input(shape,name='image')
    labels=keras.layers.Input((config.max_len),name='label')
    label_len=keras.layers.Input((1),name='label_length',dtype='int64')
    extractor=get_model(model_name,pretrained=False)
    extractor.trainable=True 
    x=extractor.layers[-2](inp)
    target_shape=(x.shape[1],x.shape[2]*x.shape[3])
    x=keras.layers.Reshape(target_shape)(x)
    #x=keras.layers.Dense(64,activation='relu')(x) #Temp
    #x=keras.layers.Dropout(0.4)(x) #Temp 
    x_prelim=keras.layers.LSTM(hidden_size,return_sequences=True,go_backwards=True)(x)
    x_lstm=keras.layers.LSTM(hidden_size,return_sequences=True,go_backwards=True)(x_prelim)
    x_gru=keras.layers.GRU(hidden_size,return_sequences=True,go_backwards=True)(x_prelim)
    x=keras.layers.Concatenate(axis=-1)([x_lstm,x_gru])
    x=keras.layers.Dense(n_classes,activation='softmax',name='dense2')(x)
    out=CTCLayer()(labels,x,label_len)
    model=Model(inputs=[inp,labels,label_len],outputs=[out])
    model.compile(optimizer=keras.optimizers.Adam(config.lr))
    return model




if __name__=="__main__":
    model=inception_extractor(config.input_shape,96)
    print(model.summary())