from tensorflow.keras import backend as K 
import numpy as np 
import config 
import tensorflow as tf
def ctc_to_text(y_pred,reverse_tokenizer):
    input_len=np.ones((y_pred.shape[0],))*y_pred.shape[1]
    results = K.ctc_decode(y_pred, input_length=input_len, greedy=True)[0][0][:, :config.max_len]
    output=[]

    for result in results:
        text=tf.strings.reduce_join(reverse_tokenizer(result)).numpy().decode('utf-8')
        try:
            last_idx=text.index('[UNK]')
        except:
            last_idx=len(text)
        text=text[:last_idx]
        output.append(text)
    return output,results 



