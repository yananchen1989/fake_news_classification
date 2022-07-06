import pandas as pd 
import json,random,os
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.metrics import *
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.model_selection import train_test_split
print(tf.__version__) # 2.9.1
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

gpus = tf.config.experimental.list_physical_devices('GPU')
print('======>',gpus,'<=======')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      # tf.config.experimental.set_virtual_device_configuration(gpu, \
      #      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)

from utils import * 


def main():
    # load data
    df_train = pd.read_csv("train.csv")
    df_train.drop_duplicates(['title','text'], inplace=True)

    df_test = pd.read_csv("test.csv")
    df_labels = pd.read_csv("labels.csv")
    df_test = pd.merge(df_test, df_labels, on='id', how='left')
    df_test.drop_duplicates(['title','text'], inplace=True)

    # EDA
    print(df_train['label'].value_counts())
    # balanced data

    for col in df_train.columns:
        print(col, df_train[col].isnull().sum())
    '''
    id 0
    title 558
    author 1957
    text 39
    label 0
    '''

    for col in df_test.columns:
        print(col, df_test[col].isnull().sum())

    '''
    id 0
    title 122
    author 503
    text 7
    '''
    # author distribution 
    # not to skew, a bit more even with long tail
    print(df_train['author'].value_counts())

    df_train = df_preprocess(df_train)
    df_test = df_preprocess(df_test)

    # special treatment for the author column
    # since authors' names are not exactly natural language
    # mapping the full names and sub names into token id and project them into embedding independently may bring some improvement
    author_id, author_seg_id = collect_author_dict(df_train)

    df_train['author_id'] = df_train['author'].map(lambda x: parse_authors(x)).map(lambda x: [author_id.get(i, 0) for i in x ])
    df_train['author_seg_id'] = df_train['author'].map(lambda x: parse_authors(x)).map(lambda x: get_author_seg_ids(x, author_seg_id))

    df_test['author_id'] = df_test['author'].map(lambda x: parse_authors(x)).map(lambda x: [author_id.get(i, 0) for i in x ])
    df_test['author_seg_id'] = df_test['author'].map(lambda x: parse_authors(x)).map(lambda x: get_author_seg_ids(x, author_seg_id))

    max_slot_author_id = df_train['author_id'].map(lambda x: len(x)).max()
    max_slot_author_seg_id = df_train['author_seg_id'].map(lambda x: len(x)).max()


    # build model
    model = get_model(author_id, author_seg_id)


    # prepare dataset
    # since here the dataset is small, only dict wrapping is used for simplicity instead of tf.dataset
    df_train_, df_val = train_test_split(df_train, test_size=0.15)

    train_dic_x, train_dic_y = get_train_dict(df_train_)
    val_dic_x, val_dic_y = get_train_dict(df_val)
    test_dic_x, test_dic_y = get_train_dict(df_test)

    # train and validation
    model.fit(train_dic_x, train_dic_y, batch_size=32, epochs=12, validation_data=(val_dic_x, val_dic_y),
                callbacks = [tf.keras.callbacks.EarlyStopping(monitor='acc', patience=3, mode='max',restore_best_weights=True )] )


    # inference
    preds = model.predict(test_dic_x, batch_size=128, verbose=1)
    preds_l = np.vectorize(lambda x: 1 if x >0.5 else 0)(preds)
    accuracy = accuracy_score(test_dic_y['label'], preds_l)

    # preds = model.predict(val_dic_x, batch_size=128, verbose=1)
    # preds_l = np.vectorize(lambda x: 1 if x >0.5 else 0)(preds)
    # accuracy = accuracy_score(val_dic_y['label'], preds_l)


    m = tf.keras.metrics.AUC(num_thresholds=preds.shape[0])
    m.update_state(test_dic_y['label'], preds)
    print(m.result().numpy())


# further improvement 
# I think fake news detection should focus more on the author side, for tracing their history records as an importance feature



if __name__ == "__main__":
    main()



