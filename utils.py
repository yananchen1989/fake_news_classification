import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text as text 
import numpy as np
import string
from tensorflow.keras.optimizers import Adam


def df_preprocess(df):
    # fill in NA values
    for col in ['title','text']:
        df[col] = df[col].fillna('').map(lambda x: clean_title(x))
    df['author'] = df['author'].fillna('no_author')

    return df



def clean_title(title):
    return title.lower().replace("\n"," ").replace('\t', ' ').strip() #str(title.strip().lower().encode('utf-8')).replace("\n"," ").replace('\t', ' ')

def parse_authors(author_text):
    # here i suppose we need some rules to reduce the chaos in the format issue
    # more rules can be introduced
    author_text = author_text.replace(' and ', ', ')
    for s in string.punctuation:
        if s not in ['.',',']:
            author_text = author_text.replace(s, '')
    return [i.strip() for i in author_text.split(',')]

def collect_author_dict(df_train):
    authors_train = set()
    authors_seg_train = set()
    for i in df_train['author'].map(lambda x: parse_authors(x)).tolist():
        authors_train.update(i)
        for j in i:
            authors_seg_train.update(j.split())

    author_id = {au:ix+1 for ix, au in enumerate(list(authors_train))} # 3297
    author_seg_id = {au:ix+1 for ix, au in enumerate(list(authors_seg_train))} # 4635
    return author_id, author_seg_id



def get_author_seg_ids(l, author_seg_id):
    res = []
    for i in l:
        for t in i.split():
            res.append(author_seg_id.get(t, 0))
    return res 



def encode_textcnn(x):
    # tri-gram
    title_conv3 = tf.keras.layers.Conv1D(128, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x) # (?, 28, 128)
    # bi-gram
    title_conv2 = tf.keras.layers.Conv1D(128, kernel_size = 2, padding = "valid", kernel_initializer = "glorot_uniform")(x) # (?, 29, 128)
    # uni-gram
    title_conv1 = tf.keras.layers.Conv1D(128, kernel_size = 1, padding = "valid", kernel_initializer = "glorot_uniform")(x) # (?, 30, 128)
    
    avg_pool_3 = tf.keras.layers.GlobalAveragePooling1D()(title_conv3)# (?, 128)
    max_pool_3 = tf.keras.layers.GlobalMaxPooling1D()(title_conv3) # (?, 128)
    avg_pool_2 = tf.keras.layers.GlobalAveragePooling1D()(title_conv2)# (?, 128)
    max_pool_2 = tf.keras.layers.GlobalMaxPooling1D()(title_conv2) # (?, 128)
    avg_pool_1 = tf.keras.layers.GlobalAveragePooling1D()(title_conv1)# (?, 128)
    max_pool_1 = tf.keras.layers.GlobalMaxPooling1D()(title_conv1) # (?, 128)   
    encode = tf.keras.layers.concatenate([ avg_pool_3, max_pool_3, avg_pool_2, max_pool_2, avg_pool_1, max_pool_1]) 
    return encode

def get_model(author_id, author_seg_id):

    preprocessor = hub.KerasLayer(
        "https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2")
    encoder = hub.KerasLayer(
        "https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base/1")

    title_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='title') # shape=(None,) dtype=string
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text') # shape=(None,) dtype=string
    author_text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='author')# shape=(None,) dtype=string

    author_input = tf.keras.layers.Input(shape=(max_slot_author_id, ), dtype=tf.int32, name='author_id')
    author_seg_input = tf.keras.layers.Input(shape=(max_slot_author_seg_id, ), dtype=tf.int32, name='author_seg_id')

    title_embed = encoder(preprocessor(title_input))["default"]
    text_embed = encoder(preprocessor(text_input))["default"]
    author_embed = encoder(preprocessor(author_text_input))["default"]

    embedding_author_id =  tf.keras.layers.Embedding(max(author_id.values()), 128,  trainable=True, mask_zero=True)
    embedding_author_seg_id =  tf.keras.layers.Embedding(max(author_seg_id.values()), 128,  trainable=True, mask_zero=True)

    embedding_author = embedding_author_id(author_input) # (None, 6, 128)
    embedding_author_seg = embedding_author_seg_id(author_seg_input) # (None, 22, 128)

    author_encode = encode_textcnn(embedding_author)
    author_seg_encode = encode_textcnn(embedding_author_seg)

    embed = tf.keras.layers.concatenate([title_embed, text_embed, author_encode, author_seg_encode])
    embed = tf.keras.layers.Dense(1024, activation='relu')(embed)

    out =  tf.keras.layers.Dense(1, activation='sigmoid', name="label")(embed)
    model = tf.keras.Model(inputs=[title_input, text_input, author_text_input, author_input, author_seg_input],\
                           outputs=out)
    model.compile(Adam(learning_rate=2e-5), "binary_crossentropy", metrics=["binary_accuracy"])


    return model 




def get_train_dict(df):
    dic_x = {}
    for col in ['title','text', 'author']:
        dic_x[col] = df[col].values.reshape(-1,1)

    dic_x['author_id'] = tf.keras.preprocessing.sequence.pad_sequences(df['author_id'].tolist(), \
                            padding='post', value=0, maxlen=max_slot_author_id)
    dic_x['author_seg_id'] = tf.keras.preprocessing.sequence.pad_sequences(df['author_seg_id'].tolist(), \
                            padding='post', value=0, maxlen=max_slot_author_seg_id)

    dic_y = {'label': df['label'].values}
    return dic_x, dic_y


