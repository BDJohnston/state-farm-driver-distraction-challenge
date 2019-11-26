#!/usr/bin/env python3

import numpy as np
import operator
import os
import sys
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd
import itertools

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, \
                                       ZeroPadding2D

# from keras.layers.normalization import BatchNormalization
# from keras.optimizers import Adam
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import model_from_json
# from sklearn.metrics import log_loss
from numpy.random import permutation
from keras import applications

import torch
import faiss

from multiprocessing import Pool, Process, Manager
# from multiprocessing import shared_memory

from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session

import tensorflow as tf
import gc

from numba import cuda
import signal
from contextlib import closing

#sys.path.append('/content/state_farm')

np.random.seed(2016)
use_cache = 1
# color type: 1 - grey, 3 - rgb
color_type_global = 3

# color_type = 1 - gray
# color_type = 3 - RGB

def shift_augmentation(X, h_range, w_range):
    #progbar = Progbar(X.shape[0])  # progress bar for augmentation status tracking

    X_shift = np.copy(X)
    size = X.shape[2:]
    for i in range(len(X)):
        h_random = np.random.rand() * h_range * 2. - h_range
        w_random = np.random.rand() * w_range * 2. - w_range
        h_shift = int(h_random * size[0])
        w_shift = int(w_random * size[1])
        for j in range(X.shape[1]):
            X_shift[i, j] = ndimage.shift(X[i, j], (h_shift, w_shift), order=0)
        #progbar.add(1)
    return X_shift

def get_im(path, img_rows, img_cols, color_type=1):
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows))

    return resized


def get_driver_data():
    dr = dict()
    path = os.path.join('state_farm', 'input', 'driver_imgs_list.csv')
    print('Read drivers data')
    f = open(path, 'r')
    line = f.readline()
    while (1):
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        dr[arr[2]] = arr[0]
    f.close()
    return dr


def load_train(img_rows, img_cols, color_type=1):
    X_train = []
    y_train = []
    driver_id = []

    driver_data = get_driver_data()

    print('Read train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join('state_farm', 'input', 'imgs', 'train',
                            'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for i, fl in enumerate(files):
            flbase = os.path.basename(fl)
            img = get_im(fl, img_rows, img_cols, color_type)
            
            if i == 0:                
                X_train.append(img)
                y_train.append(j)
                driver_id.append(driver_data[flbase])

    unique_drivers = sorted(list(set(driver_id)))
    print('Unique drivers: {}'.format(len(unique_drivers)))
    print(unique_drivers)
    return X_train, y_train, driver_id, unique_drivers


def load_test(img_rows, img_cols, color_type=1):
    print('Read test images')
    path = os.path.join('state_farm', 'input', 'imgs', 'test', '*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    thr = math.floor(len(files)/10)
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im(fl, img_rows, img_cols, color_type)
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total % thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    return X_test, X_test_id

def load_small_test(img_rows, img_cols, read_range=[0, 1000], color_type=1):
    print('Read test images')
    path = os.path.join('state_farm', 'input', 'imgs', 'test', '*.jpg')
    files = glob.glob(path)
    # Sanity check
    assert(read_range[0] < len(files))
    assert(read_range[0] < read_range[1])
    if read_range[1] > len(files):
        read_range[1] = len(files)
    files = files[read_range[0]:read_range[1]]
    X_test = []
    X_test_id = []
    total = 0
    #for fl in tqdm(files):
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im(fl, img_rows, img_cols, color_type)
        X_test.append(img)
        X_test_id.append(flbase)
    return np.array(X_test, dtype=np.float32), np.array(X_test_id)

def cache_data(data, path):
    if not os.path.isdir(os.path.join('state_farm', 'cache')):
        os.mkdir(os.path.join('state_farm', 'cache'))
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')

def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        print('Restore data from pickle........')
        file = open(path, 'rb')
        data = pickle.load(file)
    return data

def save_model(model, index, cross=''):
    json_string = model.to_json()
    if not os.path.isdir(os.path.join('state_farm', 'cache')):
        os.mkdir(os.path.join('state_farm', 'cache'))
    json_name = 'architecture' + str(index) + cross + '.json'
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    open(os.path.join('state_farm','cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('state_farm', 'cache', weight_name), overwrite=True)

def read_model(index, cross=''):
    json_name = 'architecture' + str(index) + cross + '.json'
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    model = model_from_json(open(os.path.join('state_farm', 'cache', json_name)).read())
    model.load_weights(os.path.join('state_farm', 'cache', weight_name))
    return model


def split_validation_set(train, target, test_size):
    random_state = 51
    X_train, X_test, y_train, y_test = \
        train_test_split(train, target,
                         test_size=test_size,
                         random_state=random_state)
    return X_train, X_test, y_train, y_test

def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3',
                                                 'c4', 'c5', 'c6', 'c7',
                                                 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir(os.path.join('state_farm', 'subm')):
        os.mkdir(os.path.join('state_farm', 'subm'))
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join( 'state_farm', 'subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)

def read_and_normalize_and_shuffle_train_data(img_rows, img_cols,
                                              color_type=1):

    cache_path = os.path.join('state_farm', 'cache', 'train_r_' + str(img_rows) +
                              '_c_' + str(img_cols) + '_t_' +
                              str(color_type) + '.dat')

    if not os.path.isfile(cache_path) or use_cache == 0:
        train_data, train_target, driver_id, unique_drivers = \
            load_train(img_rows, img_cols, color_type)
        cache_data((train_data, train_target, driver_id, unique_drivers),
                   cache_path)
    else:
        print('Restore train from cache!')
        (train_data, train_target, driver_id, unique_drivers) = \
            restore_data(cache_path)

    train_data = np.array(train_data, dtype=np.float32)
    train_target = np.array(train_target, dtype=np.uint8)

    if color_type == 1:
        train_data = train_data.reshape(train_data.shape[0], img_cols, img_rows, 
                                        color_type)
    else:
        train_data = train_data.transpose((0, 2, 1, 3))

    train_target = np_utils.to_categorical(train_target, 10)
    train_data = train_data.astype('float32')
    mean_pixel = [103.939, 116.779, 123.68] #bgr
    for c in range(3):
        train_data[:, :, :, c] = train_data[:, :, :, c] - mean_pixel[c]
    perm = permutation(len(train_target))
    train_data = train_data[perm]
    train_target = train_target[perm]
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, driver_id, unique_drivers


def read_and_normalize_test_data(img_rows=224, img_cols=224, color_type=1):
    cache_path = os.path.join('state_farm', 'cache', 'test_r_' + str(img_rows) +
                              '_c_' + str(img_cols) + '_t_' +
                              str(color_type) + '.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        test_data, test_id = load_test(img_rows, img_cols, color_type)
        cache_data((test_data, test_id), cache_path)
    else:
        print('Restore test from cache!')
        (test_data, test_id) = restore_data(cache_path)

    test_data = np.array(test_data, dtype=np.uint8)

    if color_type == 1:
        test_data = test_data.reshape(test_data.shape[0], img_cols, img_rows, 
                                      color_type )
    else:
        test_data = test_data.transpose((0, 2, 1, 3))

    test_data = test_data.astype('float32')
    mean_pixel = [103.939, 116.779, 123.68] #bgr
    for c in range(3):
        test_data[:, :, :, c] = test_data[:, :, :, c] - mean_pixel[c]
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    return test_data, test_id


def dict_to_list(d):
    ret = []
    for i in d.items():
        ret.append(i[1])
    return ret

def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()

def merge_several_folds_geom(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a *= np.array(data[i])
    a = np.power(a, 1/nfolds)
    return a.tolist()

def merge_several_folds_max(data, nfolds):
    a = np.array(data[0])
    #for i in range(1, nfolds):    
    a = np.amax(np.array(data), axis=1)
    return a.tolist()
  
def copy_selected_drivers(train_data, train_target, driver_id, driver_list):
    data = []
    target = []
    index = []
    for i in range(len(driver_id)):
        if driver_id[i] in driver_list:
            data.append(train_data[i])
            target.append(train_target[i])
            index.append(i)
    data = np.array(data, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    index = np.array(index, dtype=np.uint32)
    return data, target, index

def vgg_std16_model(img_rows, img_cols, color_type=1):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(img_cols, img_rows, 
                                                 color_type)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels.h5')

    # Code above loads pre-trained data and
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(10, activation='softmax'))
    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def vgg_std16_model_new(img_rows, img_cols, color_type=1):
    vgg_model = applications.VGG16(weights='imagenet',
                                   include_top=True)

    # Creating dictionary that maps layer names to the layers
    layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])
    
    # Getting output tensor of the last VGG layer that we want to include
    x = layer_dict['fc1'].output
    x = Dropout(0.5)(x)
    x = layer_dict['fc2'](x)
    x = Dropout(0.5)(x)
    # Stacking a new fully connected output layer on top of it    
    x = Dense(10, activation='softmax', name='predictions')(x)

    # Creating new model. Please note that this is NOT a Sequential() model.
    from keras.models import Model
    custom_model = Model(input=vgg_model.input, output=x)

    # Make sure that the pre-trained bottom layers are not trainable
    #for layer in custom_model.layers[:7]:
    #    layer.trainable = False

    # Learning rate is changed to 0.001
    #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=1e-6)
    custom_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return custom_model 

def vgg_std16_model_KNN_WA(model):
  
    x_KNN = model.get_layer(name='flatten').output
    
    from keras.models import Model
    custom_model_KNN = Model(input=model.input, output=x_KNN)
    
    return custom_model_KNN 

def save_fold_to_csv(index, modelStr, out):
    
    f_name = str(index) +  modelStr + '.csv'
#     f=open(os.path.join('state_farm', 'cache', f_name),'ab')
#     np.savetxt(f, out)
    pd.DataFrame(out).to_csv(os.path.join('state_farm', 'cache', f_name), header=None, index=None, mode='a')
#     f.close()
    
# Reset Keras Session
def remove_keras_sess(model):
    sess = get_session()
    
    try:
        del model # this is from global space - change this as you need
    except:
        pass
    
    clear_session()
    sess.close()
    
    print(gc.collect()) # if it's done something you should see a number being outputted

def reset_keras():
    
    sess = get_session()

    # use the same config as you used to create the session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))
    
def run_cross_validation(nfolds=10, nb_epoch=10, split=0.2, modelStr=''):

    # Now it loads color image
    # input image dimensions
    img_rows, img_cols = 224, 224
    batch_size = 64
    random_state = 20

    train_data, train_target, driver_id, unique_drivers = \
        read_and_normalize_and_shuffle_train_data(img_rows, img_cols,
                                                  color_type_global)

    num_fold = 0

    for fold in range(nfolds):
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
    
        model = vgg_std16_model_new(img_rows, img_cols, color_type=1)
        model.summary()
        model.fit(
                  train_data, train_target,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  verbose=1,
                  validation_split=split, 
                  shuffle=True)

        save_model(model, num_fold, modelStr)
        reset_keras(model)

def get_knn_wa_predictions(test_final_pool_layer_outputs, yfull_test, k=10):        
        
    torch.device('cuda:1')

    d=test_final_pool_layer_outputs.shape[1]

    res = faiss.StandardGpuResources()

    # build a flat (CPU) index
    index_flat = faiss.IndexFlatL2(d)
    # make it into a gpu index
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 1, index_flat)

    gpu_index_flat.add(test_final_pool_layer_outputs)         # add vectors to the index
    print(gpu_index_flat.ntotal)

    k = 11                          # we want to see 10 nearest neighbors
    D, I = gpu_index_flat.search(test_final_pool_layer_outputs, k)  # actual search
    D = D.transpose()
    D = D + 1e-6

    D = 1/D

    yfull_test_knn_wa = np.zeros((yfull_test.shape[0], yfull_test.shape[1])).astype('float32')
    for i in range(I.shape[0]):
        yfull_test_knn_wa[i] = np.matmul(yfull_test[I[i,1:].astype(int)].T, (D[1:,i] - np.amin(D[1:,i]))/np.sum(D[1:,i] - -np.amin(D[1:,i]))).T#/k
    gpu_index_flat.reset()
    del gpu_index_flat
    del res
    torch.cuda.empty_cache()

    print('CPU and GPU memory ok')

    return yfull_test_knn_wa

def img_process():
    
    reset_keras()
    model = read_model(index, modelStr)
    test_prediction = model.predict(X_test, batch_size=batch_size, verbose=1)
    model_KNN_WA = vgg_std16_model_KNN_WA(model)
    test_final_pool_layer_output = model_KNN_WA.predict(X_test, batch_size=batch_size, verbose=1)
    remove_keras_sess(model)
    remove_keras_sess(model_KNN_WA)
    cuda.select_device(0)
    cuda.close()
    
    return test_prediction, test_final_pool_layer_output

def process_images(index, modelStr, X_test, batch_size):
    with closing(Pool(1)) as p:
        with tf.device('/gpu:0'):
            reset_keras()
            model = read_model(index, modelStr)
            test_prediction = model.predict(X_test, batch_size=batch_size, verbose=1)
            model_KNN_WA = vgg_std16_model_KNN_WA(model)
            test_final_pool_layer_output = model_KNN_WA.predict(X_test, batch_size=batch_size, verbose=1)
            remove_keras_sess(model)
            remove_keras_sess(model_KNN_WA)
            p.terminate()
            p.close()
            p.join()
            return test_prediction, test_final_pool_layer_output

def close_pool():
    global pool
    pool.close()
    pool.terminate()
    pool.join()

def term(*args,**kwargs):
    sys.stderr.write('\nStopping...')
    stophttp = threading.Thread(target=httpd.shutdown)
    stophttp.start()
    stoppool=threading.Thread(target=close_pool)
    stoppool.daemon=True
    stoppool.start()

def test_model_KNN_use_batches_and_submit(start=1, end=1, nb_epoch=3, modelStr=''):
    img_rows, img_cols = 224, 224
    batch_size = 128
    random_state = 51
    #nb_batches * nb_batch_size = 80000
    nb_batches = 10#200
    nb_batch_size = 8000#400
    nb_epoch = nb_epoch
    nfolds = end
    print('Start testing............')

    info_string = 'loss_' + modelStr \
                  + '_r_' + str(img_rows) \
                  + '_c_' + str(img_cols) \
                  + '_folds_' + str(nfolds) \
                  + '_ep_' + str(nb_epoch)

    test_id = []
    test_res = []
    
    for index in range(1, nfolds+1):
        
        yfull_test = np.empty([nb_batch_size, 10])
        test_final_pool_layer_outputs = []
        
        for i in range(nb_batches):        
            
            print("Converting %d th test set"%i)
            # Load train data
            X_test, X_test_id = load_small_test(224, 224, read_range=[nb_batch_size*i, nb_batch_size*(i+1)], color_type=3)
            # Modify images
            print("Preprocessing images")
            X_test[:, :, :, 0] -= 103.939
            X_test[:, :, :, 1] -= 116.779
            X_test[:, :, :, 2] -= 123.68
            X_test = X_test.transpose((0, 2, 1, 3))
            
            if index == 0:
                test_id.append(X_test_id)

            print(i, index)
            # 1,2,3,4,5
            # Store test predictions
            test_prediction = 0
            test_prediction, test_final_pool_layer_output = process_images(index, modelStr, X_test, batch_size);
            test_final_pool_layer_outputs.append(test_final_pool_layer_output)
            if i == 0:
                yfull_test = test_prediction
            else:
                yfull_test = np.vstack((yfull_test, test_prediction))
        
        print('CPU memory ok')

        test_final_pool_layer_outputs = np.concatenate(test_final_pool_layer_outputs)
        test_final_pool_layer_outputs = test_final_pool_layer_outputs.astype('float32')
    
        pool_out_file = os.path.join('state_farm', 'cache', str(index) + 'pool_out' + info_string)
        np.save(pool_out_file, test_final_pool_layer_outputs, allow_pickle=False, fix_imports=False)
        pred_out_file = os.path.join('state_farm', 'cache', str(index) + 'pred_out'+ info_string)
        np.save(pred_out_file, yfull_test, allow_pickle=False, fix_imports=False)
        
        yfull_test = get_knn_wa_predictions(test_final_pool_layer_outputs, yfull_test, k=10)        
        
        test_res.append(yfull_test)
    
    tt = np.asarray(merge_several_folds_mean(test_res, nfolds))
    
    tst = np.asarray(test_id)
    
    create_submission(tt[8000:,:], tst[7726:], info_string)

if __name__ == "__main__":
    
    # nfolds, nb_epoch, split
    m_name = '_dropout_vgg_16_2x20_8_15_0_15'
    
    run_cross_validation(8, 15, 0.15, m_name)

    save_model_pred_and_flat_final_pool_out(1, 8, 15, m_name)

    test_model_KNN_use_batches_and_submit(1, 8, 15, m_name)
