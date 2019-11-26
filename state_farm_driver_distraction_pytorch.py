from __future__ import print_function, division
import os
import sys
import glob
import math
import datetime
import itertools
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR
import time
import copy
from torchsummary import summary
from skorch import NeuralNet, NeuralNetClassifier
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler, Checkpoint, ProgressBar, Freezer
from skorch.dataset import CVSplit
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import cross_val_predict
import faiss
from PIL import Image 

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

torch.manual_seed(2019);

# dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

torch.set_default_tensor_type('torch.cuda.FloatTensor')

root_dir = '/state_farm/'

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
    
def get_knn_wa_predictions(test_final_pool_layer_outputs, yfull_test, k=11):        
        
    # torch.device('cuda:1')

    d=test_final_pool_layer_outputs.shape[1]

    res = faiss.StandardGpuResources()

    # build a flat (CPU) index
    index_flat = faiss.IndexFlatL2(d)
    # make it into a gpu index
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 1, index_flat)

    gpu_index_flat.add(test_final_pool_layer_outputs) # add vectors to the index
    # print(gpu_index_flat.ntotal)

    k = k # for 10 nearest neighbors, set k = 11
    D, I = gpu_index_flat.search(test_final_pool_layer_outputs, k) # actual search

    D = D.transpose()
    D = D + 1e-6

    D = 1/D
    
    yfull_test_knn_wa = np.zeros((yfull_test.shape[0], yfull_test.shape[1])).astype('float32')
    for i in range(I.shape[0]):
        yfull_test_knn_wa[i] = np.matmul(yfull_test[I[i,1:].astype(int)].T, (D[1:,i] - np.amin(D[1:,i]))/np.sum(D[1:,i] - -np.amin(D[1:,i]))).T
    gpu_index_flat.reset()
    del gpu_index_flat
    del res
    torch.cuda.empty_cache()

    return yfull_test_knn_wa
    
class DriverDistractionTrainDataset(Dataset):
    """Driver Distraction dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Opti/onal transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file)
        
        self.label_enc = LabelEncoder()
        
        label_enc_classes = self.label_enc.fit_transform(np.array(self.df['classname'].tolist()))

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_class = self.df.iloc[idx, 1]
        
        img_name = os.path.join(self.root_dir,
                                img_class,
                                self.df.iloc[idx, 2])

        image = Image.open(img_name)  

        img_class = self.label_enc.transform(np.array(img_class).reshape(-1,1))

        if self.transform:
            image = self.transform(image)
            # print(np.array(transforms.ToPILImage()(image[0]).convert("RGB")).shape)
        return image, img_class

def view_train_dataset():
    img_rows, img_cols = 224, 224

    train_dataset = DriverDistractionTrainDataset(csv_file=root_dir+'driver_imgs_list.csv',
                        root_dir=root_dir+'train',
                        transform=transforms.Compose([
                            transforms.Resize((img_rows, img_cols)),    
                            transforms.ToTensor(),                      
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                std=[0.229, 0.224, 0.225])
                        ]))
    
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                   shuffle=True, num_workers=4)

    dataiter = iter(dataloader)
    
    fig = plt.figure()

    for i in range(len(train_dataset)):
        X, y = dataiter.next()
        
        print(i, X.shape, y.shape)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        image = transforms.ToPILImage()(X[0]).convert("RGB")
        plt.imshow(image)
        if i == 3:
            plt.show()
            break

class CategoricalCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(CategoricalCrossEntropy, self).__init__()
        self.loss = nn.NLLLoss()
        
    def forward(self, output, target): 
        # Target will be one-hot: (batch_size, n_classes)
        # But PyTorch only uses class labels.
        # PyTorch doesn't automatically broadcast loss into higher
        # dimensions, so we need to flatten it out.

        # There is only one input for this loss function.

        # Flatten it out into: (lots of entries)
        target = target.view(-1)

        output = output.view(-1, output.size(output.dim() - 1))

        return self.loss(output, target)   

class SaveFeatures():
    features=[]
    def __init__(self, m): 
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features.append(output)
    def clear_features(self): self.features.clear()
    def remove(self): self.hook.remove()

class PretrainedModel(nn.Module):
    def __init__(self, training):
        super(PretrainedModel, self).__init__()
        self.training = training
        model_ft = models.vgg16(pretrained=True)
        self.sfs = SaveFeatures(model_ft.avgpool)

        model_ft.classifier[6] = nn.Linear(4096, 10)
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        summary(model_ft, (3, 224, 224))

        self.model_ft = model_ft   

    def forward(self, x):
        x = self.model_ft(x)  
        if self.training:
            x = self.log_softmax(x)
        else:
            x = self.softmax(x)
        return x

def build_net(num_epochs, batch_size, training, img_rows, img_cols, split, modelStr):

    net = NeuralNetClassifier(
        module=PretrainedModel,
        module__training=training, 
        criterion=CategoricalCrossEntropy,
        lr=1e-3,
        batch_size=batch_size,
        max_epochs=num_epochs,
        optimizer=optim.SGD,
        optimizer__momentum=0.9,
        optimizer__nesterov=True,
        iterator_train__shuffle=True,
        # iterator_train__num_workers=4,
        iterator_valid__shuffle=True,
        # iterator_valid__num_workers=4,
        train_split=CVSplit(cv=split, stratified=True, random_state=0),# None,
        callbacks=[ ('lrscheduler', LRScheduler(policy='StepLR', step_size=10, gamma=0.1)),
                    ('checkpoint', Checkpoint(f_params='best_model.pt', fn_prefix=modelStr, monitor='valid_loss_best')),
                    ('freezer', Freezer(lambda x: not x.startswith('model_ft.classifier'))),
                    ('progressbar', ProgressBar())
                  ],
        device='cuda:0' # comment to train on cpu
    )

    return net        
    
def train_ensemble(nmodels=10, nb_epoch=10, split=0.1, modelStr=''):

    # Now it loads color image
    # input image dimensions
    img_rows, img_cols = 224, 224
    batch_size = 32
    random_state = 20
    training = True

    df = pd.read_csv(root_dir+'driver_imgs_list.csv')
    label_enc = LabelEncoder()
    y = label_enc.fit_transform(np.array(df['classname'].tolist()))

    train_dataset = DriverDistractionTrainDataset(csv_file=root_dir+'driver_imgs_list.csv',
                                           root_dir=root_dir+'train',
                                           transform=transforms.Compose([
                                               transforms.Resize((img_rows, img_cols)),    
                                               transforms.ToTensor(),                      
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                    std=[0.229, 0.224, 0.225])
                                           ]))

    for model_num in range(nmodels):
        print('Start ensemble number {} from {}'.format(model_num, nmodels))

        net = build_net(nb_epoch, batch_size, training, img_rows, img_cols, split, root_dir + modelStr + '_model_num_' + str(model_num))

        net.fit(train_dataset, y=y);

        del net

def test_model_KNN_use_batches_and_submit(start=1, nb_models=1, nb_epoch=3, modelStr=''):
    img_rows, img_cols = 224, 224
    batch_size = 128
    random_state = 51
    #nb_batches * nb_batch_size = 80000
    nb_batches = 10#200
    nb_batch_size = 8000#400
    training = False
    print('Start testing............')

    info_string = 'loss_' + modelStr \
                  + '_r_' + str(img_rows) \
                  + '_c_' + str(img_cols) \
                  + '_folds_' + str(nb_models) \
                  + '_ep_' + str(nb_epoch)
                  
    test_id = []
    test_res = []

    for index in range(1, nb_models+1):
        
        yfull_test = np.empty([nb_batch_size, 10])
        test_final_pool_layer_outputs = []
        
        for i in range(nb_batches):        
          
            print(i, index)
          
            test_dataset = torchvision.datasets.ImageFolder(
                root_dir+'test', 
                transform==transforms.Compose([
                    transforms.Resize((img_rows, img_cols)),
                    transforms.ToTensor(),                          
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225])
                ]))
            
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                   shuffle=False)#, num_workers=4)
            
            # Store test predictions

            criterion = CategoricalCrossEntropy
            checkpoint = Checkpoint(f_params='best_model.pt', fn_prefix=root_dir + modelStr + '_model_num_' + str(model_num))
            net = NeuralNetClassifier(
                module=PretrainedModel,
                module__training=training,
                criterion=criterion,
                lr=1e-3,
                batch_size=batch_size,
                max_epochs=1,
                # iterator_train__num_workers=4,
                # iterator_valid__num_workers=4,
                device='cuda:0' # comment to train on cpu
            )
         
            net.initialize()
            net.load_params(checkpoint=checkpoint)
          
            with torch.no_grad():
                for i in range(nb_batches):
                    net.module_.sfs.clear_features()
                    y_pred = y_pred.clear()
                    for i, (X, _) in enumerate(test_loader, 0):
                        y_pred.append(net.predict(X))
                        sample_fnames, _ = test_loader.dataset.samples[i]
                        test_id.append(sample_fnames)
                    avgpool_outs.append(net.module_.sfs.features)
                    if i == 0:
                        yfull_test = y_pred
                    else:
                        yfull_test = np.vstack((yfull_test, y_pred))
      
        avgpool_outs = np.concatenate(net.module_.sfs)
        avgpool_outs = avgpool_outs.astype('float32')
        print(avgpool_outs.shape)
    
        pool_out_file = os.path.join(root_dir + 'cache', str(index) + 'pool_out' + info_string)
        np.save(pool_out_file, avgpool_outs, allow_pickle=False, fix_imports=False)
        pred_out_file = os.path.join(root_dir + 'cache', str(index) + 'pred_out'+ info_string)
        np.save(pred_out_file, yfull_test, allow_pickle=False, fix_imports=False)
        
        yfull_test = get_knn_wa_predictions(avgpool_outs, yfull_test)        
        
        test_res.append(yfull_test)
    
    tt = np.asarray(merge_several_folds_mean(test_res, nb_models))
    
    tst = np.concatenate(test_id).ravel()
        
    create_submission(tt[8000:,:], tst[7726:], info_string)

if __name__ == __main__:

    m_name = '_pytorch_vgg_16_dropout_2x20_8_15_0_15'

    view_train_dataset()

    train_ensemble(8, 15, 0.15, m_name)

    test_model_KNN_use_batches_and_submit(1, 8, 15, m_name)
