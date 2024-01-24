
from __future__ import print_function

import argparse
import pandas as pd

from utils import *
import numpy as np
import torchvision
from torchvision.transforms import ToTensor
import os
import torch
from torch.utils.data import TensorDataset



"""Hyperparameters"""
parser = argparse.ArgumentParser(description='Configurations for Training')
args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.POSITIVE_CLASS = 8
args.BAG_COUNT = 50
args.VAL_BAG_COUNT = 1000
args.BAG_SIZE = 20
args.NUM_ITERATION= 10
args.POS_PER= 0.6
args.NEG_PER= 0.4
args.test_ratio= 0.9
args.TEST_BAG_COUNT = int(args.test_ratio*args.VAL_BAG_COUNT)


args.seed= 42
args.drop_out= 0.25
args.n_classes= 2
args.model_type = 'abmil_multihead'
args.max_epochs = 20
args.opt = "adam"
args.lr= 1e-3
args.reg=3e-2
args.temp= [1, 1, 1, 1]
args.early_stopping = True
args.results_dir= '.\\results'
args.exp_name= args.model_type + f'_{args.POS_PER}_{args.NEG_PER}_four'
args.plot= False
args.save_name = 'summary.csv'


"""Dataset Preparation"""
train_data = torchvision.datasets.MNIST('mnist', train= True, transform = ToTensor(), 
                                        download= True)
x_train, y_train = train_data.data, train_data.targets

test_data = torchvision.datasets.MNIST('mnist', train= False, transform = ToTensor(), 
                                        download= True)
x_val, y_val = test_data.data, test_data.targets


"""Training Loop"""
all_test_auc = []
all_val_auc = []
all_test_acc = []
all_val_acc = []
all_test_loss= []
all_val_loss= []

all_test= []
all_vals= []
folds = np.arange(args.NUM_ITERATION)    
for it in range(args.NUM_ITERATION):
    
    """results directory"""
    if not os.path.isdir(os.path.join(args.results_dir, args.exp_name)):
        os.mkdir(os.path.join(args.results_dir, args.exp_name))
    
    print("\n\n")
    seed_torch(args.seed+it)
    
    """Train Data"""
    train_data, train_labels = create_soft_bags(x_train, y_train, args.POSITIVE_CLASS, args.BAG_COUNT
                                           , args.BAG_SIZE, pos_perc= args.POS_PER, neg_perc= args.NEG_PER, 
                                           seed= it)    
    
    val_data, val_labels = create_soft_bags(x_val, y_val, args.POSITIVE_CLASS, args.VAL_BAG_COUNT
                                       , args.BAG_SIZE, pos_perc= args.POS_PER, neg_perc= args.NEG_PER, 
                                       seed= it)
    
    """Test & Val Data"""
    test_data, test_labels = val_data[:args.TEST_BAG_COUNT, :, :, :], val_labels[:args.TEST_BAG_COUNT]
    val_data, val_labels = val_data[args.TEST_BAG_COUNT:, :, :, :], val_labels[args.TEST_BAG_COUNT:]
    
    """Datasets"""
    dataset_train = TensorDataset(torch.tensor(train_data), torch.tensor(train_labels))   
    dataset_val = TensorDataset(torch.tensor(val_data), torch.tensor(val_labels))    
    dataset_test = TensorDataset(torch.tensor(test_data), torch.tensor(test_labels))    

    datasets = (dataset_train,  dataset_val,  dataset_test)
    
    if not args.plot:
        test_auc, val_auc, test_acc, val_acc, test_loss, val_loss, test_atts, val_atts = train(datasets, it, args)
    
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        all_test_loss.append(test_loss)
        all_val_loss.append(val_loss)
        
    all_test.append(dataset_test)
    all_vals.append(dataset_val)
    
    
if not args.plot:
    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc,
    'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc, 
    'test_loss' : all_test_loss, 'val_loss' : all_val_loss})
    
    final_df.to_csv(os.path.join(args.results_dir, args.exp_name, args.save_name))

    "Compute the average and std of the metrics"
    test_auc_ave= np.mean(all_test_auc)
    test_acc_ave= np.mean(all_test_acc)
    test_loss_ave= np.mean(all_test_loss)
    
    test_auc_std= np.std(all_test_auc)
    test_acc_std= np.std(all_test_acc)
    test_loss_std= np.std(all_test_loss)
    
    val_auc_ave= np.mean(all_val_auc)
    val_acc_ave= np.mean(all_val_acc)
    val_loss_ave= np.mean(all_val_loss)
    
    val_auc_std= np.std(all_val_auc)
    val_acc_std= np.std(all_val_acc)
    val_loss_std= np.std(all_val_loss)
        
    print('\nVal:\nloss ± std: {0:.3f} ± {1:.3f}'.format(val_loss_ave, val_loss_std))
    
    print('\nTest:\nauc ± std: {0:.3f} ± {1:.3f}, acc ± std: {2:.3f} ± {3:.3f}'.format(
        test_auc_ave, test_auc_std, test_acc_ave, test_acc_std))
    
    print('\nMisc:\nauc ± std (val): {0:.3f} ± {1:.3f}, acc ± std (val): {2:.3f} ± {3:.3f},'
          'loss ± std (test): {4:.3f} ± {5:.3f} \n'.format(val_auc_ave, val_auc_std, val_acc_ave,
          val_acc_std, test_loss_ave, test_loss_std))
            
    print(f'Hyperparameters: {args} \n')

    num= np.argmax(all_test_auc)
    
else:
    all_test_auc= list(pd.read_csv(os.path.join(args.results_dir, args.exp_name, args.save_name))['test_auc'])    
    
    num= np.argmax(all_test_auc)




"""Plot Some results"""
plot(
    all_test[num],
    "positive",
    plot_size= 1,
    args= args,
    fold_num= num
)




plot(
    all_test[num],
    "negative",
    plot_size= 10,
    args= args,
    fold_num= num
)

    