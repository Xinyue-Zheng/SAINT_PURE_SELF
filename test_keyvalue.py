import torch
from torch import nn
from models import SAINT, SAINT_vision,SAINT_con, SAINT_dict

import pandas as pd

from data_openml import DataSetCatCon, data_prep_df,DataSetCatCon_without_target, data_prep_keyvalue, DataSetCatCon_dict
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import count_parameters, classification_scores, mean_sq_error
from augmentations import embed_data_mask
from augmentations import add_noise
from pretraining import SAINT_pretrain,pretrain_process, pretrain_process_dict

import os
import numpy as np
parser = argparse.ArgumentParser()

parser.add_argument('--dset_id', default = "1461", type=int)
parser.add_argument('--vision_dset', action = 'store_true')
parser.add_argument('--task', default = "binary", type=str,choices = ['binary','multiclass','regression'])
parser.add_argument('--cont_embeddings', default='MLP', type=str,choices = ['MLP','Noemb','pos_singleMLP'])
parser.add_argument('--embedding_size', default=32, type=int)
parser.add_argument('--transformer_depth', default=6, type=int)
parser.add_argument('--attention_heads', default=8, type=int)
parser.add_argument('--attention_dropout', default=0.1, type=float)
parser.add_argument('--ff_dropout', default=0.1, type=float)
parser.add_argument('--attentiontype', default='colrow', type=str,choices = ['col','colrow','row','justmlp','attn','attnmlp'])

parser.add_argument('--optimizer', default='AdamW', type=str,choices = ['AdamW','Adam','SGD'])
parser.add_argument('--scheduler', default='cosine', type=str,choices = ['cosine','linear'])

parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batchsize', default=256, type=int)
parser.add_argument('--savemodelroot', default='./bestmodels', type=str)
parser.add_argument('--run_name', default='testrun', type=str)
parser.add_argument('--set_seed', default= 1 , type=int)
parser.add_argument('--dset_seed', default= 1 , type=int)
parser.add_argument('--active_log', action = 'store_true')

parser.add_argument('--pretrain', action = 'store_true')
parser.add_argument('--pretrain_epochs', default=50, type=int)
parser.add_argument('--pt_tasks', default=['contrastive','denoising'], type=str,nargs='*',choices = ['contrastive','contrastive_sim','denoising'])
parser.add_argument('--pt_aug', default=[], type=str,nargs='*',choices = ['mixup','cutmix'])
parser.add_argument('--pt_aug_lam', default=0.1, type=float)
parser.add_argument('--mixup_lam', default=0.3, type=float)

parser.add_argument('--train_noise_type', default=None , type=str,choices = ['missing','cutmix'])
parser.add_argument('--train_noise_level', default=0, type=float)

parser.add_argument('--ssl_samples', default= None, type=int)
parser.add_argument('--pt_projhead_style', default='diff', type=str,choices = ['diff','same','nohead'])
parser.add_argument('--nce_temp', default=0.7, type=float)

parser.add_argument('--lam0', default=0.5, type=float)
parser.add_argument('--lam1', default=10, type=float)
parser.add_argument('--lam2', default=1, type=float)
parser.add_argument('--lam3', default=10, type=float)
parser.add_argument('--final_mlp_style', default='sep', type=str,choices = ['common','sep'])


opt = parser.parse_args()
modelsave_path = os.path.join(os.getcwd(),opt.savemodelroot,opt.task,str(opt.dset_id),opt.run_name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Specify the dataset ID you want to download

df = pd.read_csv('bank_data.csv')
nan_percentage = 0.5
df.replace("unknown", np.nan, inplace=True)

cont_columns=['V6', 'V13', 'V1', 'V12', 'V15', 'V14', 'V10']

for column in cont_columns:
    nan_percentage = 0.1  # e.g., 10% of values

    # Calculate the number of values to replace
    num_nan = int(len(df) * nan_percentage)

    # Randomly select indices to replace with NaN
    nan_indices = df.sample(n=num_nan, random_state=1).index

    # Set the selected indices in the specified column to NaN
    df.loc[nan_indices, column] = np.nan



df_dict = df.to_dict(orient='records')

column_headers = list(df_dict[0].keys())
cat_columns = []
cont_columns = []
for column in df.columns:
    # Remove NaNs temporarily
    non_na_data = df[column].dropna()
    see = pd.api.types.is_numeric_dtype(non_na_data)
    # Check if column is numerical by attempting conversion
    if pd.api.types.is_numeric_dtype(non_na_data) or pd.to_numeric(non_na_data, errors='coerce').notna().all():
        cont_columns.append(column)
    else:
        cat_columns.append(column)

# cat_columns=['V2', 'V3', 'V4', 'V5', 'V7', 'V8', 'V9', 'V11', 'V16','Class']
# cont_columns=['V6', 'V13', 'V1', 'V12', 'V15', 'V14', 'V10']

#cat_dims, cat_idxs, con_idxs, X_train, X_valid, X_test, train_mean, train_std = data_prep_df(df,cat_columns,cont_columns, datasplit=[.65, .15, .2])
cat_dims, cat_idxs, con_idxs, key_dims, X_train, X_valid, X_test, train_mean, train_std = data_prep_keyvalue(df_dict,cat_columns,cont_columns, datasplit=[.65, .15, .2])
continuous_mean_std = np.array([train_mean,train_std]).astype(np.float32) 

##### Setting some hyperparams based on inputs and dataset
_,nfeat = X_train['data'].shape
if nfeat > 100:
    opt.embedding_size = min(4,opt.embedding_size)
    opt.batchsize = min(64, opt.batchsize)
if opt.attentiontype != 'col':
    opt.transformer_depth = 1
    opt.attention_heads = 4
    opt.attention_dropout = 0.8
    opt.embedding_size = 16
    if opt.optimizer =='SGD':
        opt.ff_dropout = 0.4
        opt.lr = 0.01
    else:
        opt.ff_dropout = 0.8

# train_ds = DataSetCatCon_without_target(X_train, cat_idxs, continuous_mean_std)
# trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True,num_workers=4)

# valid_ds = DataSetCatCon_without_target(X_valid, cat_idxs, continuous_mean_std)
# validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False,num_workers=4)

# test_ds = DataSetCatCon_without_target(X_test, cat_idxs, continuous_mean_std)
# testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False,num_workers=4)

train_ds = DataSetCatCon_dict(X_train, cat_idxs, continuous_mean_std)
trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True,num_workers=4)

valid_ds = DataSetCatCon_dict(X_valid, cat_idxs, continuous_mean_std)
validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False,num_workers=4)

test_ds = DataSetCatCon_dict(X_test, cat_idxs, continuous_mean_std)
testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False,num_workers=4)

model = SAINT_dict(
key_dims=key_dims,
categories = tuple(cat_dims), 
num_continuous = len(con_idxs),                
dim = opt.embedding_size,                           
dim_out = 1,                       
depth = opt.transformer_depth,                       
heads = opt.attention_heads,                         
attn_dropout = opt.attention_dropout,             
ff_dropout = opt.ff_dropout,                  
mlp_hidden_mults = (4, 2),       
cont_embeddings = opt.cont_embeddings,
attentiontype = opt.attentiontype,
final_mlp_style = opt.final_mlp_style,
)
model.to(device)
model = pretrain_process_dict(model, trainloader, validloader, opt,device, modelsave_path)
