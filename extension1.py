import pandas as pd
import lightfm
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k
import time

##tune params##
no_components = 30
alpha = 0.01

start = time.time()
train= pd.read_parquet('/home/sj2539/final-project-group_3/train_data_large.parquet')
val= pd.read_parquet('/home/sj2539/final-project-group_3/val_data_large.parquet')
test= pd.read_parquet('/home/sj2539/final-project-group_3/test_data_large.parquet')

# build train coo matrix
train_dt = Dataset()
train_dt.fit((x for x in train['userId'].unique()),(x for x in train['movieId'].unique()))
train_interaction, weight = train_dt.build_interactions(((row[1]['userId'], row[1]['movieId'], row[1]['rating']) for row in train.iterrows()))


model = LightFM(no_components=no_components, loss='warp',item_alpha=alpha, user_alpha=alpha)
model.fit(train_interaction, num_threads=1) 

train_precision = precision_at_k(model, train_interaction, k=100).mean()

print('rank: %d, alpha: %f, train MAP:  %f',(no_components, alpha, train_precision))

user_index_mapping = train_dt.mapping()[0]
item_index_mapping = train_dt.mapping()[2]
matrix_shape = train_dt.interactions_shape()

val = val[val['userId'].isin(train['userId']) & val['movieId'].isin(train['movieId'])]
test = test[test['userId'].isin(train['userId']) & test['movieId'].isin(train['movieId'])]

# use the val data to build a matrix with the same shape of train
val_user = np.array([user_index_mapping[i] for i in val['userId']])
val_item = np.array([item_index_mapping[i] for i in val['movieId']])
val_rating = val['rating'] 
val_interaction = coo_matrix((val_rating, (val_user, val_item)), shape=matrix_shape)
val_precision = precision_at_k(model, val_interaction, k=100).mean()

print('rank: %d, alpha: %f, val MAP:  %f',(no_components, alpha, val_precision))

# use the test data to build a matrix with the same shape of train
test_user = np.array([user_index_mapping[i] for i in test['userId']])
test_item = np.array([item_index_mapping[i] for i in test['movieId']])
test_rating = test['rating'] 
test_interaction = coo_matrix((test_rating, (test_user, test_item)), shape=matrix_shape)
test_precision = precision_at_k(model, test_interaction, k=100).mean()

print('rank: %d, alpha: %f, test MAP:  %f',(no_components, alpha, test_precision))
end = time.time()
total_time = end-start
print('time: %f',total_time)
