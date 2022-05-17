import getpass
import numpy as np 
from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql import DataFrameStatFunctions as statFunc
from pyspark.sql.functions import isnan, when, count, col
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.window import Window
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

import pandas as pd
import lightfm
from scipy.sparse import coo_matrix, csr_matrix
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k
from time import time

def main(spark, netID):

	train_als = spark.read.parquet('/home/sj2539/final-project-group_3/train_data_large.parquet') 
	test_als = spark.read.parquet('/home/sj2539/final-project-group_3/test_data_large.parquet')

	train_lfm= pd.read_parquet('/home/sj2539/final-project-group_3/train_data_large.parquet') 
	test_lfm= pd.read_parquet('/home/sj2539/final-project-group_3/test_data_large.parquet')

	best_als_rank= 300 
	best_als_reg= 0.01
	best_lfm_comp= 100
	best_lfm_reg= 0.1

	train_als.createOrReplaceTempView('train_als')
	test_als.createOrReplaceTempView('test_als') 

	als_time= [] 
	als_test_precision_ls= []

	lfm_time= [] 
	lfm_test_precision_ls= []

	train_movie_ratio= [0.2, 0.8]

	for ratio in train_movie_ratio: 
		used_id, rest_id = [i.rdd.flatMap(lambda x: x).collect() for i in train_als.select('movieId').distinct().randomSplit([ratio, 1.0- ratio], 1024)]
		temp_train_als= train_als.where(train_als.userId.isin(used_id))

		# ALS model
		als_start_time= time()
		als= ALS(rank=best_als_rank, maxIter=10, regParam=best_als_reg, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
		model = als.fit(temp_train_als)
		als_end_time= time()
		als_time.append(als_end_time - als_start_time)

		test_users = test_als.select("userId").distinct()
		test_users_rec = model.recommendForUserSubset(test_users, 100) 
		pred= test_users_rec.select(test_users_rec['userId'], test_users_rec.recommendations['movieId'].alias('pred'))
		true = test_als.groupby('userId').agg(func.collect_set('movieId').alias('true')) 
		predAndtrue = pred.join(true, 'userId').rdd.map(lambda row: (row[1], row[2])) 
		als_test_precision = RankingMetrics(predAndtrue).precisionAt(100) 
		als_test_precision_ls.append(np.mean(als_test_precision))

		# Lightfm model
		used_id= list(used_id)
		train_dt = Dataset()
		temp_train_lfm= train_lfm[train_lfm['movieId'].isin(used_id)]
		temp_test_lfm= test_lfm[test_lfm['movieId'].isin(used_id)]
		train_dt.fit((x for x in temp_train_lfm['userId'].unique()),(x for x in temp_train_lfm['movieId'].unique()))
		train_interaction, weight = train_dt.build_interactions(((row[1]['userId'], row[1]['movieId'], row[1]['rating']) for row in temp_train_lfm.iterrows())) 
		user_index_mapping = train_dt.mapping()[0]
		item_index_mapping = train_dt.mapping()[2]
		matrix_shape = train_dt.interactions_shape() 
		temp_test_lfm = temp_test_lfm[temp_test_lfm['userId'].isin(temp_train_lfm['userId']) & temp_test_lfm['movieId'].isin(temp_train_lfm['movieId'])]
		test_user = np.array([user_index_mapping[i] for i in temp_test_lfm['userId']])
		test_item = np.array([item_index_mapping[i] for i in temp_test_lfm['movieId']])
		test_rating = temp_test_lfm['rating']
		test_interaction = coo_matrix((test_rating, (test_user, test_item)), shape=matrix_shape)

		lfm_start_time= time()
		model = LightFM(no_components=10, loss='warp',item_alpha=0.01, user_alpha=0.01)
		model.fit(train_interaction, num_threads=1) 
		lfm_end_time= time()
		lfm_time.append(lfm_end_time - lfm_start_time) 
		lfm_test_precision = precision_at_k(model, test_interaction, k=100).mean() 
		lfm_test_precision_ls.append(lfm_test_precision)

	print(als_time, lfm_time,als_test_precision_ls,lfm_test_precision_ls)


if __name__ == "__main__":

    # Create the spark session object
    # spark = SparkSession.builder.appName('part1').getOrCreate()
    spark = SparkSession.builder.getOrCreate()

    # # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
