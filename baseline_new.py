# -*- coding: utf-8 -*- 

import getpass
import numpy as np
# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql import DataFrameStatFunctions as statFunc
from pyspark.sql.functions import isnan, when, count, col
from pyspark.mllib.evaluation import RankingMetrics

from pyspark.sql.functions import udf, struct
from pyspark.sql.types import FloatType



def main(spark, netID):
    '''Main routine for Final Project 
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    ''' 

    train_dt= spark.read.parquet('hdfs:/user/sj2539/train_data_large.parquet')
    val_dt= spark.read.parquet('hdfs:/user/sj2539/val_data_large.parquet')
    test_dt= spark.read.parquet('hdfs:/user/sj2539/test_data_large.parquet')

    train_dt.createOrReplaceTempView('train_dt')
    val_dt.createOrReplaceTempView('val_dt')
    test_dt.createOrReplaceTempView('test_dt')

    print('Find the top 100 popular movies from the training set')
    rating_num= train_dt.groupBy('movieId', 'rating').count()  
    mul_udf = udf(lambda x, y: x*y , FloatType()) 
    rating_num = rating_num.withColumn("result", mul_udf(rating_num['rating'], rating_num['count']))
    rating_num.createOrReplaceTempView('rating_num')

    # select out the top 100 movies
    top_100_movie= spark.sql('select movieId, cast(sum(result) as float)/sum(count)  weight_socre \
                                from  rating_num \
                                group by movieId \
                                order by weight_socre desc limit 100')

    #filter_val = val_dt.where(val_dt.movieId.isin([i for i in train_dt.select('movieId').distinct()])) 
    #filter_test = test_dt.where(test_dt.movieId.isin([i for i in train_dt.select('movieId').distinct()]))

    filter_val = val_dt.where(val_dt['movieId'].isin(train_dt.select('movieId').distinct().rdd.flatMap(lambda x: x).collect()))
    filter_test = test_dt.where(test_dt['movieId'].isin(train_dt.select('movieId').distinct().rdd.flatMap(lambda x: x).collect()))


    print('result on validation set')
    val_users = filter_val.select("userId").distinct()
    rec_list = top_100_movie.select(top_100_movie.movieId).agg(func.collect_list('movieId')) 

    rec= val_users.rdd.cartesian(rec_list.rdd).map(lambda row: (row[0][0], row[1][0])).toDF() 
    pred = rec.select(rec._1.alias('userId'), rec._2.alias('pred')) 
    true = filter_val.orderBy(filter_val['rating'].desc()).groupby('userId').agg(func.collect_set('movieId').alias('true'))
    predAndtrue = pred.join(true, 'userId').rdd.map(lambda row: (row[1], row[2]))

    val_map = RankingMetrics(predAndtrue).precisionAt(100) 

    print('result on test set')
    test_users = filter_test.select("userId").distinct()
    rec_list = top_100_movie.select(top_100_movie.movieId).agg(func.collect_list('movieId')) 
    rec= test_users.rdd.cartesian(rec_list.rdd).map(lambda row: (row[0][0], row[1][0])).toDF() 
    pred = rec.select(rec._1.alias('userId'), rec._2.alias('pred')) 
    true = filter_test.orderBy(filter_test['rating'].desc()).groupby('userId').agg(func.collect_set('movieId').alias('true'))
    predAndtrue = pred.join(true, 'userId').rdd.map(lambda row: (row[1], row[2]))

    test_map = RankingMetrics(predAndtrue).precisionAt(100) 

    print('Metric Performance on validation and test')
    # Use MAP as evaluation metric
    print(f'MAP on validation set = {val_map}') 
    print(f'MAP on test set = {test_map}') 

    # Use Recall as evaluation metric
    #print(f'recall on validation set = {val_recall}') 
    #print(f'recall on test set = {test_recall}') 

if __name__ == "__main__":

    # Create the spark session object
    # spark = SparkSession.builder.appName('part1').getOrCreate()
    spark = SparkSession.builder.getOrCreate()

    # # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
