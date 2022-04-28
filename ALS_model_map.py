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

def main(spark, netID):
    train= spark.read.parquet('hdfs:/user/lc4866/train_data.parquet')
    val= spark.read.parquet('hdfs:/user/lc4866/val_data.parquet')
    test= spark.read.parquet('hdfs:/user/lc4866/test_data.parquet')

    train.createOrReplaceTempView('train')
    val.createOrReplaceTempView('val')
    test.createOrReplaceTempView('test')

    def als_model(train, rank, reg):
      als = ALS(rank=rank, maxIter=10, regParam=reg, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
      model = als.fit(train)
      return model 



    regs = [0.01,0.05,0.1,0.2,0.3,0.4,0.5]
    ranks = [10, 20, 30, 40, 50]

    best_map=0
    best_rank=0
    best_reg=0

    for rank in ranks:
      for reg in regs:
        als= als_model(train, rank, reg)  
        val_users = val.select("userId").distinct()
        val_users_rec = als.recommendForUserSubset(val_users, 100) 
        pred= val_users_rec.select(val_users_rec['userId'], val_users_rec.recommendations['movieId'].alias('pred'))
        true = val.orderBy(val['rating'].desc()).groupby('userId').agg(func.collect_set('movieId').alias('true'))
        predAndtrue = pred.join(true, 'userId').rdd.map(lambda row: (row[1], row[2])) 

        val_map = RankingMetrics(predAndtrue).meanAveragePrecisionAt(100)
        
        if val_map > best_map:
            best_map = val_map
            best_rank = rank
            best_reg = reg 

    best_als= als_model(train, best_rank, best_reg)  
    test_users = test.select("userId").distinct()
    test_users_rec = als.recommendForUserSubset(test_users, 100) 
    pred= test_users_rec.select(test_users_rec['userId'], test_users_rec.recommendations['movieId'].alias('pred'))
    true = test.orderBy(test['rating'].desc()).groupby('userId').agg(func.collect_set('movieId')).a




if __name__ == "__main__":

    # Create the spark session object
    # spark = SparkSession.builder.appName('part1').getOrCreate()
    spark = SparkSession.builder.getOrCreate()

    # # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
