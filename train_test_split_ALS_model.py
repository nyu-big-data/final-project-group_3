import getpass
import numpy as np 
from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql import DataFrameStatFunctions as statFunc
from pyspark.sql.functions import isnan, when, count, col
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.window import Window


def main(spark, netID):
   '''Main routine for Final Project 
   Parameters
   ----------
   spark : SparkSession object
   netID : string, netID of student to find files in HDFS
   ''' 

   # Load the boats.txt and sailors.json data into DataFrame 
   ratings = spark.read.csv(f'hdfs:/user/lc4866/ratings.csv', schema='userId INT,  movieId INT, rating FLOAT, timestamp INT')
   movies = spark.read.csv(f'hdfs:/user/lc4866/movies.csv', schema='movieId INT,title STRING, genres STRING')
  
   # Give the dataframe a temporary view so we can run SQL queries
   ratings.createOrReplaceTempView('ratings')
   #movies.createOrReplaceTempView('movies')
   # data quality check
   # check duplicate
   print('check duplicate')
   duplicate= ratings.groupBy("userId", "movieId").count()
   duplicate.filter(duplicate['count']>=2).show() 

   # check missing
   print('check missing')
   ratings.select([count(when(isnan(c), c)).alias(c) for c in ratings.columns]).show()
   ratings= ratings.na.drop() 

   # check data range
   print('filter out the rating within 0.5-5')
   ratings= ratings.filter(ratings.rating <=5).filter( ratings.rating >=0.5) 


   # filter out the movies with more than 10 records 
   temp= ratings.groupby('movieId').count()
   temp= temp.filter( temp['count'] >=10) 
   base_ratings= ratings.where(ratings.movieId.isin([i for i in temp.select('movieId').distinct()])) 


   # filter out the users with more than 10 records 
   temp= base_ratings.groupby('userId').count()
   temp= temp.filter( temp['count'] >=10) 
   base_ratings= base_ratings.where(base_ratings.userId.isin([i for i in temp.select('userId').distinct()])) 


   print('Splitting into training, validation, and testing set based on user_Id')
   train_id, val_id, test_id = [i.rdd.flatMap(lambda x: x).collect() for i in base_ratings.select('userId').distinct().randomSplit([0.6, 0.2, 0.2], 1024)]


   train_dt = base_ratings.where(base_ratings.userId.isin(train_id))
   val_dt = base_ratings.where(base_ratings.userId.isin(val_id))
   test_dt = base_ratings.where(base_ratings.userId.isin(test_id)) 

   print('Adjusting training and validation set')
   window = Window.partitionBy('userId').orderBy('timestamp')
   val_dt = (val_dt.select("userId","movieId","rating", 'timestamp', func.row_number().over(window).alias("order_num")))


   val_dt.createOrReplaceTempView('val_dt')
   val_to_train= spark.sql('select val_dt.userId, val_dt.movieId, val_dt.rating from val_dt left join\
                         (select userId, count(*) total_num from  val_dt group by userId)temp \
                         on val_dt.userId= temp.userId where order_num <= cast( total_num/2 as int)')

   train_dt= train_dt.drop('timestamp')
   train_dt.createOrReplaceTempView('train_dt') 


   final_train = train_dt.union(val_to_train)

   print('Adjusting training and test set')
   window = Window.partitionBy('userId').orderBy('timestamp')
   #test_dt = (test_dt.select("userId","movieId","rating", func.row_number().over(window).alias("order_num")))


   test_dt.createOrReplaceTempView('test_dt') 
   test_to_train= spark.sql('select test_dt.userId, test_dt.movieId, test_dt.rating \
                          from test_dt  \
                          left join (select userId, count(*) total_num from test_dt group by userId) temp\
                         on test_dt.userId = temp.userId where order_num <= cast( total_num/2 as int)')

   final_train = final_train.union(test_to_train)
   final_train.repartition(500).write.mode('overwrite').parquet('train_data.parquet')

   final_val= spark.sql('select val_dt.userId, val_dt.movieId, val_dt.rating from val_dt left join (select userId, count(*) total_num from  val_dt group by userId)temp \
                         on val_dt.userId= temp.userId where order_num > cast( total_num/2 as int)')
   final_val.createOrReplaceTempView('final_val')
   final_val= spark.sql('select * from final_val where movieId in (select distinct movieId from final_train)')
   final_val.repartition(500).write.mode('overwrite').parquet('val_data.parquet')

   final_test= spark.sql('select test_dt.userId, test_dt.movieId, test_dt.rating from test_dt left join\
                         (select userId, count(*) total_num from test_dt group by userId) temp\
                         on test_dt.userId= temp.userId where order_num > cast( total_num/2 as int)')
   final_test.createOrReplaceTempView('final_test')
   final_test= spark.sql('select * from final_test where movieId in (select distinct movieId from final_train)')
   final_test.repartition(500).write.mode('overwrite').parquet('test_data.parquet')



if __name__ == "__main__":

    # Create the spark session object
    # spark = SparkSession.builder.appName('part1').getOrCreate()
    spark = SparkSession.builder.getOrCreate()

    # # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
