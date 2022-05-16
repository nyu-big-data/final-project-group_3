import getpass
import numpy as np 
import time
from pyspark.sql import SparkSession

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics

from pyspark.sql.functions import split

    
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType


## UMAP
import umap.umap_ as umap
import umap.plot
from matplotlib import pyplot as plt

def main(spark, netID):
    train_data = spark.read.parquet('hdfs:/user/lc4866/train_data.parquet')
    val_data = spark.read.parquet('hdfs:/user/lc4866/val_data.parquet')
    test_data = spark.read.parquet('hdfs:/user/lc4866/test_data.parquet')
    
    movies = spark.read.csv("movies.csv", header=True, schema = 'movieId int, title STRING, genres STRING')

    ##### data we are using to plot the UMAP
    using_data = val_data
    rank = 50
    reg = 0.01
    ##### modified above 

    als = ALS(rank= rank, maxIter=10, regParam=reg, 
              userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop",
              nonnegative = True, 
             implicitPrefs = True)
    model = als.fit(train_data)
    predictions = model.transform(using_data)
    # scoreAndLabels = val_data.join(predictions,'userId').rdd.map(lambda tup: (tup[2], tup[4]))


    # ## extension 4.1
    # print("******* extension 4.1 *******")
    # predictions.createOrReplaceTempView('predictions')
    # movies.createOrReplaceTempView("movies")

    # split_col = split(movies['genres'], "\|")
    # # print(split_col)
    # genre_df = movies.withColumn('genre', split_col.getItem(0)).select('movieId', 'title', 'genre')
    # genre_df.createOrReplaceTempView('genre_df')
    # print(f"Number of unique genres: {genre_df.select('genre').distinct().count()}")

    # merged_df = spark.sql('select userId, mean(rating) as avg_rating, mean(prediction) as avg_prediction\
    #                   from predictions\
    #                   group by userId\
    #                   order by userId asc')
    # # merged_df.show(10)
    # merged_df.createOrReplaceTempView('merged_df')


    # df = spark.sql("select userId, tb.genre, count(tb.genre) as cnt \
    #             from (genre_df g right join predictions p on g.movieId = p.movieId) tb\
    #             group by userId, genre order by userId, count(genre) desc")
    
    # df.createOrReplaceTempView('df')
    # # df.show(10)
    # user_genre = spark.sql('select userId, genre, cnt from \
    #               (select *, row_number() over(partition by userId order by cnt desc) as rk\
    #               from df) temp\
    #           where rk =1')
    #           # group by userId, genre order by max_cnt')
    # user_genre.createOrReplaceTempView('user_genre')
    # # user_genre.show(10)

    # user_df = spark.sql('select m.userId, avg_rating,  if(avg_prediction<0.5, 1, 0) low_prediction, genre, cnt\
    #                   from merged_df m join df d on m.userId=d.userId \
    #                   order by userId asc, avg_prediction asc')
    # user_df.createOrReplaceTempView('user_df')
    # # user_df.show(10)     

    # t_test= spark.sql("select high.*, low.low_mean, low.low_n, low.low_s2 from (select genre, avg(cnt) as high_mean, count(cnt) as high_n, std(cnt)*std(cnt) as high_s2 from user_df\
    #                where low_prediction=0 group by genre) high inner join\
    #                (select genre, avg(cnt) as low_mean, count(cnt) as low_n, std(cnt)*std(cnt) as low_s2 from user_df\
    #                where low_prediction=1 group by genre) low  on high.genre=low.genre \
    #                where high_n>0 and low_n>0 and high_mean>0 and low_mean>0 and high_s2>0 and low_s2>0")
     
    # mul_udf = udf(lambda high_mean, high_n, high_s2, low_mean, low_n, low_s2:  abs(high_mean- low_mean)/((high_s2/high_n + low_s2/low_n)**0.5) , DoubleType())  
    # temp = t_test.withColumn("result", mul_udf(t_test['high_mean'], t_test['high_n'],t_test['high_s2'],t_test['low_mean'],t_test['low_n'],t_test['low_s2'])) 
    
    # print(f"******* t-test table first 30 rows ******* \n: {temp.show(30)}")


    ## extension 4.2
    print("******* extension 4.2 ******* ")
    movies.createOrReplaceTempView("genre")
    split_col = split(genre['genres'], "\|")
    # print(split_col)
    genre_df = genre.withColumn('genre', split_col.getItem(0))
    # genre_df.show()

    concat_df = genre_df.join(model.itemFactors,'id').select('id','title','genre', 'features')#.rdd.map(lambda tup: (tup[2], tup[4]))

    concat_df.createOrReplaceTempView("concat_df")
    grouped = spark.sql("select genre, count(*) as cnt\
                          from concat_df \
                          group by genre\
                          order by cnt desc")
    print("the top 20 rows of grouped: ", grouped.show(20))

    itemFactors =concat_df.select('features').rdd.flatMap(lambda x: x).collect()
    arr = np.reshape(np.array(itemFactors),(len(itemFactors), len(itemFactors[0])))
    # print(arr.shape)
    # print("type(arr): ", type(arr))
    mapper = umap.UMAP()
    val_t = mapper.fit(arr)


    ## plot and save image
    labels =np.array(concat_df.select('genre').rdd.flatMap(lambda x: x).collect())
    # plot=umap.plot.points(val_t, labels = labels,')
    fig = plt.figure()
    print(f"saving the 2D figure, using the data {using_data}.parquet, figure named: {using_data.png}")
    umap.plot.points(val_t, labels = labels).get_figure().savefig(f'{using_data}.png')



if __name__ == "__main__":

    # Create the spark session object
    # spark = SparkSession.builder.appName('part1').getOrCreate()
    spark = SparkSession.builder.getOrCreate()

    # # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)