import getpass
import numpy as np 
from pyspark.sql import SparkSession

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics

from pyspark.sql.functions import split
from pyspark.sql.functions import sqrt


from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

## for extension 4.2
import pynndescent 
pynndescent.rp_trees.FlatTree.__module__  = "pynndescent.rp_trees"
# UMAP
import umap.umap_ as umap
import umap.plot

from matplotlib import pyplot as plt

def main(spark, netID):
    # train_data = spark.read.parquet('hdfs:/user/lc4866/train_data.parquet')
    # val_data = spark.read.parquet('hdfs:/user/lc4866/val_data.parquet')
    # test_data = spark.read.parquet('hdfs:/user/lc4866/test_data.parquet')
    train_data = spark.read.parquet('train_data_large.parquet')
    val_data = spark.read.parquet('val_data_large.parquet')
    test_data = spark.read.parquet('test_data_large.parquet')
    
    movies = spark.read.csv("movies-large.csv", header=True, schema = 'movieId int, title STRING, genres STRING')

    ##### data we are using to plot the UMAP
    using_data = train_data
    rank = 300
    reg = 0.01
    ##### modified above 

    als = ALS(rank= rank, maxIter=10, regParam=reg, 
              userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop",
              nonnegative = True, 
             implicitPrefs = True)
    model = als.fit(using_data)
    predictions = model.transform(using_data)
    # scoreAndLabels = val_data.join(predictions,'userId').rdd.map(lambda tup: (tup[2], tup[4]))


    ## extension 4.1
    print("******* extension 4.1 *******")
    predictions.createOrReplaceTempView('predictions')
    movies.createOrReplaceTempView("movies")

    split_col = split(movies['genres'], "\|")
    
    ## genre_df: movieId ~ genre
    genre_df = movies.withColumn('genre', split_col.getItem(0)).select('movieId', 'title', 'genre')
    genre_df.createOrReplaceTempView('genre_df')
    print("Number of unique genres: \n ")
    print(genre_df.select('genre').distinct().count())
    
    ## merged_df: per user: his avg rating, avg prediction confidence
    merged_df = spark.sql('select userId, mean(rating) as avg_rating, mean(prediction) as avg_prediction\
                      from predictions\
                      group by userId\
                      order by userId asc')
    merged_df.createOrReplaceTempView('merged_df')
    # merged_df.show(10)

    ## df: per user, genre group, how many ratings
    df = spark.sql("select userId, tb.genre, count(tb.genre) as cnt \
                from (genre_df g right join predictions p on g.movieId = p.movieId) tb\
                group by userId, genre order by userId, count(genre) desc")
    
    df.createOrReplaceTempView('df')
    # df.show(10)

    ## user_df: per user, genre group, add avg_rating and low_prediction property for that user
    user_df = spark.sql('select m.userId, avg_rating, if(avg_prediction<(select percentile(avg_prediction, 0.5) from merged_df), 1, 0) low_prediction, genre, cnt\
                      from merged_df m join df d on m.userId=d.userId \
                      order by userId asc, avg_prediction asc')
    user_df.createOrReplaceTempView('user_df')
    user_df.show(10)     


    ## t- test staitiscs
    low_user_df = spark.sql("select * from user_df where low_prediction =1")
    low_user_df.createOrReplaceTempView('low_user_df')


    high_user_df = spark.sql("select * from user_df where low_prediction =0")
    high_user_df.createOrReplaceTempView('high_user_df')


    ## t_test tables
    low_test_df = spark.sql("select genre, mean(cnt) as xbar_1, count(cnt) as n1, variance(cnt) as s1_sq\
                        from low_user_df\
                        group by genre")
    low_test_df.createOrReplaceTempView('low_test_df')
    print("low_test_df(5):\n")
    low_test_df.show(5)


    high_test_df = spark.sql("select genre, mean(cnt) as xbar_2, count(cnt) as n2, variance(cnt) as s2_sq\
                        from high_user_df\
                        group by genre")
    high_test_df.createOrReplaceTempView('high_test_df')
    print("high_test_df(5):\n")
    high_test_df.show(5)

    t_test_df = spark.sql("select low.genre,  xbar_1, n1, s1_sq, xbar_2, n2, s2_sq, \
                                (xbar_1-xbar_2)/sqrt( (((n1-1)*s1_sq + (n2-1)*s2_sq)/(n1+n2-2)) * (1/n1+1/n2)) as test_statistic\
                        from low_test_df low join high_test_df high on low.genre = high.genre")
    t_test_df.createOrReplaceTempView('t_test_df')
    print("******* t-test table first 30 rows ******* \n")
    t_test_df.show(30)
    


    ## extension 4.2
    print("******* extension 4.2 ******* ")
    genre=movies
    genre.createOrReplaceTempView("genre")
    split_col = split(genre['genres'], "\|")
    # print(split_col)
    genre_df = genre.withColumn('genre', split_col.getItem(0))
    print("genre_df:")
    genre_df.show(5)

    print("model.itemfactors:")
    model.itemFactors.show(5)

    concat_df = genre_df.join(model.itemFactors,genre_df['movieId'] == model.itemFactors['id']).select('id','title','genre', 'features')#.rdd.map(lambda tup: (tup[2], tup[4]))
    concat_df.show(5)
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
    print("saving the 2D figure, using the data %s parquet, figure named: %s.png ", (using_data, using_data))
    umap.plot.points(val_t, labels = labels).get_figure().savefig('rank_%d_reg_%.3f_test_large.png'%(rank, reg))



if __name__ == "__main__":

    # Create the spark session object
    # spark = SparkSession.builder.appName('part1').getOrCreate()
    spark = SparkSession.builder.getOrCreate()

    # # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
