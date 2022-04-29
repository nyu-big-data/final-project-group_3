from pyspark.sql import SparkSession
import getpass
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as func
from pyspark.mllib.evaluation import RankingMetrics

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import Row

from pyspark.mllib.evaluation import RegressionMetrics


def main(spark, netID):
    train_data= spark.read.parquet('hdfs:/user/lc4866/train_data.parquet')
    val_data= spark.read.parquet('hdfs:/user/lc4866/val_data.parquet')
    test_data= spark.read.parquet('hdfs:/user/lc4866/test_data.parquet')

    ## regs = [0.01]#,0.05,0.1]#0.2,0.3,0.4,0.5]
    ## ranks = [100]#,100,200,500]

    ## best_rank=0
    ## best_reg=0
    ## MSE_dir ={}
    ## for rank in ranks:
    ##   for reg in regs:
    ##     print("current rank, reg: ", rank, reg)
    ##     als = ALS(rank=rank, maxIter=10, regParam=reg, 
    ##               userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
    ##     print("als = ALS passed...")
    ##     model = als.fit(train_data)
    ##     print("model fitted....")
    ##     predictions = model.transform(val_data)
    ##     print("model transformed...")
    ##     scoreAndLabels = val_data.join(predictions,'userId').rdd.map(lambda tup: (tup[2], tup[4]))
    ##     print("scoreAndLabels: ", scoreAndLabels)
    ##     metrics = RegressionMetrics(scoreAndLabels)
    ##     print("metrics defined..")
    ##     mse = metrics.rootMeanSquaredError
    ##     print("mse calculated")
    ##     MSE_dir[(reg, rank)] = mse
    ##     print("Finished current rank, reg: ", rank, reg)


    ## sorted_dir = {k:v for k, v in sorted(MSE_dir.items(), key=lambda x: x[1], reverse=False)}
    ## print(f"best reg, rank = {list(sorted_dir.items())[0][0]}")


    ## fit best ALS model on train set; predict on test set
    ## (best_reg, best_rank) = (list(sorted_dir.items())[0][0])
    best_reg = 0.1
    best_rank =100
    best_als = ALS(rank=best_rank, maxIter=10, regParam=best_reg, 
              userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
    best_model = best_als.fit(train_data)
    ## predictions = best_model.transform(test_data)
    
    ## MSE
    # scoreAndLabels = test_data.join(predictions,'userId').rdd.map(lambda tup: (tup[2], tup[4]))
    # metrics = RegressionMetrics(scoreAndLabels)
    # mse = metrics.rootMeanSquaredError
    # print(f"using best reg ={best_reg}, best rank = {best_rank}, get test mse = {mse}")

    ## MAP
    ## pred = predictions.orderBy(predictions['prediction'].desc()).groupby('userId').agg(func.collect_set('movieId').alias('pred'))
    ## true = test_data.orderBy(test_data['rating'].desc()).groupby('userId').agg(func.collect_set('movieId').alias('true'))
    ## predAndtrue = pred.join(true, 'userId').rdd.map(lambda row: (row[1], row[2])) 
    ## test_map = RankingMetrics(predAndtrue).meanAveragePrecisionAt(100) 

    ## print(f'test MAP is {test_map}')


    test_users = test_data.select("userId").distinct()
    test_users_rec = best_model.recommendForUserSubset(test_users, 100) 
    pred= test_users_rec.select(test_users_rec['userId'], test_users_rec.recommendations['movieId'].alias('pred'))
    true = test_data.orderBy(test_data['rating'].desc()).groupby('userId').agg(func.collect_set('movieId').alias('true'))
    print("pred: \n", pred.show())
    print("true: \n", true.show())
    pred.join(true, 'userId').show(10)
    predAndtrue = pred.join(true, 'userId').rdd.map(lambda row: (row[1], row[2])) 
    test_map = RankingMetrics(predAndtrue).precisionAt(100)
    print(f"using best reg ={best_reg}, best rank = {best_rank}, get test MAP = {test_map}")




if __name__ == "__main__":

    # Create the spark session object
    # spark = SparkSession.builder.appName('part1').getOrCreate()
    spark = SparkSession.builder.getOrCreate()

    # # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
