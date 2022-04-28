from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import Row
from tqdm import tqdm

from pyspark.mllib.evaluation import RegressionMetrics
regs = [0.01]#,0.05,0.1,0.2,0.3,0.4,0.5]
ranks = [5,10,50,100,200,500]

best_rank=0
best_reg=0
MSE_dir ={}
for rank in tqdm(ranks):
  for reg in regs:
    als = ALS(rank=rank, maxIter=10, regParam=reg, 
              userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
    model = als.fit(train_data)
    predictions = model.transform(val_data)
    scoreAndLabels = val_data.join(predictions,'userId').rdd.map(lambda tup: (tup[2], tup[5]))
    metrics = RegressionMetrics(scoreAndLabels)
    mse = metrics.rootMeanSquaredError
    MSE_dir[(reg, rank)] = mse

sorted_dir = {k:v for k, v in sorted(MSE_dir.items(), key=lambda x: x[1], reverse=False)}
print(f"best reg, rank = {list(sorted_dir.items())[0][0]}")

(best_reg, best_rank) = (list(sorted_dir.items())[0][0])

als = ALS(rank=best_rank, maxIter=10, regParam=best_reg, 
          userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
model = als.fit(train_data)
predictions = model.transform(test_data)
scoreAndLabels = test_data.join(predictions,'userId').rdd.map(lambda tup: (tup[2], tup[5]))
metrics = RegressionMetrics(scoreAndLabels)
mse = metrics.rootMeanSquaredError
print(f"using best reg ={best_reg}, best rank = {best_rank}, get test mse = {mse}")