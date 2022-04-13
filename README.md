# DSGA1004 - BIG DATA
## Final project

*Handout date*: 2022-04-13

*Submission deadline*: 2022-05-17


# Overview

In the final project, you will apply the tools you have learned in this class to solve a realistic, large-scale applied problem.
Specifically, you will build and evaluate a collaborative-filter based recommender system. 

In either case, you are encouraged to work in **groups of up to 3 students**:

- Groups of 1--2 will need to implement one extension (described below) over the baseline project for full credit.
- Groups of 3 will need to implement two extensions for full credit.

## The data set

In this project, we'll use the [MovieLens](https://grouplens.org/datasets/movielens/latest/) datasets collected by 
> F. Maxwell Harper and Joseph A. Konstan. 2015. 
> The MovieLens Datasets: History and Context. 
> ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

The data is hosted in NYU's HPC environment under `/scratch/work/courses/DSGA1004-2021/movielens`.

Two versions of the dataset are provided: a small sample (`ml-latest-small`, 9000 movies and 600 users) and a larger sample (`ml-latest`, 58000 movies and 280000 users).
Each version of the data contains rating and tag interactions, and the larger sample includes "tag genome" data for each movie, which you may consider as additional features beyond
the collaborative filter.
Each version of the data includes a README.txt file which explains the contents and structure of the data which are stored in CSV files.

I strongly recommend to thoroughly read through the dataset documentation before beginning, and make note of the documented differences between the smaller and larger datasets.
Knowing these differences in advance will save you many headaches when it comes time to scale up.

## Basic recommender system [80% of grade]

1.  As a first step, you will need to partition the rating data into training, validation, and test samples as discussed in lecture.
    I recommend writing a script do this in advance, and saving the partitioned data for future use.
    This will reduce the complexity of your experiment code down the line, and make it easier to generate alternative splits if you want to measure the stability of your
    implementation.

2.  Before implementing a sophisticated model, you should begin with a popularity baseline model as discussed in class.
    This should be simple enough to implement with some basic dataframe computations.
    Evaluate your popularity baseline (see below) before moving on to the enxt step.

3.  Your recommendation model should use Spark's alternating least squares (ALS) method to learn latent factor representations for users and items.
    Be sure to thoroughly read through the documentation on the [pyspark.ml.recommendation module](https://spark.apache.org/docs/3.0.1/ml-collaborative-filtering.html) before getting started.
    This model has some hyper-parameters that you should tune to optimize performance on the validation set, notably: 
      - the *rank* (dimension) of the latent factors, and
      - the regularization parameter.

### Evaluation

Once you are able to make predictions—either from the popularity baseline or the latent factor model—you will need to evaluate accuracy on the validation and test data.
Scores for validation and test should both be reported in your write-up.
Evaluations should be based on predictions of the top 100 items for each user, and report the ranking metrics provided by spark.
Refer to the [ranking metrics](https://spark.apache.org/docs/3.0.1/mllib-evaluation-metrics.html#ranking-systems) section of the Spark documentation for more details.

The choice of evaluation criteria for hyper-parameter tuning is up to you, as is the range of hyper-parameters you consider, but be sure to document your choices in the final report.
As a general rule, you should explore ranges of each hyper-parameter that are sufficiently large to produce observable differences in your evaluation score.

If you like, you may also use additional software implementations of recommendation or ranking metric evaluations, but be sure to cite any additional software you use in the project.


### Using the cluster

Please be considerate of your fellow classmates!
The Peel cluster is a limited, shared resource. 
Make sure that your code is properly implemented and works efficiently. 
If too many people run inefficient code simultaneously, it can slow down the entire cluster for everyone.


## Extensions [20% of grade]

For full credit, implement an extension on top of the baseline collaborative filter model.
Again, if you're working in a group of 3, you must implement two extensions for full credit.

The choice of extension is up to you, but here are some ideas:

  - *Comparison to single-machine implementations*: compare Spark's parallel ALS model to a single-machine implementation, e.g. [lightfm](https://github.com/lyst/lightfm) or [lenskit](https://github.com/lenskit/lkpy).  Your comparison should measure both efficiency (model fitting time as a function of data set size) and resulting accuracy.
  - *Fast search*: use a spatial data structure (e.g., LSH or partition trees) to implement accelerated search at query time.  For this, it is best to use an existing library such as [annoy](https://github.com/spotify/annoy), [nmslib](https://github.com/nmslib/nmslib), or [scann](https://github.com/google-research/google-research/tree/master/scann) and you will need to export the model parameters from Spark to work in your chosen environment.  For full credit, you should provide a thorough evaluation of the efficiency gains provided by your spatial data structure over a brute-force search method.
  - *Cold-start*: using supplementary metadata (tags, genres, etc), build a model that can map observable data to the learned latent factor representation for items.  To evaluate its accuracy, simulate a cold-start scenario by holding out a subset of items during training (of the recommender model), and compare its performance to a full collaborative filter model.  *Hint:* you may want to use dask for this.
  - *Qualitative error analysis*: using your best-performing latent factor model, investigate the mistakes that it makes.  This can be done in a number of ways, including (but not limited to):
    - investigating the trends and genres of the users who produce the lowest-scoring predictions
    - visualizing the learned item representation via dimensionality reduction techniques (e.g. T-SNE or UMAP) with additional data for color-coding (genre tags, popularity, etc)
    - clustering users by their learned representations and identifying common trends in each cluster's consumption behavior

Other extension ideas are welcome as well, but please check with the instructional staff before proceeding.

## What to turn in

In addition to all of your code, produce a final report (not to exceed 6 pages), describing your implementation, evaluation results, and extensions.
Your report should clearly identify the contributions of each member of your group. 
If any additional software components were required in your project, your choices should be described and well motivated here.  

Include a PDF of your final report through Brightspace.  Specifically, your final report should include the following details:

- Link to your group's GitHub repository
- Documentation of how your train/validation/test splits were generated
    - Any additional pre-processing of the data that you decide to implement
- Choice of evaluation criteria
- Evaluation of popularity baseline on small and full datasets
- Documentation of latent factor model's hyper-parameters
- Evaluation of latent factor model on small and full datasets
- Documentation of extension(s)

Any additional software components that you use should be cited and documented with installation instructions.

## Timeline

It will be helpful to commit your work in progress to the repository.
Toward this end, we recommend the following timeline with a preliminary submission on 2022-04-29:

- [ ] 2022/04/22: popularity baseline model and evaluation on small subset.
- [ ] **2022/04/29**: checkpoint submission with baseline results on both small and large datasets.  Preliminary results for matrix factorization on the small dataset.
- [ ] 2022/05/06: scale up to the large dataset and develop extensions.
- [ ] 2022/05/17: final project submission.  **NO EXTENSIONS PAST THIS DATE.**
