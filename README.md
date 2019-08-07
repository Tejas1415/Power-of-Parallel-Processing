# ParallelProcessingFinalProj

## Linear Regression
To split the data into five folds, import splitData.py and run the following command: `splitData.makeFolds('filename', sparkContext)`

To run Linear Regression run the following command, where $1 is 1 for linear regression and 0 for logistic regression:
`spark-submit --executor-memory 100G --driver-memory 100G ParallelRegression_kfoldCV.py dataFolder 5 $1 --N 40 --maxiter numIterations --beta betaOutputFolder --perfOut performaceOutputFolder --lam LambdaValue`

Alternatively, run `sbatch regression.bash $1 $2 $3` where $1 is 1 for linear regression and 0 for logistic regression, $2 is the max number of iterations, and $3 is the lambda value.

To loop multiple values of lambda, run `sbatch loopLam.bash`. To change the range of lambda values, edit the file. 

## Random Forests
Go into the `random_forest` directory. There you will find two files: `RandomForestWithSpark.py` and `RandomForestWithoutSpark.py`. You can run these files with:

```bash
python RandomForestWithSpark.py
python RandomForestWithoutSpark.py
```


## Neural Networks
Change directory into the neural networks. Start by processing the data with:
```bash
python read_equal_data.py $PATH_TO_CSV
```
Once you have created the `x_train_eq.pkl` and `y_train_eq.pkl` files, you can run the model:
```
python main_file.py
```
