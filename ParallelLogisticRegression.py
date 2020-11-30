# -*- coding: utf-8 -*-
import numpy as np
import argparse
from time import time
from SparseVector import SparseVector
from LogisticRegression import readBeta,writeBeta,gradLogisticLoss,logisticLoss,lineSearch
from operator import add
from pyspark import SparkContext

def readDataRDD(input_file,spark_context):
    """  Read data from an input file. Each line of the file contains tuples of the form

                    (x,y)  

         x is a dictionary of the form:                 

           { "feature1": value, "feature2":value, ...}

         and y is a binary value +1 or -1.

         The result is stored in an RDD containing tuples of the form
                 (SparseVector(x),y)             

    """ 
    return spark_context.textFile(input_file)\
                        .map(eval)\
                        .map(lambda (x,y):(SparseVector(x),y))



	
def getAllFeaturesRDD(dataRDD):                
    """ Get all the features present in grouped dataset groupedDataRDD.
 
	The input is:
            - groupedDataRDD: a groupedRDD containing pairs of the form (partitionID,dataList), where 
              partitionID is an integer and dataList is a list of (SparseVector(x),y) values
 

        The return value is an RDD containing the above features.
    """                
    return dataRDD.flatMap(lambda (x,y):x.keys())\
                         .distinct()


def totalLossRDD(dataRDD,beta,lam = 0.0):
    """  Given a β represented by RDD betaRDD and a grouped dataset data represented by groupedDataRDD  compute 
         the regularized total logistic loss :

            L(β) = Σ_{(x,y) in data}  l(β;x,y)  + λ ||β ||_2^2             
        
         Inputs are:
            - groupedDataRDD: a groupedRDD containing pairs of the form (partitionID,dataList), where 
              partitionID is an integer and dataList is a list of (SparseVector(x),y) values
            - partitionsToFeaturesRDD: an RDD mapping partitions to relevant features, created by mapFeaturesToPartitionsRDD
            - betaRDD: a vector β represented as an RDD of (feature,value) pairs
            - lam: the regularization parameter λ

         The output should be the scalar value L(β)
    """

    
    # Compute  λ ||β ||_2^2

    reg = lam * beta.dot(beta)

    # Create intermediate rdd
   
    tot = dataRDD.map(lambda (x,y):logisticLoss(beta,x,y))\
                  .reduce(add) 

    return tot+reg        

def gradTotalLossRDD(dataRDD,beta,lam = 0.0):
    """  Given a β represented by RDD betaRDD and a grouped dataset data represented by groupedDataRDD  compute 
         the regularized total logistic loss :

            ∇L(β) = Σ_{(x,y) in data}  ∇l(β;x,y)  + 2λ β                
        
         Inputs are:
            - groupedDataRDD: a groupedRDD containing pairs of the form (partitionID,dataList), where 
              partitionID is an integer and dataList is a list of (SparseVector(x),y) values
            - featuresToPartitionsRDD: an RDD mapping features to relevant partitionIDs, created by mapFeaturesToPartitionsRDD
            - betaRDD: a vector β represented as an RDD of (feature,value) pairs
            - lam: the regularization parameter λ

         The output should be an RDD storing ∇L(β) as key value pairs of the form:
               (feature,value)
    """

    # Compute 2λ β
    reg = 2 * lam * beta


    tot = dataRDD.map(lambda (x,y):gradLogisticLoss(beta,x,y))\
                  .reduce(add)
    
    
    grad = tot + reg
    return grad
    



def test(dataRDD,beta):
    """ Output the quantities necessary to compute the accuracy, precision, and recall of the prediction of labels in a dataset under a given β.
        
        The accuracy (ACC), precision (PRE), and recall (REC) are defined in terms of the following sets:

                 P = datapoints (x,y) in data for which <β,x> > 0
                 N = datapoints (x,y) in data for which <β,x> <= 0
                 
                 TP = datapoints in (x,y) in P for which y=+1  
                 FP = datapoints in (x,y) in P for which y=-1  
                 TN = datapoints in (x,y) in N for which y=-1
                 FN = datapoints in (x,y) in N for which y=+1

        For #XXX the number of elements in set XXX, the accuracy, precision, and recall of parameter vector β over data are defined as:
         
                 ACC(β,data) = ( #TP+#TN ) / (#P + #N)
                 PRE(β,data) = #TP / (#TP + #FP)
                 REC(β,data) = #TP/ (#TP + #FN)

        Inputs are:
             - data: an RDD containing pairs of the form (x,y)
             - beta: vector β

        The return values are 
             - ACC,PRE,REC
    """
    
    # Create intermediate rdd
    pairs = dataRDD.map(lambda (x,y): (int(np.sign(beta.dot(x))), int(y)))\
			      .map(lambda (pred_label,true_label) : (pred_label,pred_label*true_label) )
    TP = 1.* pairs.filter(lambda (pred,gr) : (pred,gr) == (1,1)).count()                          
    FP = 1.* pairs.filter(lambda (pred,gr) : (pred,gr) == (1,-1)).count()                          
    TN = 1.* pairs.filter(lambda (pred,gr) : (pred,gr) == (-1,1)).count()                          
    FN = 1.* pairs.filter(lambda (pred,gr) : (pred,gr) == (-1,-1)).count()                          
    P = TP + FP
    N = TN + FN
    acc = (TP+TN)/(P+N)
    pre = TP/(TP+FP)
    rec = TP/(TP+FN)
    return acc,pre,rec



def train(dataRDD,beta_0,lam,max_iter,eps,test_data=None):
    """ Train a logistic model over a grouped dataset.
        
        Inputs are:
            - groupedDataRDD: a groupedRDD containing pairs of the form (partitionID,dataList), where 
              partitionID is an integer and dataList is a list of (SparseVector(x),y) values
            - featuresToPartitionsRDD: an RDD mapping features to relevant partitionIDs, created by mapFeaturesToPartitionsRDD
            - betaRDD: a vector β represented as an RDD of (feature,value) pairs
            - lam: the regularization parameter λ

            - max_iter: the maximum number of iterations
            - eps: the ε-tolerance
            - N: the number of partitions
            - test_data (optional): test data. If this is present, it is used to compute accuracy, precision, and recall at each step.

    """
    k = 0
    gradNorm = 2*eps
    beta = beta_0
    start = time()
    while k<max_iter and gradNorm > eps:
        obj =  totalLossRDD(dataRDD,beta,lam)
        
        grad = gradTotalLossRDD(dataRDD,beta,lam)
        gradNormSq = grad.dot(grad)
        gradNorm = np.sqrt(gradNormSq)
    
        fun = lambda  x: totalLossRDD(dataRDD,x,lam)
        gamma =  lineSearch(fun,beta,grad,obj,gradNormSq)
        beta = beta - gamma * grad

        if test_data == None:
            print 'k = ',k,'\tt = ',time()-start,'\tL(β_k) = ',obj,'\t||∇L(β_k)||_2 = ',gradNorm,' γ = ', gamma
        else:
	    acc,pre,rec = test(test_data,beta)
            print 'k = ',k,'\tt = ',time()-start,'\tL(β_k) = ',obj,'\t||∇L(β_k)||_2 = ',gradNorm,'\tACC = ',acc,'\tPRE = ',pre,'\tREC = ',rec
        k = k + 1

    return beta,gradNorm,k         




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Parallel Sparse Logistic Regression.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('traindata',default=None, help='Input file containing (x,y) pairs, used to train a logistic model')
    parser.add_argument('--testdata',default=None, help='Input file containing (x,y) pairs, used to test a logistic model')
    parser.add_argument('--beta', default='beta', help='File where beta is stored (when training) and read from (when testing)')
    parser.add_argument('--lam', type=float,default=0.0, help='Regularization parameter λ')
    parser.add_argument('--max_iter', type=int,default=100, help='Maximum number of iterations')
    parser.add_argument('--N',type=int,default=40,help='Level of parallelism/number of partitions')
    parser.add_argument('--eps', type=float, default=0.01, help='ε-tolerance. If the l2_norm gradient is smaller than ε, gradient descent terminates.') 

    verbosity_group = parser.add_mutually_exclusive_group(required=False)
    verbosity_group.add_argument('--verbose', dest='verbose', action='store_true')
    verbosity_group.add_argument('--silent', dest='verbose', action='store_false')
    parser.set_defaults(verbose=False)

    args = parser.parse_args()
  
    sc = SparkContext(appName='Parallel Sparse Logistic Regression')
    
    if not args.verbose :
        sc.setLogLevel("ERROR")        

    print 'Reading training data from',args.traindata
    traindataRDD = readDataRDD(args.traindata,sc)
    print 'Read',traindataRDD.count(),'training data points with',getAllFeaturesRDD(traindataRDD).count(),'features in total'
    
    if args.testdata is not None:
        print 'Reading test data from',args.testdata
        testdataRDD = readDataRDD(args.testdata,sc)
        print 'Read',testdataRDD.count(),'data points with',getAllFeaturesRDD(testdataRDD).count(),'features'
    else:
	testdataRDD = None

    beta0 = SparseVector({}) 

    print 'Training on data from',args.traindata,'with λ =',args.lam,', ε = ',args.eps,', max iter = ',args.max_iter
    beta, gradNorm, k = train(traindataRDD,beta_0=beta0,lam=args.lam,max_iter=args.max_iter,eps=args.eps,test_data=testdataRDD) 
    print 'Algorithm ran for',k,'iterations. Converged:',gradNorm<args.eps
    print 'Saving trained β in',args.beta
    writeBeta(args.beta,beta)

   	
