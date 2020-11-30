from pyspark import SparkContext
import numpy as np

def addToTopK(topKSoFar,x,k):
    """ Augments a list of top-k elements with item x, and returns a new top-k list. 
    """
    topKSoFar.append(x)
    topKSoFar.sort(reverse=True)
    if len(topKSoFar)>k:
        topKSoFar.pop()
    return topKSoFar

def mergeTopK(topKSoFar1,topKSoFar2,k):
    """ Receives two top-k element lists and merges them
    """
    mergedList = topKSoFar1+topKSoFar2
    mergedList.sort(reverse=True)
    if len(mergedList)>k:
        mergedList = mergedList[:k]
    return mergedList

def my_takeOrdered(rdd,K):
    """ Implements takeOrdered via an aggregate operation"""
    return rdd.aggregate([],
           lambda tpk,x:addToTopK(tpk,x,K),
           lambda tpk1,tpk2:mergeTopK(tpk1,tpk2,K))

if __name__=="__main__":
    sc = SparkContext('local[10]','MyTakeOrdered')
    rdd = sc.parallelize(range(1000)).sample(False,0.5)
    ord1= my_takeOrdered(rdd,13)
    ord2= rdd.takeOrdered(13,key = lambda x:-x)
    print ord1,ord2


