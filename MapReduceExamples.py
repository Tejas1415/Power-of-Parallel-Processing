from pyspark import SparkContext
import numpy as np
from operator import add



#Simple examples illustrating how some transforms can be implemented via map,
#reduce and aggregate

def my_collect(rdd):
    '''Computes collect() via a map and reduce'''
    return rdd.map(lambda x:[x])\
           .reduce(add)

def my_count(rdd):
    '''Computes count() via a map and reduce'''
    return rdd.map(lambda x:1)\
           .reduce(add)

#Two computations of the average
def my_average1(rdd):
    '''Computes the average via map and reduce '''
    total,count =rdd.map(lambda x:(x,1))\
           .reduce(lambda x,y:
                (x[0]+y[0],x[1]+y[1]) )
    return 1.*total/count
       
def my_average2(rdd):
    '''Computes the average via aggregate'''
    total,count =rdd.aggregate(
                 (0,0),                          #zero element 
                 lambda pair,data:               #seqOp
			(pair[0]+data,pair[1]+1),
                 lambda pair1,pair2:             #combOp
			(pair1[0]+pair2[0],pair1[1]+pair2[1])
                 )
    return 1.*total/count

  
if __name__=="__main__":
   sc = SparkContext('local[10]','MyTakeOrdered')
   rdd = sc.parallelize(range(1000)).sample(False,0.5)
   mean  =rdd.mean()  
   mean1 = my_average1(rdd)
   mean2 = my_average2(rdd)
   
   print mean,mean1,mean2
