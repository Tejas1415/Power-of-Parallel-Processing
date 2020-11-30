import sys
from pyspark import SparkContext

if __name__ == '__main__':
    sc = SparkContext(master='local[10]', appName='WordCount')
    lines = sc.textFile(sys.argv[1])
  
    lines.flatMap(lambda s: s.split()) \
         .map(lambda word: (word, 1)) \
         .reduceByKey(lambda x, y: x + y) \
                .sortBy(lambda (x,y):y, ascending=False) \
         .saveAsTextFile(sys.argv[2])                
