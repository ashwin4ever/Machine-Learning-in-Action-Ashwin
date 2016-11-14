#KNN algorithm in python


from numpy import *
import operator
from matplotlib import pyplot as plt






def autoNorm(dataSet):

      """

      Normalize Formula:
      newVal = (oldVal - min) / (max - min)
      
      """

      
      #axis = 0 gives the column
      #axis = 1 gives the row
      minVals = dataSet.min(axis = 0)
      maxVals = dataSet.max(axis = 0)

      print("Dataset: " ,dataSet[0])
      
      print("min: " , minVals)
      print("max: " , maxVals)

      #max - min
      ranges = maxVals - minVals

      print("Ranges: " , ranges)

      print("shape: " , shape(dataSet))
      normData = zeros(shape(dataSet))
      print("norm: " , normData)

      print("data shape: " , dataSet.shape[0])

      m = dataSet.shape[0]
      
      #oldVal - min
      normData = dataSet - tile(minVals , (m , 1))
      print("norm data: " , normData)

      #oldVal - min / (max - min)
      normData = normData / tile(ranges , (m , 1))

      return normData , minVals , ranges
      


#read file
def file2matrix(fileName):
      fr = open(fileName)
      numLines = len(fr.readlines())
      #print(numLines)

      returnMat = zeros((numLines , 3))
      classLabels = []

      fr = open(fileName)
      idx = 0

      for line in fr.readlines():
            line = line.strip()
            #print(line)
            listFromLine = line.split('\t')
            #print(listFromLine[-1])
            
            returnMat[idx , :] = listFromLine[0 : 3]
            classLabels.append(float(listFromLine[-1]))
            idx += 1
      return returnMat , classLabels

      

def createDataSet():
      group = array([[1.0 , 1.1] , [1.0 , 1.0] , [0 , 0] , [0 , 0.1]])
      labels = ['A' , 'A' , 'B' , 'B']
      return group , labels


def knn(inX , dataSet , labels , k):
      dataSetSize = dataSet.shape[0]
      print("data set size: " , dataSetSize)
      diffMat = tile(inX , (dataSetSize , 1)) - dataSet
      #print(diffMat , "\n")
      sqDiffMat = diffMat**2
      sqDist = sqDiffMat.sum(axis = 1)
      dist = sqDist ** 0.5

      sortedIdx = dist.argsort()

      classCount = {}
      #print(dist , "\n")
      #print(sortedIdx)

      for i in range(k):
            voteLabel = labels[sortedIdx[i]]
            classCount[voteLabel] = classCount.get(voteLabel , 0) + 1

      #print("class count: " , classCount , "\n")

      sortedClassCount = sorted(classCount.items() , key = operator.itemgetter(1) , reverse = True)
      return sortedClassCount[0][0]




def testKNN():
      k = 0.10
      datingMat , datingLables = file2matrix("datingTestSet2.txt")
      normMat , ranges , minVals = autoNorm(datingMat)
      m = normMat.shape[0]
      #print(m)

      numTestVecs = int(m * k)
      #print("num test: " , numTestVecs)

      errorCount = 0.0
      #print(normMat[1 : 10])
      #print(normMat[1 : 5 , :])

      #var[n , :] will select the entire nth row in the matrix
      #var[: , n] will select the enitre nth column in the matrix

      
      for i in range(numTestVecs):
            classifierResult = knn(normMat[i , :] , normMat[numTestVecs : m , :] , \
                                   datingLables[numTestVecs : m] , 3)
            print("The classifier returned : %d , the real answer is : %d"\
                  %(classifierResult , datingLables[i]))

testKNN()
      

#g , l = createDataSet()
#print(g)
#print("\n")

#print(classify0([0 , 0] , g , l , 3))

#datingMat , datingLabel = file2matrix("datingTestSet2.txt")
#print(m)
#print(l[0 : 10])

#print(datingMat[0 , 1])

#n , r , minV = autoNorm(datingMat)

#print(n)

##fig = plt.figure()
##ax = fig.add_subplot(1 , 1 , 1)
##ax.scatter(datingMat[: , 1] , datingMat[: , 2] , (array(datingLabel))* 15.0 , (array(datingLabel))* 15.0)
##plt.show()
