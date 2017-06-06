# -*- coding: utf-8 -*-
"""
Created on Thu May 11 08:14:00 2017

Aim: to set up decision tree algorithm

@author: rogue
"""

from math import log  
import operator  
  
''''' 
   函数功能：计算给定集合的熵 
   输入：给定数据集 
   输出：熵的值 
'''  
def calculateEntropy(dataSet):  
    # 计算数据集共有多少元组  
    numOfObj = len(dataSet)  
      
    # 用labelCounts存储数据集的分类标签  
    labelCounts = {}  
      
    # 遍历数据集  
    for obj in dataSet:  
        # 取分类标签  
        currentLabel = obj[-1]  
        # 如果标签不在labelCounts中，则在labelCounts中插入  
        # 该标签键，并设值为0  
        if currentLabel not in labelCounts.keys():  
            labelCounts[currentLabel] = 0  
        # 该标签值 +1  
        labelCounts[currentLabel] += 1  
      
    entropy = 0.0  
    # 遍历labelCounts，根据熵的公式计算值  
    for label in labelCounts:  
        # 该标签在总数据集中占的概率  
        prob = float(labelCounts[label])/numOfObj  
        entropy -= prob * log(prob,2)  
    return entropy  
  
''''' 
    函数功能：根据划分属性轴和某一属性值，得到划分后的数据子集 
    输入：给定数据集dataSet，划分属性轴axis，给定属性值value 
    输出：划分后的数据子集 
'''  
def splitDataSet(dataSet, axis, value):  
    retDataSet = []  
    # 遍历dataSet中的所有对象，如果其在axis轴上值为value  
    # 则将该元组存入retDataSet[]，除了该轴数据  
    for obj in dataSet:  
        if obj[axis] == value:  
            reducedObj = obj[:axis]  
            reducedObj.extend(obj[axis+1:])  
            retDataSet.append(reducedObj)  
    return retDataSet  
  
''''' 
    函数功能：选择划分效果最好的属性 
    输入：给定数据集 
    输出：划分属性轴的下标 
'''  
def bestFeatureToSplit(dataSet):  
    # 初始化信息增益和划分属性的轴下标  
    maxInfoGain = 0.0  
    splitAxis = -1  
      
    # 计算给定数据集的熵  
    wholeEntropy = calculateEntropy(dataSet)  
      
    # 计算共有多少属性，-1的含义是去掉分类标签那一列  
    featureNum = len(dataSet[0]) - 1  
      
    # 计算共有多少元组  
    trainingNum = len(dataSet)  
      
    # 对每一个属性，对其进行划分，并计算信息增益，如果该信息增益比  
    # 当前记录的信息增益值大，则更新;否则，继续下一个属性  
    for i in range(featureNum):  
        # 获得属性轴为i的所有属性值，并存储在featureList中  
        ''''' 
            注意这里书上用两行代码实现，更简单 
            featureList = [example[j] for example in dataSet] 
            uniqueVals = set(featureList)   # set is unique 
        '''  
        featureList = []  
        for j in range(trainingNum):  
            if dataSet[j][i] not in featureList:  
                featureList.append(dataSet[j][i])  
            else:  
                continue  
              
        # 现在已获得了第i+1列属性的所有值，计算共有多少不同值  
        iFeatureNum = len(featureList)  
          
        # 计算划分后所有数据子集的熵和  
        splitedEntropy = 0.0  
        for k in range(iFeatureNum):  
            # 获得划分子集  
            retDataSet = splitDataSet(dataSet, i, featureList[k])  
            # 得到子集元组数量  
            retDataSetNum = len(retDataSet)  
            # 得到子集熵  
            retDataSetEntropy = calculateEntropy(retDataSet)  
            # 累加  
            splitedEntropy += (float(retDataSetNum)/float(trainingNum))*retDataSetEntropy  
          
        # 判断当前划分的信息增益是否大于maxInfoGain  
        if (wholeEntropy - splitedEntropy) > maxInfoGain:  
            maxInfoGain = wholeEntropy - splitedEntropy  
            splitAxis = i  
        else:  
            continue  
      
    return splitAxis  
              
''' 
    函数功能：当划分结束时，如果某一子集中的所有分类标签还不相同，那么 
            用这个函数，来选择该空间中某一标签值最多的作为标签，与kNN 
            算法中的相同 
    输入：标签list 
    输出：标签值最多的标签 
'''  
def majorityCount(classList):  
    classCount = {}  
    for vote in classList:  
        if vote not in classCount.keys():  
            classCount[vote] = 0  
        classCount[vote] += 1  
    # sort  
    sortedClassCount = sorted(classCount.iteritems(),  
                              key=operator.itemgetter(1),reverse=True)  
      
    return sortedClassCount[0][0]  
  
''''' 
    函数功能：递归建立决策树 
    输入：给定数据集dataSet，划分属性名称labels 
    输出：决策树（dictionary格式） 
'''  
def createTree(dataSet,labels):  
    # 获得该dataSet的所有标签  
    classList = [example[-1] for example in dataSet]  
      
    # 结束条件1：所有标签相同，则返回该标签  
    if classList.count(classList[0]) == len(classList):  
        return classList[0]  
    # 结束条件2：所有属性已经划分，返回最多数量的标签  
    if len(dataSet[0]) == 1:  
        return majorityCount(classList)  
      
    # 若没结束，则划分  
    bestFeature = bestFeatureToSplit(dataSet)  
    bestFeatureLabel = labels[bestFeature]  
      
    # 用dictionary格式存储该划分  
    myTree = {bestFeatureLabel:{}}    
    # 从labels中删除该属性  
    del(labels[bestFeature])  
              
    # 获得该划分属性的所有属性值  
    featureValues = [example[bestFeature] for example in dataSet]  
    uniqueVals = set(featureValues)  # 取唯一值  
      
    # 对每一属性值递归求决策树  
    for value in uniqueVals:  
        # 用subLabels存储labels的值是因为，递归的时候会修改list的内容  
        subLabels = labels[:]  
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet,bestFeature,value),subLabels)  
      
    return myTree  
  
''''' 
    函数功能：创建数据集和属性名称 
    输出：返回数据集和属性名称list 
'''  
def createDataset():  
    dataSet = [['Sunny','Hot','High','Weak','No'],  
               ['Sunny','Hot','High','Strong','No'],  
               ['Overcast','Hot','High','Weak','Yes'],  
               ['Rain','Mild','High','Weak','Yes'],  
               ['Rain','Cool','Normal','Weak','Yes'],  
               ['Rain','Cool','Normal','Strong','No'],  
               ['Overcast','Cool','Normal','Strong','No'],  
               ['Sunny','Mild','High','Weak','No'],  
               ['Sunny','Cool','Normal','Weak','Yes'],  
               ['Rain','Mild','Normal','Weak','Yes'],  
               ['Sunny','Mild','Normal','Strong','Yes'],  
               ['Overcast','Mild','High','Strong','Yes'],  
               ['Overcast','Hot','Normal','Weak','Yes'],  
               ['Rain','Mild','High','Strong','No'],]  
    labels = ['Outlook','Temperature','Humidity','Wind']  
    return dataSet, labels  
  
dataSet, labels = createDataset()  
  
labelsCopy = labels[:]  
  
decisionTree = createTree(dataSet,labels)  
print (decisionTree)  
  
''''' 
    函数功能：根据得到的决策树进行预测 
    输入：决策树inputTree，属性名称列表featLabels，测试数据testVector 
    输出：预测分类标签 
'''  
def classify(inputTree, featLabels, testVector):  
    # 决策树是dictionary格式，所以取第一个键firstStr和值secondDict  
    firstStr = tuple(inputTree.keys())[0]  
    secondDict = inputTree[firstStr]  
      
    # 根据键（属性）获得其轴下标  
    featIndex = featLabels.index(firstStr)  
      
    # 循环递归判断  
    for key in secondDict.keys():  
        if testVector[featIndex] == key:  
            # 如果键的类型是dict，说明是dictionary，还需递归判断  
            if type(secondDict[key]).__name__ == 'dict':  
                classLabel = classify(secondDict[key],featLabels,testVector)  
            # 否则，判断结束，直接返回  
            else:  
                classLabel = secondDict[key]  
      
    return classLabel  
  
classLabel = classify(decisionTree,labelsCopy,['Sunny','Mild','High','Weak'])  
print(classLabel) 