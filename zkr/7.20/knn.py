# -*- coding: utf-8 -*-

"""kNN最近邻算法最重要的三点：
   (1)确定k值。k值过小，对噪声非常敏感；k值过大，容易误分类
   (2)采用适当的临近性度量。对于不同的类型的数据，应考虑不同的度量方法。除了距离外，也可以考虑相似性。
   (3)数据预处理。需要规范数据，使数据度量范围一致。
"""
from numpy import *
from operator import *
def knn(new_data,dataset,labels,k):
    datasetSize = dataset.shape[0]  #shape[0]表示矩阵dataset的行数，也就是dataset中的坐标点个数
    DiffMat = tile(new_data,(datasetSize,1)) - dataset
    SqDiffMat = DiffMat ** 2                #矩阵中每一个元素都平方
    SqDistances = sum(SqDiffMat,axis = 1) #sum方法返回的一定时只有一行的矩阵，当axis = 1，矩阵每行的所有列相加
                                          #当axis = 0，矩阵的每列的所有行相加
    Distances = SqDistances ** 0.5          #矩阵中每一个元素都开方
    '''argsort:根据元素的大小从小到大排序，返回下标.如某元素是最小的，则该元素的位置为0'''
    SortDistances = Distances.argsort()
    LabelsCount = {}
    '''选择距离最小的 k 个点'''
    for i in range(k):
        Label = labels[SortDistances[i]]
        LabelsCount[Label] = LabelsCount.get(Label,0) + 1   #统计距离最近的 label们 的个数
 
    '''在前k个距离最近的点中，出现最多次的label就是测试点的类型'''
    MaxNum = 0
    for key,num in LabelsCount.items():
        if num > MaxNum:
            MaxNum = num
            InputLabel = key
 
    return InputLabel