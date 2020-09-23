# -*- coding:utf-8 -*-
import os
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, LeaveOneOut, KFold, StratifiedKFold
#将学习集划分为训练集和验证集
from sklearn.model_selection import StratifiedShuffleSplit

# def loadDataset(fileDir,filename, count, Data_X=[], Data_Y=[]):
#为最后的独立测试集进行特征筛选，得到最后的测试文件,注意将第一行特征名删除，修改recover函数，将i的起点改为1
def recover_test(filename,filename1,filename2):
    f = open(filename)
    lines = f.readlines()    #top60特征
    #print(lines)

    # print(len(lines))
    f1 = open(filename1)
    lines1 = f1.readlines()  #原始数据，文件为feature.csv
    #print(lines1)
    dataMat = []
    dataMat1 = []
    #
    for i in range(0, len(lines)):
        data = lines[i].strip().split('\t')
        dataMat.append(data)
    f1.close()



    f2 = open(filename2, 'w')
    #f2.write(lines[0])             #将基因名写入f2

    for j in range(0, len(lines1)):
        data1 = lines1[j].strip().split(',')
        dataMat1.append(data1)

    N_i = len(dataMat1)   #行数
    N_j = len(dataMat1[0]) #列数


    count = 0
    count1 = 0
    # print(dataMat1[1][0])
    # for i in range(0, len(dataMat)):
    #     temp = dataMat[i][0]
    #     f2.write(temp)
    #     f2.write(',')
    #     for j in range(1, len(dataMat1[0])):
    #         for k in range(0,len)
    #         if temp == dataMat1[0][j]:
    #             #print(dataMat1[j][0])
    #             f2.write(dataMat1[])
    # print(len(dataMat))
    # print(dataMat[2][0])
    for i in range(1,N_i) :
        # temp = dataMat1[i][0]
        # #print(temp)
        # f2.write(temp)
        #f2.write(',')
        print(i+1)
        #for j in range (0,N_j):
        for k in range(0,len(dataMat)):
            for j in range(0, N_j):
                if dataMat[k][0] == dataMat1[0][j]:
                    # print(dataMat[k][0])
                    f2.write(dataMat1[i][j])
                    f2.write(',')

        f2.write(dataMat1[i][j])    #将标签写进去
        f2.write('\n')
        # if (i < N_i):
        #     f2.write('\n')

    f2.close()




def devide_into_test_and_train(fileDir,filename,percent):
    f = open(filename)
    lines = f.readlines()

    dataMat = []
    N = len(lines)
    for i in range(0, N):
        data = lines[i].strip().split(",")
        dataMat.append(data)

    N_i = len(dataMat)
    print(N_i)
    N_j = len(dataMat[0])
    print(N_j)
    Data_X = []
    Data_Y = []
    for i in range(1, N_i):
        Y = dataMat[i][N_j - 1]  # 标签
        if (Y == '1'):  # 正常
            Data_Y.append(1)
        else:
            Data_Y.append(2)

        for index in range(0, len(dataMat[i]) - 1):
            dataMat[i][index] = float(dataMat[i][index])

        lines1 = []
        for j in range(0, N_j - 1):
            lines1.append(float(dataMat[i][j]))
        Data_X.append(lines1)  # 要去掉最后一列label

    # print(len(Data_X[0]))
    # print(Data_X[0])
    # print(len(Data_Y))

    x_train, x_test, y_train, y_test = train_test_split(Data_X, Data_Y, test_size=percent)
    # ss = StratifiedShuffleSplit(n_splits=5, test_size=percent, train_size=1.0-percent, random_state=0)
    #
    # for train_index, test_index in ss.split(Data_X, Data_Y):
    #     print("TRAIN:", train_index, "TEST:", test_index)  # 获得索引值



    # print(x_train)
    #
    # print(y_train)

    #
    # print("********************")
    #
    # print(y_train[0])
    #

    #
    # if (os.path.exists(mkpath) != True):
    #     os.mkdir(mkpath)

    f1 = open(fileDir + '\\train.csv', 'w')
    f2 = open(fileDir+ '\\valid.csv', 'w')
    # 写入基因名

    for k in range(0, len(dataMat[0])):
        f1.write(dataMat[0][k])
        if k == len(dataMat[0]) - 1:
            f1.write('\n')
        else:
            f1.write(',')
    for k in range(0, len(dataMat[0])):
        f2.write(dataMat[0][k])
        if k == len(dataMat[0]) - 1:
            f2.write('\n')
        else:
            f2.write(',')

    for i in range(0, len(x_train)):
        for j in range(0, len(x_train[0])):
            f1.write(str(x_train[i][j]))  # 写入训练集数据
            f1.write(',')

        f1.write(str(y_train[i]))  # 对应写入标签
        f1.write('\n')

    for i in range(0, len(x_test)):
        for j in range(0, len(x_test[0])):
            f2.write(str(x_test[i][j]))  # 写入训练集数据
            f2.write(',')

        f2.write(str(y_test[i]))  # 对应写入标签
        f2.write('\n')

def devide_into_learn_and_test_label(fileDir,filename,percent):
    f = open(filename)
    lines = f.readlines()

    dataMat = []
    N = len(lines)
    for i in range(0, N):
        data = lines[i].strip().split(",")
        dataMat.append(data)

    N_i = len(dataMat)
    print(N_i)
    N_j = len(dataMat[0])
    print(N_j)
    Data_X = []
    Data_Y = []
    for i in range(1, N_i):
        Y = dataMat[i][N_j - 1]  # 标签
        if (Y == '1'):  # 正常
            Data_Y.append(1)
        else:
            Data_Y.append(2)

        for index in range(0, len(dataMat[i])-1):
            dataMat[i][index] = float(dataMat[i][index])

        lines1 = []
        for j in range(0, N_j - 1):
            lines1.append(float(dataMat[i][j]))
        Data_X.append(lines1)  # 要去掉最后一列label

    #x_train, x_test, y_train, y_test = train_test_split(Data_X, Data_Y, test_size=percent)
    ss = StratifiedShuffleSplit(n_splits=10, test_size=percent, train_size=1-percent, random_state=0)  #
    n=0
    for train_index, test_index in ss.split(Data_X, Data_Y):
        n=n+1
        print("TRAIN:", train_index, "TEST:", test_index)  # 获得索引值

        mkpath = fileDir + "\\split\\" + str(n) + "\\"

        if (os.path.exists(mkpath) != True):
            os.mkdir(mkpath)

        f1 = open(mkpath + 'learn.csv', 'w')
        f2 = open(mkpath + 'test.csv', 'w')
        # 写入基因名

        for k in range(0, len(dataMat[0])):
            f1.write(dataMat[0][k])
            if k == len(dataMat[0]) - 1:
                f1.write('\n')
            else:
                f1.write(',')
        for k in range(0, len(dataMat[0])):
            f2.write(dataMat[0][k])
            if k == len(dataMat[0]) - 1:
                f2.write('\n')
            else:
                f2.write(',')


        for num in train_index:
            for i in range(0,len(Data_X[num])):
                f1.write(str(Data_X[num][i]))
                f1.write(',')
            f1.write(str(Data_Y[num]))
            f1.write("\n")

        for num in test_index:
            for i in range(0, len(Data_X[num])):
                f2.write(str(Data_X[num][i]))
                f2.write(',')
            f2.write(str(Data_Y[num]))
            f2.write("\n")



        #
        # for i in range(0, len(Data_X)):
        #     for j in range(0, len(Data_X[0])):
        #         f1.write(str(Data_X))  # 写入训练集数据
        #         f1.write(',')
        #
        #     f1.write(str(y_train[i]))  # 对应写入标签
        #     f1.write('\n')
        #
        # for i in range(0, len(x_test)):
        #     for j in range(0, len(x_test[0])):
        #         f2.write(str(x_test[i][j]))  # 写入训练集数据
        #         f2.write(',')
        #
        #     f2.write(str(y_test[i]))  # 对应写入标签
        #     f2.write('\n')


###############将选出来的50个交叉特征分为学习集和独立测试集
fileDir = r"C:\Users\JQ\experiment\data\BREA"
filename = r"C:\Users\JQ\experiment\data\BREA\MI\MI_SE50_features_3.csv"
percent = 0.2
devide_into_learn_and_test_label(fileDir,filename,percent)

########再将学习集learn划分为 训练集和测试集

for i in range(1,11):
    fileDir1 = r"C:\Users\JQ\experiment\data\BREA\split"

    filename1 = fileDir1 + "\\" + str(i) + "\\learn.csv"
    fileDir2 =  fileDir1 + "\\" +str(i)
    percent = 0.4
    devide_into_test_and_train(fileDir2,filename1,percent)
#
#
# #将特征选择后的学习集分为训练集和验证集的顺序
# fileDir = r"C:\Users\JQ\experiment\data\KIRC\0\final"
# filename = fileDir +r"\best_features5.csv"
# devide_into_test_and_train(fileDir,filename)

##为最后的独立测试集进行特征筛选，得到最后的测试文件,注意将第一行特征名删除，修改recover函数，将i的起点改为1
# DATA_DIR = r"C:\Users\JQ\experiment\data\LAUD\0"
# DATA_DIR0 = r"C:\Users\JQ\experiment\data\LAUD\0\final"
# filename1 = DATA_DIR + r"\0_test_new.csv"     #原始数据
# filename = DATA_DIR0 +r"\jx50_features.txt"    #选出来的特征的基因名
# filename2 = DATA_DIR +r"\final_top50_test.csv"    #恢复后的文件
# recover_test(filename, filename1, filename2)
