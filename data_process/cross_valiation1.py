# -*- coding:utf-8 -*-
import os
import random

import convert


def devide_into_test_out(filedir, number):
    Data_X = []
    Data_Y = []

    for root, dirs, files in os.walk(filedir):
        print(files)

    for j in range(0, len(files)):
        temp = os.path.splitext(files[j])[0]
        filedirt = filedir + "/" + str(temp)
        if os.path.exists(filedirt != True):
            os.mkdir(filedirt)

        for i in range(0, number):
            fname = filedir + "/" + str(files[j])
            print(fname)
            loaddataset(filedirt, fname, i, Data_X, Data_Y)

def loaddataset(filedir, filename, count, Data_X, Data_Y):
    dataMl = []
    f = open(filename)
    lines = f.readlines()

    for i in range(1, len(lines)):
        data = lines[i].strip().split('\t')
        dataMl.append(data)
    # print(dataMl[1][0])
    normal = 0
    tumor = 0
    for i in range(0, len(dataMl)):
        Y = dataMl[i][0].strip().split('-')
        if Y[3][0] == '0' and Y[3][1] == '1':
            Data_Y.append(1)
            tumor += 1
        elif Y[3][0] == '1' and Y[3][1] == '1':
            Data_Y.append(2)
            normal += 1
        else:
            Data_Y.append(1)
            tumor += 1

    print("样本总数：" + str(normal + tumor))
    print("正常样本总数：" + str(normal))
    print("癌症样本总数：" + str(tumor))

    mkpath = filedir + "\\" + str(count) + "\\"
    if (os.path.exists(mkpath) != True):
        os.mkdir(mkpath)

    fname1 = mkpath + "/" + str(count) + "_test.txt"
    fname2 = mkpath + "/" + str(count) + "_train.txt"

    f1 = open(fname1, 'w')
    f2 = open(fname2, 'w')
    f1.write(lines[0])
    f2.write(lines[0])

    for i in range(0, len(lines)-1):
         if i % 10 == count:      #测试集   该方法无法保证选取数据的随机性
             f1.write(lines[i+1])
         else:                      #训练集
             f2.write(lines[i+1])
    f1.close()
    f2.close()

    f1_T = mkpath + "/" + str(count) + "_test_T.txt"
    f2_T = mkpath + "/" + str(count) + "_train_T.txt"
    convert.T_name(f2_T, fname2)  # 转置
    convert.T_name(f1_T, fname1)  # 转置

    # DATA_H = []
    # f3 = open(filename)
    # lines = f3.readlines()
    # N = len(lines)
    # datamat = []
    #
    # for i in range(0, N):
    #     data = lines[i].strip().split('\t')
    #     datamat.append(data)
    #
    # tumor = 0
    # normal = 0
    # for i in range(1, N):
    #     Y = datamat[i][0].strip().split('-')
    #
    #     if (Y[3][0] == '1' and Y[3][1] == '1'):
    #         DATA_H.append(datamat[1])
    #         normal += 1
    #     else:
    #         DATA_H.append(datamat[2])
    #         tumor += 1
    # #
    # print("样本总数：" + str(normal + tumor))
    # print("正常样本总数：" + str(normal))
    # print("癌症样本总数：" + str(tumor))
    # #
    # mkpath = filedir + "\\" + str(count) + "\\"
    # if (os.path.exists(mkpath) != True):
    #     os.mkdir(mkpath)

    # fname1 = mkpath + "/" + str(count) + "_vali.txt"
    # fname2 = mkpath + "/" + str(count) + "_train.txt"
    #
    # f1 = open(fname1, 'w')
    # f2 = open(fname2, 'w')
    # f1.write(lines[0])
    # f2.write(lines[0])
    #
    # filename1 = mkpath + "/" + str(count) + "_test.txt"
    # filename2 = mkpath + "/" + str(count) + "_train.txt"
    # f4 = open(filename1, 'w')
    # f5 = open(filename2, 'w')
    #
    # f4.write(lines[0])
    # f5.write(lines[0])
    #
    # split_normal = 0.9
    # split_tumor = 0.9
    # count_normal = 0
    # count_tumor = 0
    # count_num = 10
    #
    # for i in range(0, normal):
    #     if (random.random() < split_normal and count_normal < 1.5 * count_num):
    #         count_normal += 1
    #         f4.write(lines[i + 1])
    #     else:
    #         f5.write(lines[i + 1])
    #
    # for i in range(normal, normal + tumor):
    #     if (random.random() < split_tumor and count_tumor < 1.5 * count_num):
    #         count_tumor += 1
    #         f4.write(lines[i + 1])
    #     else:
    #         f5.write(lines[i + 1])
    #
    # f4.close()
    # f5.close()
    # f4_T = mkpath + "/" + str(count) + "_test_T.txt"
    # f5_T = mkpath + "/" + str(count) + "_train_T.txt"
    # convert.T_name(f4_T, filename1)  # 转置
    # convert.T_name(f5_T, filename2)  # 转置

def cross_count(filedir, filename, count, Data_X, Data_Y):
    for i in range(0, count):
        print(i)
        loaddataset(filedir, filename, i, Data_X, Data_Y)





