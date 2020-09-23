# -*- coding:utf-8 -*-
import os
import numpy as np
import add_label
# 第二步：文件转置

#
def T_name(filename2, fileData):

    filename1 = fileData
    f = open(filename1)

    temp = f.readlines()
    temp_1 = []

    for i in range(0, len(temp)):
        data = temp[i].strip().split('\t')
        temp_1.append(data)
    print(len(temp_1))
    print(len(temp_1[0]))
    le = len(temp_1)
    le1 = len(temp_1[0])

    ff = open(filename2, 'w')

    for i in range(0, len(temp_1[0])):
        print(i)
        for j in range(0, len(temp_1)):
            if(j < le-1):
                ff.write(temp_1[j][i])
                ff.write('\t')
            else:
                ff.write(temp_1[j][i])
        if i < le1 - 1:
            ff.write("\n")
    ff.close()

#
fileDir =r"C:\Users\JQ\experiment\data\BREA\compare"
fileData =  r"C:\Users\JQ\experiment\data\BREA\compare\diffmRNAExp_snp_0.1.txt"
filename = r"C:\Users\JQ\experiment\data\BREA\compare\diffmRNAExp_snp_0.1_T.txt"
T_name(filename,fileData)
filename1 = fileDir +"\\diffmRNAExp_snp_0.1_T_new.csv"
add_label.add(fileDir,filename,filename1)





# filedir = r"C:\Users\JQ\TCGA_TOOL\TCGA-KIRC\0\lncrna"
# filename = filedir + "/hnsc_mrna_train3T.txt"
#
# filename1 = filename
# f = open(filename1)
# temp = f.readlines()
# temp_1 = []
# for i in range(0, len(temp)):
#     data = temp[i].strip().split('\t')
#     temp_1.append(data)
# print(len(temp_1))
# print(len(temp_1[0]))
# le = len(temp_1)
# le1 = len(temp_1[0])
#     # filename2 = fileDir + "/prostate_mrna_tr0_T.txt"
#
# filename2 = filedir + "/hnsc_mrna_train3_T.txt"
#
# ff = open(filename2, 'w')
#
# for i in range(0, len(temp_1[0])):
#     for j in range(0, len(temp_1)):
#          if (j < le - 1):
#             ff.write(temp_1[j][i])
#             ff.write('\t')
#          else:
#             ff.write(temp_1[j][i])
#     if i < le1 - 1:
#         ff.write("\n")
# ff.close()
#




