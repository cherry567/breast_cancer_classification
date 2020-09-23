
import sklearn.feature_selection as fs
import pandas as pd
import numpy as np
def recover(filename,filename1,filename2):
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
    for i in range(0,N_i) :
        # temp = dataMat1[i][0]
        # #print(temp)
        # f2.write(temp)
        #f2.write(',')
        print(i+1)
        for k in range(0,len(dataMat)):
            for j in range(0, N_j):

                if dataMat[k][0] == dataMat1[0][j]:
                    #print(dataMat[k][0])
                    f2.write(dataMat1[i][j])
                    f2.write(',')

        f2.write(dataMat1[i][j])    #将标签写进去
        f2.write('\n')
        # if (i < N_i):
        #     f2.write('\n')

    f2.close()


def text_save(filename, data):#filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename,'w')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','\t') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")


data = pd.read_csv(r"C:\Users\JQ\experiment\data\BREA\MI\diffmRNAExp_snp_0.1_T_new.csv")  #feature.csv是Lasso降维后的特征集，得到的是LASSO降维后GBDT选出来的重要特征
#data = pd.read_csv(r"C:\Users\JQ\experiment\data\LAUD\0\final_JX_test.csv")
x_columns = []
for x in data.columns:
    if x not in ['label']:
        x_columns.append(x)
X = data[x_columns]
y = data['label']

features = fs.mutual_info_classif(X, y, discrete_features='auto', n_neighbors=3,
                                    copy=True, random_state=0)
data = pd.Series(data = features,
                index = x_columns)
data = data.sort_values(ascending = False)       #按值排序
print(data)

indexes = data[0:50].index              ####取前50个

#print(indexes)

################互信息不为0的：KIRC  172  ;LAUD  ;BREA 922################

MI_features = r"C:\Users\JQ\experiment\data\BREA\MI\MI_SE50_features_0.1.txt"
file = open(MI_features,'w')
text_save(MI_features,indexes)
#
DATA_DIR0 = r"C:\Users\JQ\experiment\data\BREA"
DATA_DIR = r"C:\Users\JQ\experiment\data\BREA\MI"
#DATA_DIR0 = r"C:\Users\JQ\experiment\data\LAUD\0\other"
filename1 = DATA_DIR0 + r"\snp_exp_1%.csv"          #原始数据
filename = DATA_DIR +r"\MI_SE50_features_0.1.txt"    #选出来的特征的基因名
filename2 = DATA_DIR +r"\MI_SE50_features_0.1.csv"   #恢复后的文件
recover(filename, filename1, filename2)