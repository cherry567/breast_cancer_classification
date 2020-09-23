import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import recover_data_new
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, train_test_split
import time

import matplotlib.pylab as plt


def loadDataset(filename, train_X=[], train_Y=[]):
    # filename是划分训练集和测试集文件夹中的训练集train.txt
    f = open(filename)
    lines = f.readlines()

    dataMat = []
    N = len(lines)
    for i in range(0, N):
        data = lines[i].strip().split(",")
        dataMat.append(data)
    N_i = len(dataMat)
    N_j = len(dataMat[0])

    for i in range(0, N_i):
        Y = dataMat[i][N_j-1]  #标签
        if (Y == '1'):  # 正常
            train_Y.append(1)
        else:
            train_Y.append(2)

        for index in range(0, len(dataMat[i])):
            dataMat[i][index] = float(dataMat[i][index])

        lines1=[]
        for j in range(0,N_j-1):
            lines1.append(float(dataMat[i][j]))
        train_X.append(lines1)   #要去掉最后一列label



def recover(filename,filename1,filename2):
    f = open(filename)
    lines = f.readlines()    #top60特征
    #print(lines)

    # print(len(lines))
    f1 = open(filename1)
    lines1 = f1.readlines()  #原始数据，文件为feature.csv
    #print(lines1)
    dataMat = []      #存放基因名
    dataMat1 = []
    for i in range(0, len(lines)):
        data = lines[i].strip().split('\t')

        dataMat.append(data)
    print(dataMat[0])

    f2 = open(filename2, 'w')
    #f2.write(lines[0])             #将基因名写入f2

    for j in range(0, len(lines1)):        #train.csv
        data1 = lines1[j].strip().split(',')
        dataMat1.append(data1)

    print(dataMat1[0][1])

    N_i = len(dataMat1)   #行数
    N_j = len(dataMat1[0]) #列数

    for i in range(0,N_i) :
        #temp = dataMat1[i][0]
        # #print(temp)
        #print(i+1)
        for k in range(0,len(dataMat)):
            for j in range(0, N_j):
                if dataMat[k][0] == dataMat1[0][j]:
                    #print(dataMat[k][0])
                    f2.write(dataMat1[i][j])
                    f2.write(',')
        f2.write(dataMat1[i][N_j-1])    #将标签写进去
        f2.write('\n')
    f2.close()


# x_train, x_test, y_train, y_test = train_test_split(X, y,test_size=0.1,random_state=0)

def text_save(filename, data):#filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename,'w')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','\t') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")


# #data = pd.read_csv(r"C:\Users\JQ\experiment\data\KIRC\learn_test\hnsc_mrna_train0.csv")
data = pd.read_csv(r"C:\Users\JQ\experiment\data\BREA\split\1\train.csv")
x_columns = []
for x in data.columns:
    if x not in ['label']:
        x_columns.append(x)
x_train = data[x_columns]
y_train = data['label']

data = pd.read_csv(r"C:\Users\JQ\experiment\data\BREA\split\1\valid.csv")
x_columns = []
for x in data.columns:
    if x not in ['label']:
        x_columns.append(x)
x_valid = data[x_columns]
y_valid = data['label']
#
data = pd.read_csv(r"C:\Users\JQ\experiment\data\BREA\split\1\test.csv")
x_columns = []
for x in data.columns:
    if x not in ['label']:
        x_columns.append(x)
x_test= data[x_columns]
y_test = data['label']

#########BREA参数：
starttime = time.time()
gbr = GradientBoostingClassifier(
####################################0.5参数############################################
#    n_estimators=265,
#     max_depth=11,
#    min_samples_leaf=47,
#     min_samples_split=18,
#     learning_rate=0.14314973262456368,                 #####0.8165,
#     subsample= 0.9134787899113554,
# random_state=10
# )

###############################1.0最优参数########################################################
#    n_estimators=234,
#     max_depth=9,
#    min_samples_leaf=49,
#     min_samples_split=49,
#     learning_rate=0.1,                 #####0.8165,
#     subsample= 0.9833,
# random_state=10
# )

    n_estimators=74,
    max_depth=19,
    min_samples_leaf=36,
    min_samples_split=67,
    learning_rate= 0.33653933039595074,  #####0.8165,
    subsample=0.96259334356605,
    random_state=10
    )

######################################1.5最优参数#####################################################
# n_estimators=500,
#     max_depth=49,
#    min_samples_leaf=24,
#     min_samples_split=10,
#     learning_rate=0.2806,                 #####0.8165,
#     subsample= 0.9370,
# random_state=10
# )


# ######################################2最优参数#####################################################
# n_estimators=112,
#     max_depth=41,
#    min_samples_leaf=79,
#     min_samples_split=90,
#     learning_rate=0.8440,                 #####0.8165,
#     subsample= 0.9254,
# random_state=10
# )
# ######################################2.5最优参数#####################################################
# n_estimators=497,
#     max_depth=3,
#    min_samples_leaf=45,
#     min_samples_split=3,
#     learning_rate=0.2614,                 #####0.8165,
#     subsample= 0.9439,
# random_state=10
# )

# ######################################3最优参数#####################################################
# n_estimators=226,
#     max_depth=43,
#    min_samples_leaf=78,
#     min_samples_split=79,
#     learning_rate=0.1254,                 #####0.8165,
#     subsample= 0.8803,
# random_state=10
# )

gbr.fit(x_train,y_train)
acc_gbr_valid = gbr.score(x_valid,y_valid)
y_pred = gbr.predict(x_valid)
tumor = 0
normal = 0
tumor_sum =0
normal_sum =0
for i in range(0,len(y_valid)):
    if y_valid[i] ==1:
        tumor_sum +=1
    else:
        normal_sum +=1

for i in range(0,len(y_pred)):

    if y_valid[i]==1 & y_pred[i] == y_valid[i]:
            tumor+=1
    if y_valid[i]==2 & y_pred[i] == y_valid[i]:
            normal +=1
tumor_ratio = tumor/tumor_sum   #####癌症样本准确率
normal_ratio = normal/normal_sum   ###正常样本准确率
# print(tumor_ratio)
# print(normal_ratio)
acc_gbr_test = gbr.score(x_test,y_test)
y_pred_t = gbr.predict(x_test)

tumor_t = 0
normal_t = 0
tumor_t_sum =0
normal_t_sum =0
for i in range(0,len(y_test)):
    if y_test[i] ==1:
        tumor_t_sum +=1
    else:
        normal_t_sum +=1

for i in range(0,len(y_pred_t)):

    if y_test[i]==1 & y_pred_t[i] == y_test[i]:
            tumor_t +=1
    if y_test[i]==2 & y_pred_t[i] == y_test[i]:
            normal_t +=1
tumor_t_ratio = tumor_t/tumor_t_sum   #####癌症样本准确率
normal_t_ratio = normal_t/normal_t_sum   ###正常样本准确率

balance_acc_test = (tumor_t_ratio*tumor_ratio + normal_t_ratio*normal_ratio)/2

endtime = time.time()

print("valid:", str(acc_gbr_valid))
print("test: ",str(acc_gbr_test))
print("balanced:", str(balance_acc_test))
print("time:",str(endtime-starttime))

#
# ######重要特征可视化##################
# importance = gbr.feature_importances_
# #
# x_columns_new=[]
# for i in range(0,len(x_columns)):
#     ss = x_columns[i].strip().split("|")
#     x_columns_new.append(ss[0])
#
# Impt_Series_new = pd.Series(importance, index = x_columns_new)  ##########top30
#
# feat_imp = Impt_Series_new.sort_values(ascending=False)
#
# feat_imp_30 = feat_imp[0:30]
#
# print(feat_imp_30)
#
# feat_imp_30.plot(kind='bar', title='Feature Importances',color ='r')
#
# plt.ylabel('Feature Importance Score')
# plt.savefig('features.pdf')
# plt.show()
#


# # # #
# # # # #################################重要特征存入文件
# importance = gbr.feature_importances_
# Impt_Series = pd.Series(importance, index = x_columns)
# print(Impt_Series)
# index = Impt_Series.index
# Impt_Series = Impt_Series.sort_values(ascending = False)   #排序
# print(Impt_Series)
# print("*****")
# result = np.array(Impt_Series[0:50].index)    #最佳为15
# print(result)
#
# print(Impt_Series.index)
#
# feature_gbdt = r"C:\Users\JQ\experiment\data\BREA\GBDT_R_features.txt"
# #
# # #将result写入文件保存
# file = open(feature_gbdt,'w')
#
# text_save(feature_gbdt,result)
#
# #恢复数据，第一个文件为原始文件，第二个为选出的特征名，第三个为保存的文件名
# file0 = r"C:\Users\JQ\experiment\data\BREA\merge_50_features.csv"
# file1 = r"C:\Users\JQ\experiment\data\BREA\GBDT_R_features.csv"
# recover(feature_gbdt,file0,file1)

