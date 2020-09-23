import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, LeaveOneOut, KFold, StratifiedKFold
from sklearn import preprocessing
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score,confusion_matrix,roc_curve,auc
import matplotlib.pyplot as plt
# # # #
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
    # print(dataMat[0])

    f2 = open(filename2, 'w')
    #f2.write(lines[0])             #将基因名写入f2

    for j in range(0, len(lines1)):        #train.csv
        data1 = lines1[j].strip().split(',')
        dataMat1.append(data1)

    #print(dataMat1[0][1])

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


#
########################将训练集、测试集、验证集 按照GBDT——feature 做特征选择
# fileDir = r"C:\Users\JQ\experiment\data\BREA"
# filepath = fileDir + "\learn_test"
# filename = fileDir +'\\GBDT_R_features.txt'
# filename1 = filepath +'\\train.csv'
# filename2 = filepath +'\\final_R_train.csv'
# filename3 = filepath +'\\valid.csv'
# filename4 = filepath +'\\final_R_valid.csv'
# #
# Dir =  r"C:\Users\JQ\experiment\data\BREA\learn_test"
# file_test  =  Dir +"\\test.csv"
# filename_test = Dir +"\\final_R_test.csv"
# #
# recover(filename,file_test,filename_test)    #534个样本恢复
# recover(filename,filename1,filename2)
# recover(filename,filename3,filename4)


def balance(y_pred,y_valid,y_pred_t,y_test):
    tumor = 0
    normal = 0
    tumor_sum = 0
    normal_sum = 0
    for i in range(0, len(y_valid)):
        if y_valid[i] == 1:
            tumor_sum += 1
        else:
            normal_sum += 1

    for i in range(0, len(y_pred)):

        if y_valid[i] == 1 & y_pred[i] == y_valid[i]:
            tumor += 1
        if y_valid[i] == 2 & y_pred[i] == y_valid[i]:
            normal += 1
    tumor_ratio = tumor / tumor_sum  #####癌症样本准确率
    normal_ratio = normal / normal_sum  ###正常样本准确率

    tumor_t = 0
    normal_t = 0
    tumor_t_sum = 0
    normal_t_sum = 0
    for i in range(0, len(y_test)):
        if y_test[i] == 1:
            tumor_t_sum += 1
        else:
            normal_t_sum += 1

    for i in range(0, len(y_pred_t)):

        if y_test[i] == 1 & y_pred_t[i] == y_test[i]:
            tumor_t += 1
        if y_test[i] == 2 & y_pred_t[i] == y_test[i]:
            normal_t += 1
    tumor_t_ratio = tumor_t / tumor_t_sum  #####癌症样本准确率
    normal_t_ratio = normal_t / normal_t_sum  ###正常样本准确率

    balance_acc_test = (tumor_t_ratio * tumor_ratio + normal_t_ratio * normal_ratio) /2

    return balance_acc_test

def metrics(confusion):
    acc=  (confusion[0][0]+confusion[1][1])/(confusion[0][0]+confusion[0][1]+confusion[1][0]+confusion[1][1])
    ses = confusion[0][0] /(confusion[0][0]+confusion[0][1])
    spc = confusion[1][1]/(confusion[1][1]+confusion[1][0])
    prc = confusion[0][0]/(confusion[0][0]+confusion[1][0])
    f1 = 2*confusion[0][0]/(2*confusion[0][0]+confusion[1][0]+confusion[0][1])

    # print("acc:"+str(acc))
    # print("sec:" +str(ses))
    # print("spc:" +str(spc))
    # print("prc:" +str(prc))
    # print("f1:"+ str(f1))
    return acc,ses,spc,prc,f1


gbdt_acc = []
gbdt_b_acc =[]
gbdt_ses =[]
gbdt_spc=[]
gbdt_prc=[]
gbdt_f1 =[]
gbdt_auc =[]


for i in range(1,11):
    filename = r"C:\Users\JQ\experiment\data\BREA\split"
    data = pd.read_csv(filename + "\\" + str(i) + "\\train.csv")  # 全体特征
    x_columns = []
    for x in data.columns:
        if x not in ['label']:
            x_columns.append(x)
    X = data[x_columns]
    y = data['label']
    x_train = data[x_columns]
    y_train = data['label']

    data = pd.read_csv(filename + "\\" + str(i) + "\\valid.csv")
    x_columns = []
    for x in data.columns:
        if x not in ['label']:
            x_columns.append(x)
    x_valid = data[x_columns]
    y_valid = data['label']

    data = pd.read_csv(filename + "\\" + str(i) + "\\test.csv")
    x_columns = []
    for x in data.columns:
        if x not in ['label']:
            x_columns.append(x)
    x_test = data[x_columns]
    y_test = data['label']

    # from sklearn import svm
    # clf_svm = svm.SVC(kernel='linear')
    # clf_svm.fit(x_train,y_train)
    # y_svm_pred = clf_svm.predict(x_valid)
    # y_svm_pred_t = clf_svm.predict(x_test)
    # balance1 = balance(y_svm_pred,y_valid,y_svm_pred_t,y_test)
    # auc_svm = roc_auc_score(y_test,y_svm_pred_t)
    # svm_confusion = confusion_matrix(y_test,y_svm_pred_t)
    # print("balance SVM:" + str(balance1))
    # print("AUC_SVM:"+str(auc_svm))
    # print("SVM confusion:")
    # metrics(svm_confusion)
    #
    # from sklearn.ensemble import RandomForestClassifier
    #
    # RF = RandomForestClassifier(n_estimators=20)
    # RF.fit(x_train,y_train)
    # y_RF_pred = RF.predict(x_valid)
    # y_RF_pred_t = RF.predict(x_test)
    # balance2 = balance(y_RF_pred,y_valid,y_RF_pred_t,y_test)
    # auc_RF = roc_auc_score(y_test,y_RF_pred_t)
    # RF_confusion = confusion_matrix(y_test,y_RF_pred_t)
    # print("balance RF:" + str(balance2))
    # print("AUC_RF:"+str(auc_RF))
    # print("RF confusion:")
    # metrics(RF_confusion)
    #
    # from sklearn.ensemble import RandomForestClassifier
    #
    # # from sklearn.cross_validation import cross_val_score, ShuffleSplit
    # RF_2 = RandomForestClassifier(n_estimators=1000)
    # RF_2.fit(x_train, y_train)
    # y_RF_2_pred =RF_2.predict(x_valid)
    # y_RF_2_pred_t = RF_2.predict(x_test)
    # balance3 = balance(y_RF_2_pred,y_valid,y_RF_2_pred_t,y_test)
    # auc_RF2 =roc_auc_score(y_test,y_RF_2_pred_t)
    # RF2_confusion = confusion_matrix(y_test,y_RF_2_pred_t)
    # print("balance RF2:" + str(balance3))
    # print("AUC_RF2:"+str(auc_RF2))
    # print("RF2 confusion:")
    # metrics(RF2_confusion)
    #
    # from sklearn import tree
    #
    # clf_tree = tree.DecisionTreeClassifier(criterion='entropy')
    # clf_tree = clf_tree.fit(x_train, y_train)
    # y_TREE_pred = clf_tree.predict(x_valid)
    # y_TREE_pred_t = clf_tree.predict(x_test)
    # balance4 = balance(y_TREE_pred,y_valid,y_TREE_pred_t,y_test)
    # auc_tree =roc_auc_score(y_test,y_TREE_pred_t)
    # tree_confusion = confusion_matrix(y_test,y_TREE_pred_t)
    # print("balance TREE:" + str(balance4))
    # print("AUC_tree:"+str(auc_tree))
    # print("TREE confusion:")
    # metrics(tree_confusion)
    #
    # # from sklearn import tree
    # #
    # # clf_tree1 = tree.DecisionTreeClassifier()   #默认基尼系数
    # # clf_tree1 = clf_tree.fit(x_train, y_train)
    #
    # from sklearn import neighbors
    #
    # clf_knn = neighbors.KNeighborsClassifier()
    # clf_knn.fit(x_train,y_train)
    # y_knn_pred = clf_knn.predict(x_valid)
    # y_knn_pred_t = clf_knn.predict(x_test)
    # balance5 = balance(y_knn_pred,y_valid,y_knn_pred_t,y_test)
    # auc_knn =roc_auc_score(y_test,y_knn_pred_t)
    # knn_confusion = confusion_matrix(y_test,y_knn_pred_t)
    # print("balance KNN:" + str(balance5))
    # print("AUC_knn:"+str(auc_knn))
    # print("knn confusion:")
    # metrics(knn_confusion)

    from sklearn.ensemble import GradientBoostingClassifier

    gbr = GradientBoostingClassifier(
    #  n_estimators=74,
    #     max_depth=19,
    #     min_samples_leaf=36,
    #     min_samples_split=67,
    #     learning_rate=0.33653933039595074,  #####0.8165,
    #     subsample=0.96259334356605,
    #     random_state=10
    # )

    # n_estimators=127,
    #     max_depth=9,
    #     min_samples_leaf=17,
    #     min_samples_split=20,
    #     learning_rate= 0.1889,  #####0.8165,
    #     subsample=0.8139,
    #     random_state=10
    #     )
    ########11111111111111111111111111111111111111BO##########################################
    # n_estimators = 235,
    # max_depth = 19,
    # min_samples_leaf = 46,
    # min_samples_split = 58,
    # learning_rate =  0.04838582279938517,  #####0.8165,
    # subsample = 0.9891359053949287,
    # random_state = 10
    # )
    ###########11111111111111111111111RD##########################
    # n_estimators = 271,
    # max_depth = 23,
    # min_samples_leaf = 53,
    # min_samples_split = 55,
    # learning_rate =  0.1594398172778165,  #####0.8165,
    # subsample = 0.8231493992953516,
    # random_state = 10
    # )
    ##########222222222222222222BO##################最优
    n_estimators = 374,
    max_depth = 29,
    min_samples_leaf = 88,
    min_samples_split = 12,
    learning_rate =0.5373220899834616,  #####0.8165,
    subsample = 0.846203750084136,
    random_state = 10
    )
    ###############2222222222222RD
    # n_estimators = 67,
    # max_depth = 20,
    # min_samples_leaf = 93,
    # min_samples_split = 35,
    # learning_rate = 0.030286120485679326,  #####0.8165,
    # subsample =  0.8328799036070788,
    # random_state = 10
    # )
    #############################333333333333333BO
    # n_estimators = 358,
    # max_depth = 23,
    # min_samples_leaf = 52,
    # min_samples_split = 70,
    # learning_rate = 0.2797941215670739,  #####0.8165,
    # subsample = 0.9321368111903375,
    # random_state = 10
    # )
        ##################################333333RD
    #     n_estimators=82,
    #     max_depth=42,
    #     min_samples_leaf=56,
    #     min_samples_split=64,
    #     learning_rate=0.14537979952691704,  #####0.8165,
    #     subsample=0.9760518597431092,
    #     random_state=10
    # )

#####################444444444BO
        # n_estimators=412,
        #     max_depth=9,
        #     min_samples_leaf=60,
        #     min_samples_split=72,
        #     learning_rate=0.20479554131615968,  #####0.8165,
        #     subsample=0.9958775857920726,
        #     random_state=10
        # )
######################44444444RD
#             n_estimators = 416,
#             max_depth = 12,
#             min_samples_leaf = 32,
#             min_samples_split = 44,
#             learning_rate = 0.27828778773713747,  #####0.8165,
#             subsample = 0.9480390573398242,
#             random_state = 10
#         )

#############################5555555RD
        #     n_estimators = 237,
        #     max_depth = 34,
        #     min_samples_leaf = 86,
        #     min_samples_split = 22,
        #     learning_rate =0.6443744115487993,  #####0.8165,
        #     subsample = 0.9105421033225315,
        #     random_state = 10
        # )
#########################555555BO
        #     n_estimators = 474,
        #     max_depth = 28,
        #     min_samples_leaf = 87,
        #     min_samples_split = 89,
        #     learning_rate =0.027136389286726015,  #####0.8165,
        #     subsample = 0.8719086882161583,
        #     random_state = 10
        # )
####################666666RD    最优
        #     n_estimators = 130,
        #     max_depth = 23,
        #     min_samples_leaf = 94,
        #     min_samples_split = 54,
        #     learning_rate =0.0829094998990422,  #####0.8165,
        #     subsample =  0.8176170806599637,
        #     random_state = 10
        # )
#####################666666BO
    # n_estimators = 302,
    # max_depth = 28,
    # min_samples_leaf = 31,
    # min_samples_split = 61,
    # learning_rate = 0.04164996932068887,  #####0.8165,
    # subsample = 0.9028945206061637,
    # random_state = 10
    # )
#########################77777RD
    # n_estimators = 153,
    # max_depth = 29,
    # min_samples_leaf = 22,
    # min_samples_split = 10,
    # learning_rate =  0.03722959187506328,  #####0.8165,
    # subsample =  0.8825137316466605,
    # random_state = 10
    # )
######################777777BO
    #     n_estimators=122,
    #     max_depth=44,
    #     min_samples_leaf=73,
    #     min_samples_split=41,
    #     learning_rate=0.07692109714557302,  #####0.8165,
    #     subsample=0.8825137316466605,
    #     random_state=10
    # )
    # ########################8888RD
    #     n_estimators=414,
    #     max_depth=7,
    #     min_samples_leaf=19,
    #     min_samples_split=77,
    #     learning_rate=0.2860336309242806,  #####0.8165,
    #     subsample=0.9933219217119194,
    #     random_state=10
    # )
    ##########################8888BO
    #      n_estimators=238,
    #     max_depth=16,
    #     min_samples_leaf=70,
    #     min_samples_split=84,
    #     learning_rate=0.6954629752796418,  #####0.8165,
    #     subsample=0.999219920154008,
    #     random_state=10
    # )

    # ####################999999RD
    #          n_estimators=254,
    #         max_depth=7,
    #         min_samples_leaf=78,
    #         min_samples_split=13,
    #         learning_rate=0.0760937193688753,  #####0.8165,
    #         subsample=0.9636631827836868,
    #         random_state=10
    #     )

    # ####################99999BO
    #      n_estimators=375,
    #     max_depth=14,
    #     min_samples_leaf=72,
    #     min_samples_split=26,
    #     learning_rate=0.04850690819245295,  #####0.8165,
    #     subsample= 0.9077637360771138,
    #     random_state=10
    # )
###########################10 RD
    #     n_estimators=316,
    #     max_depth=29,
    #     min_samples_leaf=94,
    #     min_samples_split=19,
    #     learning_rate=0.04850690819245295,  #####0.8165,
    #     subsample=0.9588579405215747,
    #     random_state=10
    # )


    ###########################10  BO
    #     n_estimators=196,
    #     max_depth=36,
    #     min_samples_leaf=91,
    #     min_samples_split=15,
    #     learning_rate=0.013129646271213751,  #####0.8165,
    #     subsample= 0.9050781844584811,
    #     random_state=10
    # )


    clf_gbr = gbr.fit(x_train, y_train)
    y_gbr_pred = clf_gbr.predict(x_valid)
    y_gbr_pred_t = clf_gbr.predict(x_test)
    balance6 = balance(y_gbr_pred, y_valid, y_gbr_pred_t, y_test)
    gbdt_b_acc.append(balance6)
    auc_gbr = roc_auc_score(y_test, y_gbr_pred_t)
    gbdt_auc.append(auc_gbr)
    gbr_confusion = confusion_matrix(y_test, y_gbr_pred_t)

    # print("balance GBDT:" + str(balance6))
    # print("AUC_gbr:" + str(auc_gbr))
    # print("gbr confusion:")
    acc,ses,spc,prc,f1 = metrics(gbr_confusion)
    gbdt_acc.append(acc)
    gbdt_ses.append(ses)
    gbdt_spc.append(spc)
    gbdt_prc.append(prc)
    gbdt_f1.append(f1)

ave_acc = np.mean(gbdt_acc)
ave_b_acc= np.mean(gbdt_b_acc)
ave_auc = np.mean(gbdt_auc)
ave_ses = np.mean(gbdt_ses)
ave_spc =np.mean(gbdt_spc)
ave_prc = np.mean(gbdt_prc)
ave_f1= np.mean(gbdt_f1)

# std_acc = np.std(gbdt_acc)
# std_b_acc= np.std(gbdt_b_acc)
# std_auc = np.std(gbdt_auc)
# std_ses = np.std(gbdt_ses)
# std_spc =np.std(gbdt_spc)
# std_prc = np.std(gbdt_prc)
# std_f1= np.std(gbdt_f1)


# std_acc = np.var(gbdt_acc)
# std_b_acc= np.var(gbdt_b_acc)
# std_auc = np.var(gbdt_auc)
# std_ses = np.var(gbdt_ses)
# std_spc =np.var(gbdt_spc)
# std_prc = np.var(gbdt_prc)
# std_f1= np.var(gbdt_f1)

print("acc:"+ str(ave_acc))
print("b_acc:" + str(ave_b_acc))
print("auc:" + str(ave_auc))
print("ses:" + str(ave_ses))
print("spc：" + str(ave_spc))
print("prc: "+ str(ave_prc))
print("f1:" + str(ave_f1))

# print("acc:"+ str(std_acc))
# print("b_acc:" + str(std_b_acc))
# print("auc:" + str(std_auc))
# print("ses:" + str(std_ses))
# print("spc：" + str(std_spc))
# print("prc: "+ str(std_prc))
# print("f1:" + str(std_f1))






