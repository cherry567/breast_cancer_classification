
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
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

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
# from sklearn.cross_validation import cross_val_score, ShuffleSplit
    RF_2 = RandomForestClassifier(n_estimators=301, max_depth=48, min_samples_leaf=2, min_samples_split=3)
    RF_2.fit(x_train, y_train)
    y_RF_2_pred = RF_2.predict(x_valid)
    y_RF_2_pred_t = RF_2.predict(x_test)
    balance3 = balance(y_RF_2_pred, y_valid, y_RF_2_pred_t, y_test)
    gbdt_b_acc.append(balance3)
    auc_RF2 = roc_auc_score(y_test, y_RF_2_pred_t)
    gbdt_auc.append(auc_RF2)
    RF2_confusion = confusion_matrix(y_test, y_RF_2_pred_t)
    # print("balance RF2:" + str(balance3))
    # print("AUC_RF2:" + str(auc_RF2))
    # print("RF2 confusion:")
    # metrics(RF2_confusion)
    acc, ses, spc, prc, f1 = metrics(RF2_confusion)
    gbdt_acc.append(acc)
    gbdt_ses.append(ses)
    gbdt_spc.append(spc)
    gbdt_prc.append(prc)
    gbdt_f1.append(f1)

ave_acc = np.mean(gbdt_acc)
ave_b_acc = np.mean(gbdt_b_acc)
ave_auc = np.mean(gbdt_auc)
ave_ses = np.mean(gbdt_ses)
ave_spc = np.mean(gbdt_spc)
ave_prc = np.mean(gbdt_prc)
ave_f1 = np.mean(gbdt_f1)
#
# print("acc:"+ str(ave_acc))
# print("b_acc:" + str(ave_b_acc))
# print("auc:" + str(ave_auc))
# print("ses:" + str(ave_ses))
# print("spc：" + str(ave_spc))
# print("prc: "+ str(ave_prc))
# print("f1:" + str(ave_f1))


# std_acc = np.std(gbdt_acc)
# std_b_acc= np.std(gbdt_b_acc)
# std_auc = np.std(gbdt_auc)
# std_ses = np.std(gbdt_ses)
# std_spc =np.std(gbdt_spc)
# std_prc = np.std(gbdt_prc)
# std_f1= np.std(gbdt_f1)


std_acc = np.var(gbdt_acc)
std_b_acc= np.var(gbdt_b_acc)
std_auc = np.var(gbdt_auc)
std_ses = np.var(gbdt_ses)
std_spc =np.var(gbdt_spc)
std_prc = np.var(gbdt_prc)
std_f1= np.var(gbdt_f1)
# print("acc:"+ str(ave_acc))
# print("b_acc:" + str(ave_b_acc))
# print("auc:" + str(ave_auc))
# print("ses:" + str(ave_ses))
# print("spc：" + str(ave_spc))
# print("prc: "+ str(ave_prc))
# print("f1:" + str(ave_f1))

print("acc:"+ str(std_acc))
print("b_acc:" + str(std_b_acc))
print("auc:" + str(std_auc))
print("ses:" + str(std_ses))
print("spc：" + str(std_spc))
print("prc: "+ str(std_prc))
print("f1:" + str(std_f1))