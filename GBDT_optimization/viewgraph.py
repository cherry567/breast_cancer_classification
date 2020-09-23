import csv
from random import sample
import os
import pandas as pd
import numpy as np
# import plotly as plt
import matplotlib.pylab as plt
from boto import sns
from hyperopt import fmin,hp,tpe,Trials,STATUS_OK
from sklearn.ensemble import GradientBoostingClassifier , RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize, scale
from timeit import default_timer as timer

#################################贝叶斯优化######################################
data = pd.read_csv(r"C:\Users\JQ\experiment\data\BREA\split\10\train.csv")     #全体特征
x_columns = []
for x in data.columns:
    if x not in ['label']:
        x_columns.append(x)
X = data[x_columns]
y = data['label']
x_train = data[x_columns]
y_train = data['label']

data = pd.read_csv(r"C:\Users\JQ\experiment\data\BREA\split\10\valid.csv")   #feature.csv是Lasso降维后的特征集，得到的是LASSO降维后GBDT选出来的重要特征
#data = pd.read_csv(r"C:\Users\JQ\experiment\data\LAUD\0\final_JX_test.csv")
x_columns = []
for x in data.columns:
    if x not in ['label']:
        x_columns.append(x)
x_valid = data[x_columns]
y_valid = data['label']

data = pd.read_csv(r"C:\Users\JQ\experiment\data\BREA\split\10\test.csv")   #feature.csv是Lasso降维后的特征集，得到的是LASSO降维后GBDT选出来的重要特征
#data = pd.read_csv(r"C:\Users\JQ\experiment\data\LAUD\0\final_JX_test.csv")
x_columns = []
for x in data.columns:
    if x not in ['label']:
        x_columns.append(x)
x_test = data[x_columns]
y_test = data['label']

X = x_train
y = y_train


def hyperopt_train_test(params):

    clf = GradientBoostingClassifier(**params)
    return cross_val_score(clf, X, y,cv=5).mean()


space4rf = {
        'n_estimators': hp.choice('n_estimators',range(10, 500)),
        'min_samples_split': hp.choice('min_samples_split',range(2,100)),
         'min_samples_leaf':hp.choice('min_samples_leaf',range(2,100)),
         'subsample':hp.uniform('subsample',0.8,1),
         'learning_rate':hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
        'max_depth':hp.choice('max_depth',range(3,50)),
}

best = 0
parameters = ['learning_rate','max_depth','min_samples_leaf','min_samples_split','n_estimators','subsample']

acc_data =[]
para_data = []

def f(params):
    global best
    acc = hyperopt_train_test(params)
    # if acc > best:
    best = acc
    acc_data.append(best)
    para_data.append(params)
    print('new best:', best, params)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()

best = fmin(f, space4rf, algo=tpe.suggest, max_evals=300, trials=trials)

print('best:')
print(best)
clf = GradientBoostingClassifier(**best)
acc = cross_val_score(clf, X, y).mean()
clf.fit(x_train,y_train)
acc_test = clf.score(x_test,y_test)
print("acc: ",str(acc))
print("acc_test:",str(acc_test))
best_value = []
line = str(best).strip().split(', ')
# best_data = ['0.33653933039595074','19','36','67','74','0.96259334356605']
best_data = []

for i in range(0,len(line)):
    # print(best_value[i])
    data = str(line[i]).strip().split(': ')
    data1= data[1].replace("}",'')
    best_data.append(data1)

print(best_data)

#########'learning_rate': 0.41180807939925523, 'max_depth': 40, 'max_features': 76, 'min_samples_leaf': 116,
# 'min_samples_split': 152, 'n_estimators': 184, 'subsample': 0.9738745773244893
#
clos =len(parameters)
f, axes = plt.subplots(nrows=1, ncols=clos, figsize=(18,10))
cmap = plt.cm.jet

for i, val in enumerate(parameters):
    xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
    ys = [-t['result']['loss'] for t in trials.trials]
    xs, ys = zip(*sorted(zip(xs, ys)))
    ys = np.array(ys)
    x = float(best_data[i])
    y = float(acc)
    axes[i].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.5, c=cmap(float(i)/len(parameters)))
    axes[i].annotate(s='*', xytext=(x , y),size = 30 ,xy=(x, y),weight="bold",color ='r')
    axes[i].set_title(val)
    axes[i].set_ylim([0.92,1.0])

plt.savefig('BO_10.pdf') #指定分辨
plt.show()















