# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 16:09:18 2018

@author: YHC1WX
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from ggplot import *
# 读取数据集
purchase = pd.read_csv('C:/Users/YHC1WX/Desktop/Yu Chenmin/Yuchenmin/Python/Logistic/Social_Network_Ads.csv')
# 查看数据类型
purchase.dtypes
# 查看各变量的缺失情况
purchase.isnull().sum()

# 对Gender变量作哑变量处理
dummy = pd.get_dummies(purchase.Gender)
# 为防止多重共线性，将哑变量中的Female删除
dummy_drop = dummy.drop('Female', axis = 1)

# 剔除用户ID和Gender变量
purchase = purchase.drop(['User ID','Gender'], axis = 1)
# 如果调用Logit类，需要给原数据集添加截距项
purchase['Intercept'] = 1

# 哑变量和原数据集合并
model_data = pd.concat([dummy_drop,purchase], axis = 1)
model_data

# 将数据集拆分为训练集和测试集
X = model_data.drop('Purchased', axis = 1)
y = model_data['Purchased']
# 训练集与测试集的比例为75%和25%
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75, random_state=0)

# 根据训练集构建Logistic分类器
logistic = smf.Logit(y_train,X_train).fit()
logistic.summary2()
# 重新构造分类器
logistic2 = smf.logit('Purchased~Age+EstimatedSalary', 
                      data = pd.concat([y_train,X_train], axis = 1)).fit()
logistic2.summary2()

print(logistic.aic)
print(logistic2.aic)

# 优势比
np.exp(logistic2.params)

# 根据分类器，在测试集上预测概率
prob = logistic2.predict(exog = X_test.drop('Male', axis = 1))
# 根据概率值，将观测进行分类，不妨以0.5作为阈值
pred = np.where(prob >= 0.5, 1, 0)

# 根据预测值和实际值构建混淆矩阵
cm = metrics.confusion_matrix(y_test, pred, labels=[0,1])
cm
# 计算模型的准确率
accuracy = cm.diagonal().sum()/cm.sum()
accuracy

# 绘制ROC曲线
fpr, tpr, _ = metrics.roc_curve(y_test, pred)
df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))

ggplot(df, aes(x='fpr', y='tpr')) +\
    geom_area(alpha=0.5, fill = 'steelblue') +\
    geom_line() +\
    geom_abline(linetype='dashed',color = 'red') +\
    labs(x = '1-specificity', y = 'Sensitivity',title = 'ROC Curve AUC=%.3f' % metrics.auc(fpr,tpr))
