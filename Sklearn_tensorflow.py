'''
波士顿房价问题,GBDT调参过程
'''
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
import warnings


def ignore_warn(*args, **kwargs):
    pass


warnings.warn = ignore_warn
# 波士顿房价数据
boston = load_boston()
x = boston.data
y = boston.target
# print('波士顿数据:',x.shape)       # (506,13)
# print(x[::100])
# print('波士顿房价:',y.shape)       # (506,)
# print(y[::100])

# 随机挑选
train_x_disorder, test_x_disorder, train_y_disorder, test_y_disorder =\
    train_test_split(x, y, train_size=0.8, random_state=33)
# print(len(train_x_disorder))    # 404
# print(train_x_disorder)
# 数据标准化

ss_x = preprocessing.StandardScaler()
train_x_disorder = ss_x.fit_transform(train_x_disorder)
test_x_disorder = ss_x.transform(test_x_disorder)

ss_y = preprocessing.StandardScaler()
train_y_disorder = ss_y.fit_transform(train_y_disorder.reshape(-1, 1))
test_y_disorder = ss_y.transform(test_y_disorder.reshape(-1, 1))
# print(train_y_disorder)

# 多层感知器-回归模型(神经网络模型（监督）)
model_mlp = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(20, 20, 20), random_state=1)
# print(model_mlp)多层感知器模型
'''
MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(20, 20, 20), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

'''
model_mlp.fit(train_x_disorder, train_y_disorder.ravel())
mlp_score = model_mlp.score(test_x_disorder, test_y_disorder.ravel())
print('sklearn多层感知器-回归模型得分', mlp_score)

model_gbr_disorder = GradientBoostingRegressor()
# print(model_gbr_disorder) 回归树模型
'''
GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='ls', max_depth=3, max_features=None,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=1,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=100, presort='auto', random_state=None,
             subsample=1.0, verbose=0, warm_start=False)
'''
model_gbr_disorder.fit(train_x_disorder, train_y_disorder.ravel())
gbr_score_disorder = model_gbr_disorder.score(test_x_disorder, test_y_disorder.ravel())
print('sklearn集成-回归模型得分', gbr_score_disorder)  # 准确率较高 0.853817723868
'''
print('##########参数网格优选##############调优过程较为费时，因此选定参数后，不必每次都运行调优过程############################') 
model_gbr_GridSearch = GradientBoostingRegressor() 
# 设置参数池  参考 http://www.cnblogs.com/DjangoBlog/p/6201663.html
param_grid = {'n_estimators': range(20, 81, 10),        # 迭代次数寻优
              'learning_rate': [0.2, 0.1, 0.05, 0.02, 0.01],        # 学习率寻优
              'max_depth': [4, 6, 8],       # 决策树最大深度max_depth
              'min_samples_leaf': [3, 5, 9, 14],    # 内部节点再划分所需最小样本数min_samples_leaf
              'max_features': [0.8, 0.5, 0.3, 0.1],      # 最大特征寻优
              }
# 网格调参
from sklearn.model_selection import GridSearchCV 
estimator = GridSearchCV(model_gbr_GridSearch, param_grid)
estimator.fit(train_x_disorder, train_y_disorder.ravel())
print('最优调参：', estimator.best_params_)
# {'learning_rate': 0.1, 'max_depth': 6, 'max_features': 0.5, 'min_samples_leaf': 14, 'n_estimators': 70} 
# {'learning_rate': 0.1, 'max_depth': 6, 'max_features': 0.5, 'min_samples_leaf': 3, 'n_estimators': 70}
print('调参后得分', estimator.score(test_x_disorder, test_y_disorder.ravel()))
'''

print('###画图###########################################################################')
model_gbr_best = GradientBoostingRegressor(learning_rate=0.1, max_depth=6, max_features=0.5, min_samples_leaf=3,
                                           n_estimators=70)
model_gbr_best.fit(train_x_disorder, train_y_disorder.ravel())
model_score_best = model_gbr_best.score(test_x_disorder, test_y_disorder.ravel())
# print("sklearn集成-回归模型调优后得分", model_score_best)
# 使用默认参数的模型进行预测
gbr_pridict_disorder = model_gbr_best.predict(test_x_disorder)
# 多层感知器
mlp_pridict_disorder = model_mlp.predict(test_x_disorder)
import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']        # 显示中文字体
fig = plt.figure(figsize=(20, 3))  # dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80
axes = fig.add_subplot(1, 1, 1)
line3, = axes.plot(range(len(test_y_disorder)), test_y_disorder, 'g', label=U'实际')
line1, = axes.plot(range(len(gbr_pridict_disorder)), gbr_pridict_disorder, 'b--', label=U'集成模型', linewidth=2)
line2, = axes.plot(range(len(mlp_pridict_disorder)), mlp_pridict_disorder, 'r--', label=U'多层感知器', linewidth=2)
axes.grid()
fig.tight_layout()
plt.legend(handles=[line1, line2, line3])
# plt.legend(handles=[line1,  line3])
plt.title('sklearn 回归模型')
plt.show()