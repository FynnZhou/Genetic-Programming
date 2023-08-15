#%% include libraries
import numpy as np
import pandas as pd
import graphviz
from scipy.stats import rankdata 
import pickle

from gplearn import genetic
from gplearn.functions import make_function
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
from gplearn.fitness import make_fitness

from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split


import alphalens

#%% get data from pickles
# data form: 2-D dataframe include "high, low, code, date, etc.
with open('train_2017.1_2018.1_pre3_niu.pkl','rb') as f:
    df = pickle.load(f)
#%%
group = df.groupby('code')
df['my_pct'] = (group['close'].pct_change(periods=2))
fields = ['open', 'high', 'low', 'avg', 'pre_close', 'close', 'volume']

#%% define operater functions
# 系统自带的函数群

"""
Available individual functions are:

‘add’ : addition, arity=2.
‘sub’ : subtraction, arity=2.
‘mul’ : multiplication, arity=2.
‘div’ : protected division where a denominator near-zero returns 1., arity=2.
‘sqrt’ : protected square root where the absolute value of the argument is used, arity=1.
‘log’ : protected log where the absolute value of the argument is used and a near-zero argument returns 0., arity=1.
‘abs’ : absolute value, arity=1.
‘neg’ : negative, arity=1.
‘inv’ : protected inverse where a near-zero argument returns 0., arity=1. 
‘max’ : maximum, arity=2.
‘min’ : minimum, arity=2.
‘sin’ : sine (radians), arity=1.
‘cos’ : cosine (radians), arity=1.
‘tan’ : tangent (radians), arity=1.
"""

# init_function = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min', 'sin', 'cos', 'tan']
init_function = ['add', 'sub', 'mul', 'div','sqrt', 'log','inv','sin','max','min']


# 自定义函数, make_function函数群

def _rolling_rank(data): #第 i 个元素为𝑋𝑖在向量 X 中的分位数
    value = rankdata(data)[-1]
    return value  #scipy.rankdata 排序

def _rolling_prod(data):
    return np.prod(data) #所有元素乘积


def _delta(data):
    value = np.diff(data.flatten())
    value = np.append(0, value)
    return value

def _delay(data): # d 天以前的 X 值
    period=1  #当period为正时，默认是axis = 0轴的设定，向下移动
    value = pd.Series(data.flatten()).shift(period)
    value = np.nan_to_num(value)
    return value

def _ts_sum(data): #第 i 个元素为过去 d 天𝑋𝑖值构成的时序数列之和 
    window=10
    value = np.array(pd.Series(data.flatten()).rolling(window).sum().tolist())  
    value = np.nan_to_num(value)
    return value
    #a.flatten  把a降到一维，默认是按横的方向降
    #pandas.rolling(window) 以窗口取相邻数据进行计算
    
def _sma(data): #第 i 个元素为过去 d 天𝑋𝑖值构成的时序数列之平均值
    window=10
    value = np.array(pd.Series(data.flatten()).rolling(window).mean().tolist())
    value = np.nan_to_num(value)
    return value

def _stddev(data): #第 i 个元素为过去 d 天𝑋𝑖值构成的时序数列之标准差
    window=10
    value = np.array(pd.Series(data.flatten()).rolling(window).std().tolist())
    value = np.nan_to_num(value)
    return value

def _ts_rank(data): #第 i 个元素为过去 d 天𝑋𝑖值构成的时序数列之 分位数
    window=10
    value = np.array(pd.Series(data.flatten()).rolling(10).apply(_rolling_rank).tolist())
    value = np.nan_to_num(value)
    return value

def _product(data): #第 i 个元素为过去 d 天𝑋𝑖值构成的时序数列之 乘积
    window=10
    value = np.array(pd.Series(data.flatten()).rolling(10).apply(_rolling_prod).tolist())
    value = np.nan_to_num(value)
    return value

def _ts_min(data): #第 i 个元素为过去 d 天𝑋𝑖值构成的时序数列中最小值
    window=10
    value = np.array(pd.Series(data.flatten()).rolling(window).min().tolist())
    value = np.nan_to_num(value)
    return value

def _ts_max(data): #第 i 个元素为过去 d 天𝑋𝑖值构成的时序数列中最大值
    window=10
    value = np.array(pd.Series(data.flatten()).rolling(window).max().tolist())
    value = np.nan_to_num(value)
    return value


def _rank(data): 
    value = np.array(pd.Series(data.flatten()).rank().tolist())  #pandas.rank 平均排位
    value = np.nan_to_num(value)
    return value

def _scale(data): #scale(X, a) 向量 a*X/sum(abs(X))，a 的缺省值为 1，一般 a 应为正数
    k=1
    data = pd.Series(data.flatten())
    value = data.mul(1).div(np.abs(data).sum()) 
    value = np.nan_to_num(value)
    return value

def _ts_argmax(data): #第 i 个元素为过去 d 天𝑋𝑖值构成的时序数列中最大值出现的位置
    window=10
    value = pd.Series(data.flatten()).rolling(10).apply(np.argmax) + 1 
    value = np.nan_to_num(value)
    return value

def _ts_argmin(data): #第 i 个元素为过去 d 天𝑋𝑖值构成的时序数列中最小值出现的位置
    window=10
    value = pd.Series(data.flatten()).rolling(10).apply(np.argmin) + 1 
    value = np.nan_to_num(value)
    return value

# make_function函数群
delta = make_function(function=_delta, name='delta', arity=1)
delay = make_function(function=_delay, name='delay', arity=1)
rank = make_function(function=_rank, name='rank', arity=1)
scale = make_function(function=_scale, name='scale', arity=1)
sma = make_function(function=_sma, name='sma', arity=1)
stddev = make_function(function=_stddev, name='stddev', arity=1)
product = make_function(function=_product, name='product', arity=1)
ts_rank = make_function(function=_ts_rank, name='ts_rank', arity=1)
ts_min = make_function(function=_ts_min, name='ts_min', arity=1)
ts_max = make_function(function=_ts_max, name='ts_max', arity=1)
ts_argmax = make_function(function=_ts_argmax, name='ts_argmax', arity=1)
ts_argmin = make_function(function=_ts_argmin, name='ts_argmin', arity=1)
ts_sum = make_function(function=_ts_sum, name='ts_sum', arity=1)

user_function = [delta, delay, rank, scale, sma, stddev, product, ts_rank, ts_min, ts_max, ts_argmax, ts_argmin, ts_sum]

#%% define my metric（定义适应度）
def _my_metric(y, y_pred, w):
    x1 = pd.Series(y.flatten() )
    x2 = pd.Series(y_pred.flatten()) 
    df = pd.concat([x1,x2],axis=1)
    df.columns = ['y','y_pred']
    df.sort_values(by = 'y_pred',ascending = True,inplace = True)
    num = int(len(df)*0.1)
    #df.fillna(0, inplace=True)
    
    y_high = df["y"][-num:]
    y_low = df["y"][:num]
    #value = y_high.sum() - y_low.sum()
    value = y_high.mean()/y_low.mean()
    return value

my_metric = make_fitness(function=_my_metric, greater_is_better=True)

#%% generate expressions(生成表达式)
generations = 3
function_set = init_function + user_function
metric = my_metric
init_depth=(1,4)
population_size = 100
random_state=0
tournament_size=20
est_gp = SymbolicTransformer(
                            feature_names=fields, 
                            function_set=function_set,
                            generations=generations,
                            metric='spearman',   #'spearman'秩相关系数
                            init_depth=init_depth, # 公式树的初始化深度
                            population_size=population_size,
                            tournament_size= tournament_size, 
                            random_state=random_state,
                            p_crossover = 0.4,
                            p_subtree_mutation = 0.01,
                            p_hoist_mutation = 0,
                            p_point_mutation = 0.01,
                            p_point_replace = 0.4,
                         )
#%%
# 设定标签、训练集测试集
data = df[fields].values
target = df['predict_pct'].values

test_size=0.2
test_num = int(len(data)*test_size)

X_train = data[:-test_num]
X_test = data[-test_num:]
y_train = np.nan_to_num(target[:-test_num])
y_test = np.nan_to_num(target[-test_num:])  #使用0代替数组x中的nan元素，使用有限的数字代替inf元素


est_gp.fit(X_train, y_train)

#%% 将模型保存到本地
with open('gp_model_pre_3.pkl','wb') as w_f:
    pickle.dump(est_gp,w_f,0)

#%% 导入模型
with open('gp_model_pre_3.pkl', 'rb') as r_f:
    est_gp = pickle.load(r_f)


#%%
# 获取较优的表达式

best_programs = est_gp._best_programs
best_programs_dict = {}

for p in best_programs:
    factor_name = 'alpha_' + str(best_programs.index(p) + 1)
    best_programs_dict[factor_name] = {'fitness':p.fitness_, 'expression':str(p), 'depth':p.depth_, 'length':p.length_}
     
best_programs_dict = pd.DataFrame(best_programs_dict).T
best_programs_dict = best_programs_dict.sort_values(by='fitness')
# best_programs_dict

#%%
# 需要插入一个去极值，标准化处理函数
def standardize():
    pass

def winsorrize():

factor = pd.DataFrame(est_gp.transform(df[fields]),
                      index=df.index,
                      columns=['alpha1','alpha2','alpha3','alpha4','alpha5','alpha6','alpha7','alpha8','alpha9','alpha10'])
#%%
factor['code'] = df['code']
shaped_factor = []

for j, c in enumerate(factor.columns):
    if j == factor.shape[1] - 1:
        continue
    fac = factor[[c,'code']]
    shaped_factor.append(pd.pivot_table(fac,values=c,index=fac.index,columns='code'))

# shaped data form: index-->date, columns-->code, values-->factor values
#%%
# show the expression using binary tree
def alpha_factor_graph(num):
    # num 
    factor = best_programs[num-1]
    print(factor)
    print('fitness: {0}, depth: {1}, length: {2}'.format(factor.fitness_, factor.depth_, factor.length_))

    dot_data = factor.export_graphviz()
    graph = graphviz.Source(dot_data)
#     graph.render('images/alpha_factor_graph', format='png', cleanup=True)
    
    return graph
# 
#%%
graph = alpha_factor_graph(4)
graph