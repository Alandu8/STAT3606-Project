#%%

# 加载基础package
from cmath import inf
from random import random
from re import X
import numpy as np
import pandas as pd
import os
import datetime
import random

# 加载random normal
from numpy.random import normal

# 加载univariate isotonic package
from sklearn.datasets import make_regression
from sklearn.isotonic import IsotonicRegression
# rolling window package and scaling
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture

# 加载gurobi 数据包
import gurobipy as gp
from gurobipy import GRB
from math import sqrt

# 加载PyGAD
import pygad

from scipy.stats import beta, gamma, norm, binom, uniform, t

# 加载绘图包
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 构建导出最后结果的地址并储存
folder_name = 'y_to_predict_half_day_300_linear_3only'
path = os.path.join('/home/liheng/Futures/results', folder_name)
plot_path = os.path.join(path, 'fitness')
# Check whether the specified path exists or not
isExist = os.path.exists(plot_path)
if not isExist:
    # Create a new directory because it does not exist 
    os.makedirs(plot_path)

# 设置所使用的y
y_slot = 'y_to_predict_half_day'

# 动态取bound有大问题，要进行更改
def linear_opt_bound(decision_df,combined_df,window_count,y_slot,path = plot_path):

    # 设置基础参数
    num_generations = 100 # 经过测试，20以上的generation 已经足够converge
    num_parents_mating = 5
    sol_per_pop = 10
    num_genes = 2
    init_range_low = 0
    init_range_high = 1
    mutation_probability_list = [0.25,0.1]
    gene_type = float
                
    # 定义目标函数
    # 注意：这里除了要最大化目标函数之外，还要将constraint转化成penalty
    penalty_parameter = 10
    y_numerical = combined_df[y_slot].values

    print(decision_df.shape[0])
    print(combined_df[y_slot].shape[0])

    def fitness_func(solution, solution_idx):
        converter = lambda x : -1 if x < solution[1] else (1 if x > solution[0] else 0)
        decision_df['decision'] = decision_df['decision'].apply(converter)

        gap = solution[0] - solution[1]
        indicator = lambda x : 0 if x > 0 else 1

        total_return = np.sum(decision_df.values*y_numerical) - penalty_parameter*indicator(gap)

        return total_return
                
    # 生成遗传算法对象并开始训练
    ga_instance = pygad.GA(num_generations = num_generations,
                            num_parents_mating = num_parents_mating,
                            sol_per_pop = sol_per_pop,
                            num_genes = num_genes,
                            init_range_low = init_range_low,
                            init_range_high = init_range_high,
                            fitness_func = fitness_func,
                            gene_type = gene_type,
                            parallel_processing=20,
                            mutation_type="adaptive",
                            mutation_probability =  mutation_probability_list)
    print('开始计算最佳threshold, window: ',window_count)
    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(solution)
    upper_bound = solution[0]
    lower_bound = solution[1]
    return upper_bound,lower_bound

def linear_optimization(x,combined_df,window_count,y_slot,path = plot_path):

    # Use maximum rank correlation estimator, mixed-integer programming
    # https://arxiv.org/abs/2009.03844
     
    stock = x.columns
    m = gp.Model('futures')
    # 初始化变量
    n = x.shape[0]
    dummys = m.addVars(n,n,lb=0, ub=1, vtype=GRB.BINARY)  # d_ij
    betas = pd.Series(m.addVars(stock, vtype=GRB.CONTINUOUS),index=stock) # beta_1 ... 5

    # Restriction variable M is set to be a large enough constant where always larger than the absolute bound of the index x_ij* Beta
    # Hence the inequality is changed to weaker version than strict inequality
    M = 10

    # pairwise difference for every y
    y_diff = combined_df[y_slot].values - combined_df[y_slot].values[:, None]
    #y_diff = combined_df['Rank'].values - combined_df['Rank'].values[:, None]
    y_diff = 1 * (y_diff > 0)

    # Add constraints
    m.addConstrs((((x.iloc[i,:] - x.iloc[j,:]).T).dot(betas) <= M*dummys[i,j] for i in range(n)
                            for j in range(n)
                            if i != j), name='c1')
    m.addConstrs((M*(dummys[i,j]-1)<=((x.iloc[i,:] - x.iloc[j,:]).T).dot(betas) for i in range(n)
                            for j in range(n)
                            if i != j), name='c2')

    # set objective function
    expr = gp.quicksum(dummys[i,j]*y_diff[i,j]/(n*(n-1)) for i in range(n)
                            for j in range(n)
                            if i != j)
    # in total there are two objectives
    m.NumObj = 2
    m.ObjNWeight = 1
    m.ObjNPriority = 2
    m.ObjNName = "MaxSum"
    m.setObjective(expr, GRB.MAXIMIZE)

    m.ObjNWeight = 1
    m.ObjNPriority = 1
    m.ObjNName = "MaxBeta"
    m.setObjective(gp.quicksum(betas.iloc[i] for i in range(x.shape[1])), GRB.MAXIMIZE)

    print('开始计算最佳beta, window: ',window_count)
    m.optimize()

    beta_list = []
    for v in betas:
        beta_list.append(v.X)
        
    beta_array = np.array(beta_list)
    
   
    print("Betas are",beta_array)
    
    return beta_array

# load features， 后续可以考虑将money bar加入其中
data_df = pd.read_csv("/home/liheng/Futures/data/complete_dataset_ver2.csv")
data_df.dropna(inplace=True)
data_df.index = data_df['date_time'] 
data_df = data_df.drop(['date_time'], axis = 1)
# data_df = data_df.iloc[0:330,:]

x_df = data_df[['a50_before_open_return','a50_before_open_high_close_ratio','a50_before_open_low_close_ratio']] # ,'a50_before_open_high_close_ratio','a50_before_open_low_close_ratio','sp500_past_return_1d','gdc_past_return_1d','cnyusd_exchange_rate_change_1d'
y_df = data_df[[y_slot]]

# high - volatility

# Multivariate isotonic regression
# https://www-jstor-org.eproxy.lib.hku.hk/stable/pdf/2335561.pdf?refreqid=excelsior%3Ad836f2fc891178c1f77bed09203352af&ab_segments=&origin=&acceptTC=1

train_length = 300
test_length = 15
n_splits = (y_df.shape[0])//test_length

rolling = TimeSeriesSplit(n_splits = n_splits, max_train_size=train_length, test_size=test_length)
all_windows_decision_dfs = []
beta_df = pd.DataFrame()
bound_df = pd.DataFrame()
window_count = 0
for train_index, test_index in rolling.split(x_df):
    window_count += 1
    X_train, X_test = x_df.iloc[train_index].to_numpy(), x_df.iloc[test_index].to_numpy()
    y_train, y_test = y_df.iloc[train_index], y_df.iloc[test_index]

    # normalize data
    scaler = MinMaxScaler()
    X_train_scaled_array = scaler.fit_transform(X_train, y_train)
    X_test_scaled_array = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled_array, columns = ['a50_before_open_return','a50_before_open_high_close_ratio','a50_before_open_low_close_ratio'])
    X_test_scaled = pd.DataFrame(X_test_scaled_array, columns = ['a50_before_open_return','a50_before_open_high_close_ratio','a50_before_open_low_close_ratio'])

    # sort dataframe according to rank
    X_train_scaled_df = pd.DataFrame(X_train_scaled_array, columns = ['a50_before_open_return','a50_before_open_high_close_ratio','a50_before_open_low_close_ratio'],index = y_train.index)
    combined_df = pd.concat([y_train,X_train_scaled_df],axis=1)
    combined_df['Rank'] = combined_df[y_slot].rank(ascending=True)

    
    beta_array = linear_optimization(X_train_scaled_df,combined_df,window_count,y_slot)
    # store beta array
    beta_temp_df = pd.DataFrame([beta_array])
    if beta_df is None:
        beta_df = beta_temp_df
    else:
        beta_df = pd.concat([beta_df,beta_temp_df])
    
    test_decision_array = np.matmul(X_test_scaled.to_numpy(),beta_array)
    train_decision_array = np.matmul(X_train_scaled.to_numpy(),beta_array)

    # 整理结果
    decision_df = pd.DataFrame(test_decision_array, columns = ["decision"], index = y_test.index)
    train_decision_df = pd.DataFrame(train_decision_array, columns = ["decision"], index = y_train.index)

    # 允许long & short, 寻找最佳bound
    upper_bound,lower_bound = linear_opt_bound(train_decision_df,combined_df,window_count,y_slot)
    bound_array = np.array([upper_bound,lower_bound])
    bound_temp_df = pd.DataFrame([bound_array])
    if bound_df is None:
        bound_df = bound_temp_df
    else:
        bound_df = pd.concat([bound_df,bound_temp_df])
    
    test_decision_array = np.matmul(X_test_scaled.to_numpy(),beta_array)

    # 根据rank排名导出decision df
    converter = lambda x : -1 if x < lower_bound else (1 if x > upper_bound else 0)
    decision_df['decision'] = decision_df['decision'].apply(converter)
    print(decision_df)
    
    # 储存结果
    all_windows_decision_dfs.append(decision_df)

# 整理结果并返回
final_decision_df = None
for each_decision_df in all_windows_decision_dfs:
    if final_decision_df is None:
        final_decision_df = each_decision_df
    else:
        final_decision_df = pd.concat([final_decision_df,each_decision_df])
final_decision_df = final_decision_df.sort_index()

# final_decision_df.to_csv('/home/liheng/Futures/first_test.csv',index=True)

total_return_df = data_df[data_df.index.isin(final_decision_df.index)][[y_slot]]

return_df = total_return_df.mul(final_decision_df.values)
return_df.columns = ['A']

transaction_cost = 0.00033 # 目前记录为万分之三, 完整一笔交易
cost_df = final_decision_df*transaction_cost
cost_df = cost_df.abs()

# 通过每次回报计算累计回报（加法）
cumulative_return = []
net_return = []
count = 0
for i in return_df.values:
    if len(cumulative_return) == 0:
        cumulative_return.append(i)
        net_return.append(i-cost_df.values[count])
    else:
        cumulative_return.append(i + cumulative_return[-1])
        net_return.append(i-cost_df.values[count]+net_return[-1])
    count += 1
cumulative_return = pd.DataFrame(cumulative_return, index = final_decision_df.index)
net_return = pd.DataFrame(net_return, index = final_decision_df.index)
annual_trading_days = 250
total_days = len(cumulative_return)
year_count = total_days / annual_trading_days
strategy_annualized_return = cumulative_return.iloc[-1] / year_count

# 计算win rate
true_winning_rate =  return_df.loc[return_df['A'] > 0,'A'].count()/return_df['A'].count()

plt.plot(cumulative_return, label = "raw", color = "r")
plt.plot(net_return, label = "net", color = "b")
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(100))
plt.grid()
ax = plt.gca()
ax.spines["bottom"].set_position(("data", 0))
plt.xlabel("Date", fontsize = 14)
plt.xticks(fontsize = 7)
plt.ylabel("Cumulative Return", fontsize = 14)
plt.title("Cumulative Return Performance", fontsize = 18)
plt.rcParams["savefig.dpi"] = 300
plt.legend()


beta_df.to_csv(os.path.join(path,"beta.csv"))
bound_df.to_csv(os.path.join(path,"bound.csv"))
plt.savefig(os.path.join(path,"cumulative_return.jpg"))

# 将关键结果记录在txt文件内
file1 = open(os.path.join(path,"values.txt"),"w")
  
# \n is placed to indicate EOL (End of Line)
file1.write("Raw Cumulative Return: "+str(cumulative_return.iloc[-1] * 100)+"\n")
file1.write("Raw Strategy Annualized Return: " + str(strategy_annualized_return * 100)+"\n")
file1.write("Net Annualized Return: " + str(net_return.iloc[-1] * 100)+"\n")
file1.write("Transaction Cost: "+str(transaction_cost)+"\n")
file1.write("Winning Rate: "+str(true_winning_rate* 100)+"\n")
file1.close() 




''' '''
# %%
