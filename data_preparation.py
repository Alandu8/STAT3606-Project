### 生成各种y_to_predict，以及各种features

# 加载基础package
import numpy as np
import pandas as pd
import os
import datetime

# 设置储存文件的文件夹路径
raw_data_folder_address = os.path.join("/home/liheng/Futures/data", "raw_data")

#%%

# 生成各种y_to_predict数据
def generate_y_to_predict():
    # 如果是within_day，则是在09:30:00买入，并在15:00:00卖出
    # 如果是1d，则是在09:30:00买入，并在第二天09:30:00卖出
    
    # 读取A50数据和沪深300数据
    a50_1m = pd.read_csv(os.path.join(raw_data_folder_address, "A50_1min.csv"))
    a50_1m = a50_1m.set_index("date_time")
    shsz300_1m = pd.read_csv(os.path.join(raw_data_folder_address, "SHSZ300_1min.csv"))
    shsz300_1m = shsz300_1m.set_index("date_time")
    
    # 依次生成想要的y_to_predict。首先生成a50_within_day
    a50_within_day_selected_days = list(a50_1m[a50_1m.index.str.slice(11, 19) == "09:30:00"].index.str.slice(0, 10))
    a50_within_day = pd.Series(index = a50_within_day_selected_days)
    for each_day in a50_within_day_selected_days:
        a50_within_day[each_day] = a50_1m.loc[each_day + " 15:00:00", "close"] / a50_1m.loc[each_day + " 09:30:00", "close"] - 1
    a50_within_day.name = "a50_within_day"
    
    # 生成a50_1d
    a50_1d = a50_1m[a50_1m.index.str.slice(11, 19) == "09:30:00"]["close"].pct_change().shift(-1)
    a50_1d.index = a50_1d.index.str.slice(0, 10)
    a50_1d.name = "a50_1d"
    
    # 生成a50_half_day（09:30:00-11:30:00）
    a50_half_day_selected_days = list(a50_1m[a50_1m.index.str.slice(11, 19) == "09:30:00"].index.str.slice(0, 10))
    a50_half_day = pd.Series(index = a50_half_day_selected_days)
    for each_day in a50_half_day_selected_days:
        a50_half_day[each_day] = a50_1m.loc[each_day + " 11:30:00", "close"] / a50_1m.loc[each_day + " 09:30:00", "close"] - 1
    a50_half_day.name = "a50_half_day"
    
    # 生成a50_high_open_ratio（时间范围仍然为09:30:00-15:00:00，用于结果对照）
    a50_high_open_selected_days = list(a50_1m[a50_1m.index.str.slice(11, 19) == "09:30:00"].index.str.slice(0, 10))
    a50_high_open_ratio = pd.Series(index = a50_high_open_selected_days)
    for each_day in a50_high_open_selected_days:
        a50_high_open_ratio[each_day] = a50_1m.loc[(a50_1m.index >= each_day + " 09:31:00") & (a50_1m.index <= each_day + " 15:00:00"), "high"].max() / a50_1m.loc[each_day + " 09:30:00", "close"] - 1
    a50_high_open_ratio.name = "a50_high_open_ratios"
    
    # 生成shsz300_within_day
    shsz300_within_day_selected_days = list(shsz300_1m[shsz300_1m.index.str.slice(11, 19) == "09:31:00"].index.str.slice(0, 10))
    shsz300_within_day = pd.Series(index = shsz300_within_day_selected_days)
    for each_day in shsz300_within_day_selected_days:
        shsz300_within_day[each_day] = shsz300_1m.loc[each_day + " 15:00:00", "close"] / shsz300_1m.loc[each_day + " 09:31:00", "open"] - 1
    shsz300_within_day.name = "shsz300_within_day"
    
    # 生成shsz300_1d
    shsz300_1d = shsz300_1m[shsz300_1m.index.str.slice(11, 19) == "09:31:00"]["open"].pct_change().shift(-1)
    shsz300_1d.index = shsz300_1d.index.str.slice(0, 10)
    shsz300_1d.name = "shsz300_1d"
    
    # 生成shsz300_half_day
    shsz300_half_day_selected_days = list(shsz300_1m[shsz300_1m.index.str.slice(11, 19) == "09:31:00"].index.str.slice(0, 10))
    shsz300_half_day = pd.Series(index = shsz300_half_day_selected_days)
    for each_day in shsz300_half_day_selected_days:
        shsz300_half_day[each_day] = shsz300_1m.loc[each_day + " 11:30:00", "close"] / shsz300_1m.loc[each_day + " 09:31:00", "open"] - 1
    shsz300_half_day.name = "shsz300_half_day"    
    
    # 生成shsz300_high_open_ratio
    shsz300_high_open_selected_days = list(shsz300_1m[shsz300_1m.index.str.slice(11, 19) == "09:31:00"].index.str.slice(0, 10))
    shsz300_high_open_ratio = pd.Series(index = shsz300_high_open_selected_days)
    for each_day in shsz300_high_open_selected_days:
        shsz300_high_open_ratio[each_day] = shsz300_1m.loc[(shsz300_1m.index >= each_day + " 09:31:00") & (shsz300_1m.index <= each_day + " 15:00:00"), "high"].max() / shsz300_1m.loc[each_day + " 09:31:00", "open"] - 1
    shsz300_high_open_ratio.name = "shsz300_high_open_ratio"
    
    # 生成记录数据的DataFrame
    y_to_predict_df = pd.concat([a50_within_day, a50_1d, a50_half_day, a50_high_open_ratio, shsz300_within_day, shsz300_1d, shsz300_half_day, shsz300_high_open_ratio], axis = 1)
    y_to_predict_df = y_to_predict_df.sort_index()
    y_to_predict_df.index.name = "date_time"
    
    # 注意：a50数据在2022-10-06的a50_1d数据很可能存在异常，这里手动将其移除
    # 其原因是缺少了2022-10-07的早盘数据
    y_to_predict_df.loc["2022-10-06", "a50_1d"] = np.nan

    # 保存结果
    y_to_predict_df.to_csv("y_to_predict.csv")

#%%

# 生成各种features（或者称他们为signals）
def generate_features():
    # 这里需要确保各列的时间信息使用，在这一步中完成所需的shift操作
    # 目前已发现非常好用的signal：
    # A50指数的盘前收益率（05:15:00-09:30:00）
    ### 千万切记：
    ### features在使用的时候绝不一定是越多越好
    ### 这里生成的很多features，其实有一些含义是很相似的，同时使用必然会出现多重共线性相关问题
    ### 此外，之前的很多测试发现，由于金融数据很低的信噪比，任意添加features很可能会使得之前单独预测力很强的features失效
    ### 最经典的例子就是用OLS & Logistic使用几十个features的结果，远远差于，仅使用A50指数的盘前收益率（05:15:00-09:30:00）和简单rule的结果
    
    # 记录全部feature结果的list
    all_feature_series_list = []
    
    # 关于美股的features由于时间记录问题需要单独储存并之后特殊处理才可以和前面的feature结合
    us_index_feature_series_dict = {}
    
    # 读取A50数据，沪深300指数数据，美股四大指数数据，VIX数据，美国十年期国债收益率数据和人民币兑美元汇率
    a50_1m = pd.read_csv(os.path.join(raw_data_folder_address, "A50_1min.csv"))
    a50_1m = a50_1m.set_index("date_time")
    shsz300_1m = pd.read_csv(os.path.join(raw_data_folder_address, "SHSZ300_1min.csv"))
    shsz300_1m = shsz300_1m.set_index("date_time")
    sp500_daily = pd.read_csv(os.path.join(raw_data_folder_address, "SP500_daily.csv"))
    sp500_daily = sp500_daily.set_index("date_time")
    nasdaq_daily = pd.read_csv(os.path.join(raw_data_folder_address, "NASDAQ_daily.csv"))
    nasdaq_daily = nasdaq_daily.set_index("date_time")
    dji_daily = pd.read_csv(os.path.join(raw_data_folder_address, "DJI_daily.csv"))
    dji_daily = dji_daily.set_index("date_time")
    gdc_daily = pd.read_csv(os.path.join(raw_data_folder_address, "GDC_daily.csv"))
    gdc_daily = gdc_daily.set_index("date_time")
    vix_daily = pd.read_csv(os.path.join(raw_data_folder_address, "VIX_daily.csv"))
    vix_daily = vix_daily.set_index("date_time")
    teny_yield_daily = pd.read_csv(os.path.join(raw_data_folder_address, "teny_yield_daily.csv"))
    teny_yield_daily = teny_yield_daily.set_index("date_time")
    cnyusd_daily = pd.read_csv(os.path.join(raw_data_folder_address, "CNYUSD_daily.csv"))
    cnyusd_daily = cnyusd_daily.set_index("date_time")
    
    # =========================================================================
    
    # 生成feature：A50指数的盘前收益率
    # 设定四类：前一日16:35:00-09:30:00，前一日17:00:00-09:30:00，05:15:00-09:30:00，09:00:00-09:30:00
    before_open_return_type_one_series = pd.Series()
    before_open_return_type_one_series.name = "before_open_return_type_one"
    before_open_return_type_two_series = pd.Series()
    before_open_return_type_two_series.name = "before_open_return_type_two"
    before_open_return_type_three_series = pd.Series()
    before_open_return_type_three_series.name = "before_open_return_type_three"
    before_open_return_type_four_series = pd.Series()
    before_open_return_type_four_series.name = "before_open_return_type_four"
    type_one_start_price = a50_1m[a50_1m.index.str.slice(11, 19) == "16:35:00"]
    type_two_start_price = a50_1m[a50_1m.index.str.slice(11, 19) == "17:01:00"]
    type_three_start_price = a50_1m[a50_1m.index.str.slice(11, 19) == "05:15:00"]
    type_four_start_price = a50_1m[a50_1m.index.str.slice(11, 19) == "09:01:00"]
    finish_price = a50_1m[a50_1m.index.str.slice(11, 19) == "09:30:00"]
    for each_date in finish_price.index.str.slice(0, 10):
        temp_type_one_start_price = type_one_start_price.loc[type_one_start_price.index.str.slice(0, 10) < each_date, "close"]
        temp_type_two_start_price = type_two_start_price.loc[type_two_start_price.index.str.slice(0, 10) < each_date, "open"]
        temp_type_three_start_price = type_three_start_price.loc[type_three_start_price.index.str.slice(0, 10) <= each_date, "close"]
        temp_type_four_start_price = type_four_start_price.loc[type_four_start_price.index.str.slice(0, 10) <= each_date, "open"]
        temp_finish_price = a50_1m.loc[each_date + " 09:30:00", "close"]
        if temp_type_one_start_price.size != 0:
            before_open_return_type_one_series[each_date] = temp_finish_price / temp_type_one_start_price[-1] - 1
        if temp_type_two_start_price.size != 0:
            before_open_return_type_two_series[each_date] = temp_finish_price / temp_type_two_start_price[-1] - 1
        if temp_type_three_start_price.size != 0:
            before_open_return_type_three_series[each_date] = temp_finish_price / temp_type_three_start_price[-1] - 1
        if temp_type_four_start_price.size != 0:
            before_open_return_type_four_series[each_date] = temp_finish_price / temp_type_four_start_price[-1] - 1
    all_feature_series_list.append(before_open_return_type_one_series)
    all_feature_series_list.append(before_open_return_type_two_series)
    all_feature_series_list.append(before_open_return_type_three_series)
    all_feature_series_list.append(before_open_return_type_four_series)
    
    # =========================================================================

    # 生成feature：A50指数的盘前波动率
    # 设定四类：前一日16:35:00-09:30:00，前一日17:00:00-09:30:00，05:15:00-09:30:00，09:00:00-09:30:00
    pass  # TODO
    
    # =========================================================================

    # 生成feature：A50指数的盘前交易量信息
    # 时间范围设定在09:00:00-09:30:00
    pass  # TODO
    
    # =========================================================================

    # 生成feature：A50指数的盘前high_close_ratio和low_close_ratio
    # 时间范围设定在09:00:00-09:30:00
    before_open_high_close_series = pd.Series()
    before_open_high_close_series.name = "before_open_high_close_ratio"
    before_open_low_close_series = pd.Series()
    before_open_low_close_series.name = "before_open_low_close_ratio"
    selected_data = a50_1m[(a50_1m.index.str.slice(11, 19) >= "09:01:00") & (a50_1m.index.str.slice(11, 19) <= "09:30:00")]
    for each_date in selected_data.index.str.slice(0, 10).unique():
        temp_selected_data = selected_data[selected_data.index.str.slice(0, 10) == each_date]
        close_price = temp_selected_data["close"][-1]
        high_price = temp_selected_data["high"].max()
        low_price = temp_selected_data["low"].min()
        before_open_high_close_series[each_date] = high_price / close_price - 1
        before_open_low_close_series[each_date] = low_price / close_price - 1
    all_feature_series_list.append(before_open_high_close_series)
    all_feature_series_list.append(before_open_low_close_series)
    
    # =========================================================================

    # 生成feature：A50指数的历史收益率
    # 历史时间段回顾时长为1, 2, 3, 5, 10, 20日
    # 时间点设定在05:15:00，因为在北京时间05:15:00才算是完整地结束了一个交易日
    date_time_index = a50_1m[a50_1m.index.str.slice(11, 19) == "09:30:00"].index.str.slice(0, 10)
    past_return_1d = pd.Series()
    past_return_1d.name = "a50_past_return_1d"
    past_return_2d = pd.Series()
    past_return_2d.name = "a50_past_return_2d"
    past_return_3d = pd.Series()
    past_return_3d.name = "a50_past_return_3d"
    past_return_5d = pd.Series()
    past_return_5d.name = "a50_past_return_5d"
    past_return_10d = pd.Series()
    past_return_10d.name = "a50_past_return_10d"
    past_return_20d = pd.Series()
    past_return_20d.name = "a50_past_return_20d"
    selected_data = a50_1m[a50_1m.index.str.slice(11, 19) == "05:15:00"]
    for each_date in date_time_index:
        past_return_days_start_price = selected_data.loc[selected_data.index.str.slice(0, 10) <= each_date, "close"]
        finish_price = past_return_days_start_price[-1]
        if past_return_days_start_price.size >= 1 + 1:
            past_return_1d[each_date] = finish_price / past_return_days_start_price[-1 - 1] - 1
        if past_return_days_start_price.size >= 2 + 1:
            past_return_2d[each_date] = finish_price / past_return_days_start_price[-2 - 1] - 1
        if past_return_days_start_price.size >= 3 + 1:
            past_return_3d[each_date] = finish_price / past_return_days_start_price[-3 - 1] - 1    
        if past_return_days_start_price.size >= 5 + 1:
            past_return_5d[each_date] = finish_price / past_return_days_start_price[-5 - 1] - 1
        if past_return_days_start_price.size >= 10 + 1:
            past_return_10d[each_date] = finish_price / past_return_days_start_price[-10 - 1] - 1
        if past_return_days_start_price.size >= 20 + 1:
            past_return_20d[each_date] = finish_price / past_return_days_start_price[-20 - 1] - 1
    all_feature_series_list.append(past_return_1d)
    all_feature_series_list.append(past_return_2d)
    all_feature_series_list.append(past_return_3d)
    all_feature_series_list.append(past_return_5d)
    all_feature_series_list.append(past_return_10d)
    all_feature_series_list.append(past_return_20d)

    # =========================================================================

    # 生成feature：A50指数的前一日交易时间段波动率
    # 时间段设定在09:30:00-15:00:00
    pass  # TODO
    
    # =========================================================================

    # 生成feature：SHSZ300指数的历史收益率
    # 历史时间段回顾时长为1, 2, 3, 5, 10, 20日
    # 时间点设定在15:00:00
    date_time_index = shsz300_1m[shsz300_1m.index.str.slice(11, 19) == "09:31:00"].index.str.slice(0, 10)
    past_return_1d = pd.Series()
    past_return_1d.name = "shsz300_past_return_1d"
    past_return_2d = pd.Series()
    past_return_2d.name = "shsz300_past_return_2d"
    past_return_3d = pd.Series()
    past_return_3d.name = "shsz300_past_return_3d"
    past_return_5d = pd.Series()
    past_return_5d.name = "shsz300_past_return_5d"
    past_return_10d = pd.Series()
    past_return_10d.name = "shsz300_past_return_10d"
    past_return_20d = pd.Series()
    past_return_20d.name = "shsz300_past_return_20d"
    selected_data = shsz300_1m[shsz300_1m.index.str.slice(11, 19) == "15:00:00"]
    for each_date in date_time_index:
        past_return_days_start_price = selected_data.loc[selected_data.index.str.slice(0, 10) < each_date, "close"]
        if past_return_days_start_price.size == 0:
            continue
        finish_price = past_return_days_start_price[-1]
        if past_return_days_start_price.size >= 1 + 1:
            past_return_1d[each_date] = finish_price / past_return_days_start_price[-1 - 1] - 1
        if past_return_days_start_price.size >= 2 + 1:
            past_return_2d[each_date] = finish_price / past_return_days_start_price[-2 - 1] - 1
        if past_return_days_start_price.size >= 3 + 1:
            past_return_3d[each_date] = finish_price / past_return_days_start_price[-3 - 1] - 1    
        if past_return_days_start_price.size >= 5 + 1:
            past_return_5d[each_date] = finish_price / past_return_days_start_price[-5 - 1] - 1
        if past_return_days_start_price.size >= 10 + 1:
            past_return_10d[each_date] = finish_price / past_return_days_start_price[-10 - 1] - 1
        if past_return_days_start_price.size >= 20 + 1:
            past_return_20d[each_date] = finish_price / past_return_days_start_price[-20 - 1] - 1
    all_feature_series_list.append(past_return_1d)
    all_feature_series_list.append(past_return_2d)
    all_feature_series_list.append(past_return_3d)
    all_feature_series_list.append(past_return_5d)
    all_feature_series_list.append(past_return_10d)
    all_feature_series_list.append(past_return_20d)
        
    # =========================================================================

    # 生成feature：SHSZ300指数的前一日交易时间段波动率
    # 时间段设定在09:30:00-15:00:00
    pass  # TODO

    # =========================================================================

    # 生成feature：SHSZ300指数的收盘前收益率和波动率
    # 设定时间段为：前一日14:00:00-前一日15:00:00，前一日14:45:00-前一日15:00:00
    pass  # TODO

    # =========================================================================

    # 生成feature：美股四大指数的历史收益率
    # 历史时间段回顾时长为1, 2, 3, 5, 10, 20日
    all_indexes_dict = {"sp500": sp500_daily, "nasdaq": nasdaq_daily, "dji": dji_daily, "gdc": gdc_daily}
    for each_index_name in all_indexes_dict:
        each_index_data = all_indexes_dict[each_index_name]
        past_return_1d = each_index_data.close.pct_change(1)
        past_return_1d.name = each_index_name + "_past_return_1d"
        past_return_2d = each_index_data.close.pct_change(2)
        past_return_2d.name = each_index_name + "_past_return_2d"
        past_return_3d = each_index_data.close.pct_change(3)
        past_return_3d.name = each_index_name + "_past_return_3d"
        past_return_5d = each_index_data.close.pct_change(5)
        past_return_5d.name = each_index_name + "_past_return_5d"
        past_return_10d = each_index_data.close.pct_change(10)
        past_return_10d.name = each_index_name + "_past_return_10d"
        past_return_20d = each_index_data.close.pct_change(20)
        past_return_20d.name = each_index_name + "_past_return_20d"
        us_index_feature_series_dict[past_return_1d.name] = past_return_1d
        us_index_feature_series_dict[past_return_2d.name] = past_return_2d
        us_index_feature_series_dict[past_return_3d.name] = past_return_3d
        us_index_feature_series_dict[past_return_5d.name] = past_return_5d
        us_index_feature_series_dict[past_return_10d.name] = past_return_10d
        us_index_feature_series_dict[past_return_20d.name] = past_return_20d
        
    # =========================================================================

    # 生成feature：美股四大指数的最近日内收益率
    # 不跨日期
    all_indexes_dict = {"sp500": sp500_daily, "nasdaq": nasdaq_daily, "dji": dji_daily, "gdc": gdc_daily}
    for each_index_name in all_indexes_dict:
        each_index_data = all_indexes_dict[each_index_name]
        past_return_within_day = each_index_data["close"] / each_index_data["open"] - 1
        past_return_within_day.name = each_index_name + "_past_return_within_day"
        us_index_feature_series_dict[past_return_within_day.name] = past_return_within_day

    # =========================================================================

    # 生成feature：人民币兑美元汇率的历史变化率
    # 历史时间段回顾时长为1, 2, 3, 5, 10, 20日
    exchange_rate_return_1d = cnyusd_daily.close.pct_change(1)
    exchange_rate_return_1d.name = "cnyusd_exchange_rate_change_1d"
    exchange_rate_return_2d = cnyusd_daily.close.pct_change(2)
    exchange_rate_return_2d.name = "cnyusd_exchange_rate_change_2d"
    exchange_rate_return_3d = cnyusd_daily.close.pct_change(3)
    exchange_rate_return_3d.name = "cnyusd_exchange_rate_change_3d"
    exchange_rate_return_5d = cnyusd_daily.close.pct_change(5)
    exchange_rate_return_5d.name = "cnyusd_exchange_rate_change_5d"
    exchange_rate_return_10d = cnyusd_daily.close.pct_change(10)
    exchange_rate_return_10d.name = "cnyusd_exchange_rate_change_10d"
    exchange_rate_return_20d = cnyusd_daily.close.pct_change(20)
    exchange_rate_return_20d.name = "cnyusd_exchange_rate_change_20d"
    us_index_feature_series_dict[exchange_rate_return_1d.name] = exchange_rate_return_1d
    us_index_feature_series_dict[exchange_rate_return_2d.name] = exchange_rate_return_2d
    us_index_feature_series_dict[exchange_rate_return_3d.name] = exchange_rate_return_3d
    us_index_feature_series_dict[exchange_rate_return_5d.name] = exchange_rate_return_5d
    us_index_feature_series_dict[exchange_rate_return_10d.name] = exchange_rate_return_10d
    us_index_feature_series_dict[exchange_rate_return_20d.name] = exchange_rate_return_20d
    
    # =========================================================================

    # 生成feature：VIX数据和美国十年期国债收益率数据的历史涨跌幅
    # 历史时间段回顾时长为1, 2, 3, 5, 10, 20
    # 使用absolute value而不是ratio
    all_other_info_dict = {"vix": vix_daily, "teny": teny_yield_daily}
    for each_info_name in all_other_info_dict:
        each_info_data = all_other_info_dict[each_info_name]
        change_1d_series = each_info_data.close - each_info_data.close.shift(1)
        change_1d_series.name = each_info_name + "_change_1d"
        change_2d_series = each_info_data.close - each_info_data.close.shift(2)
        change_2d_series.name = each_info_name + "_change_2d"
        change_3d_series = each_info_data.close - each_info_data.close.shift(3)
        change_3d_series.name = each_info_name + "_change_3d"
        change_5d_series = each_info_data.close - each_info_data.close.shift(5)
        change_5d_series.name = each_info_name + "_change_5d"
        change_10d_series = each_info_data.close - each_info_data.close.shift(10)
        change_10d_series.name = each_info_name + "_change_10d"
        change_20d_series = each_info_data.close - each_info_data.close.shift(20)
        change_20d_series.name = each_info_name + "_change_20d"
        us_index_feature_series_dict[change_1d_series.name] = change_1d_series
        us_index_feature_series_dict[change_2d_series.name] = change_2d_series
        us_index_feature_series_dict[change_3d_series.name] = change_3d_series
        us_index_feature_series_dict[change_5d_series.name] = change_5d_series
        us_index_feature_series_dict[change_10d_series.name] = change_10d_series
        us_index_feature_series_dict[change_20d_series.name] = change_20d_series

    # =========================================================================

    # 生成记录数据的DataFrame
    all_features_df = pd.concat(all_feature_series_list, axis = 1)
    all_features_df = all_features_df.sort_index()
    all_features_df.index.name = "date_time"
    
    # 生成记录美股feature的DataFrame并向完整的DataFrame中填充
    for each_column in us_index_feature_series_dict:
        all_features_df[each_column] = np.nan
        temp_column = us_index_feature_series_dict[each_column]
        for each_date in all_features_df.index:
            selected_price = temp_column.loc[temp_column.index < each_date][-1]
            previous_day_check = all_features_df.loc[all_features_df.index < each_date, each_column]
            if each_column.split("_")[0] in ["sp500", "nasdaq", "dji", "gdc", "vix"] and (previous_day_check.size == 0 or previous_day_check[-1] == selected_price):
                continue
            all_features_df.loc[each_date, each_column] = selected_price
    
    # =========================================================================

    # 使用以前的策略选股数量作为feature。注意在这里需要再生成以09:45:00为起点的y_to_predict数据
    # TODO  # TODO
    
    # =========================================================================
    
    # 储存文件
    all_features_df.to_csv("all_features.csv")
    
#%%

# 生成a50的y_to_predict数据和相应的各种features
def generate_night_session_data():
    
    # 读取A50数据和沪深300数据
    a50_1m = pd.read_csv(os.path.join(raw_data_folder_address, "A50_1min.csv"))
    a50_1m = a50_1m.set_index("date_time")
    shsz300_1m = pd.read_csv(os.path.join(raw_data_folder_address, "SHSZ300_1min.csv"))
    shsz300_1m = shsz300_1m.set_index("date_time")
    
    # 生成y_to_predict。首先生成a50_night_within_day
    # 交易时间设置在北京时间晚上21:00:00到北京时间第二天早上05:15:00
    a50_within_day_selected_days = list(a50_1m[a50_1m.index.str.slice(11, 19) == "21:00:00"].index.str.slice(0, 10))
    a50_within_day = pd.Series(index = a50_within_day_selected_days)
    for each_day in a50_within_day_selected_days:
        next_day = str(datetime.datetime.strptime(each_day, "%Y-%m-%d") + datetime.timedelta(days = 1))[0:10]
        a50_within_day[each_day] = a50_1m.loc[next_day + " 05:15:00", "close"] / a50_1m.loc[each_day + " 21:00:00", "close"] - 1
    a50_within_day.name = "a50_within_day_night_session"
    
    # 生成feature：A50指数的夜盘盘前收益率
    # 设定四类：05:15:00-21:00:00，09:00:00-21:00:00，16:35:00-21:00:00，17:00:00-21:00:00
    before_open_return_type_one_series = pd.Series()
    before_open_return_type_one_series.name = "before_open_return_type_one"
    before_open_return_type_two_series = pd.Series()
    before_open_return_type_two_series.name = "before_open_return_type_two"
    before_open_return_type_three_series = pd.Series()
    before_open_return_type_three_series.name = "before_open_return_type_three"
    before_open_return_type_four_series = pd.Series()
    before_open_return_type_four_series.name = "before_open_return_type_four"
    type_one_start_price = a50_1m[a50_1m.index.str.slice(11, 19) == "05:15:00"]
    type_two_start_price = a50_1m[a50_1m.index.str.slice(11, 19) == "09:01:00"]
    type_three_start_price = a50_1m[a50_1m.index.str.slice(11, 19) == "16:35:00"]
    type_four_start_price = a50_1m[a50_1m.index.str.slice(11, 19) == "17:01:00"]
    finish_price = a50_1m[a50_1m.index.str.slice(11, 19) == "21:00:00"]
    for each_date in finish_price.index.str.slice(0, 10):
        temp_type_one_start_price = type_one_start_price.loc[type_one_start_price.index.str.slice(0, 10) <= each_date, "close"]
        temp_type_two_start_price = type_two_start_price.loc[type_two_start_price.index.str.slice(0, 10) == each_date, "open"]
        temp_type_three_start_price = type_three_start_price.loc[type_three_start_price.index.str.slice(0, 10) == each_date, "close"]
        temp_type_four_start_price = type_four_start_price.loc[type_four_start_price.index.str.slice(0, 10) == each_date, "open"]
        temp_finish_price = a50_1m.loc[each_date + " 21:00:00", "close"]
        if temp_type_one_start_price.size != 0:
            before_open_return_type_one_series[each_date] = temp_finish_price / temp_type_one_start_price[-1] - 1
        if temp_type_two_start_price.size != 0:
            before_open_return_type_two_series[each_date] = temp_finish_price / temp_type_two_start_price[-1] - 1
        if temp_type_three_start_price.size != 0:
            before_open_return_type_three_series[each_date] = temp_finish_price / temp_type_three_start_price[-1] - 1
        if temp_type_four_start_price.size != 0:
            before_open_return_type_four_series[each_date] = temp_finish_price / temp_type_four_start_price[-1] - 1
    
    # 生成feature：A50指数和SHSZ300指数的日盘盘中收益率
    # 时间范围设定在09:30:00-15:00:00
    a50_daytime_return = pd.Series()
    a50_daytime_return.name = "a50_daytime_return"
    shsz300_daytime_return = pd.Series()
    shsz300_daytime_return.name = "shsz300_daytime_return"
    selected_date = a50_1m[a50_1m.index.str.slice(11, 19) == "21:00:00"].index.str.slice(0, 10)
    for each_date in selected_date:
        a50_temp_data = a50_1m[(a50_1m.index >= each_date + " 09:31:00") & (a50_1m.index <= each_date + " 15:00:00")]
        shsz300_temp_data = shsz300_1m[(shsz300_1m.index >= each_date + " 09:31:00") & (shsz300_1m.index <= each_date + " 15:00:00")]
        if a50_temp_data.shape[0] > 0:
            a50_daytime_return[each_date] = a50_temp_data.loc[a50_temp_data.index.str.slice(11, 19) == "15:00:00", "close"][0] / a50_temp_data.loc[a50_temp_data.index.str.slice(11, 19) == "09:31:00", "open"][0] - 1
        if shsz300_temp_data.shape[0] > 0:
            shsz300_daytime_return[each_date] = shsz300_temp_data.loc[shsz300_temp_data.index.str.slice(11, 19) == "15:00:00", "close"][0] / shsz300_temp_data.loc[shsz300_temp_data.index.str.slice(11, 19) == "09:31:00", "open"][0] - 1

    # 生成feature：A50指数的盘前high_close_ratio和low_close_ratio
    # 时间范围设定在17:00:00-21:00:00
    before_open_high_close_series = pd.Series()
    before_open_high_close_series.name = "before_open_high_close_ratio"
    before_open_low_close_series = pd.Series()
    before_open_low_close_series.name = "before_open_low_close_ratio"
    selected_data = a50_1m[(a50_1m.index.str.slice(11, 19) >= "17:01:00") & (a50_1m.index.str.slice(11, 19) <= "21:00:00")]
    for each_date in selected_data.index.str.slice(0, 10).unique():
        temp_selected_data = selected_data[selected_data.index.str.slice(0, 10) == each_date]
        close_price = temp_selected_data["close"][-1]
        high_price = temp_selected_data["high"].max()
        low_price = temp_selected_data["low"].min()
        before_open_high_close_series[each_date] = high_price / close_price - 1
        before_open_low_close_series[each_date] = low_price / close_price - 1
    
    # 将数据结合起来并储存
    all_y_and_X_df = pd.concat([a50_within_day, before_open_return_type_one_series, before_open_return_type_two_series, before_open_return_type_three_series, before_open_return_type_four_series, a50_daytime_return, shsz300_daytime_return, before_open_high_close_series, before_open_low_close_series], axis = 1)
    all_y_and_X_df = all_y_and_X_df.sort_index()
    all_y_and_X_df.index.name = "date_time"
    all_y_and_X_df.to_csv("/home/liheng/Futures/data/y_and_X_df_night_session.csv")


generate_night_session_data()


