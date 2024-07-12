import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import geopy.distance
from tslearn.metrics import dtw

aqi_data = pd.read_csv('/home/zxh/code/dataprocessing/Beijing_aq-20181920-3.csv')
station_data = pd.read_excel('/home/zxh/code/MGFSTM-Remote/data/beijing_data-2018/station_aq.xlsx')
poi_data = pd.read_csv('/home/zxh/code/dataprocessing/PoiData/poi_one_hot_vectors_1.csv', header=None, index_col=0)  # 第一列作为索引
# 为AQI数据添加列名
aqi_data.columns = [f'station{i}' for i in range(0, aqi_data.shape[1])]

# 计算站点之间的地理距离
distances = []
for i in range(len(station_data)):
    station1_coords = (station_data.loc[i, 'latitude'], station_data.loc[i, 'longitude'])
    station2_coords = (station_data.loc[0, 'latitude'], station_data.loc[0, 'longitude'])
    distance = geopy.distance.geodesic(station1_coords, station2_coords).kilometers
    distances.append(distance)

# 计算皮尔逊相关系数
corr_values = []
for col in aqi_data.columns[0:]:
    corr, _ = pearsonr(aqi_data['station1'], aqi_data[col])
    corr_values.append(corr)

# 计算poi相似性
reference = poi_data.iloc[1, :]  # 第二个站点作为基准
poi_values = []
for station_name, station_data in poi_data.iterrows():
    correlation = reference.corr(station_data)
    poi_values.append(correlation)

print(distances)
print(corr_values)
print(poi_values)
