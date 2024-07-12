import numpy as np
import pandas as pd
from geopy.distance import geodesic
# from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tslearn.metrics import dtw
from scipy.stats import pearsonr
# from dtaidistance import dtw


def generate_dis_matrix():
    # 读取原始数据集，该数据集应包含站点名称、纬度和经度等信息
    data_file = '/home/zxh/code/MGFSTM-Remote/data/beijing_data-2018/station_aq.xlsx'
    data = pd.read_excel(data_file)
    # 提取站点名称、纬度和经度信息
    station_names = data['station_name']
    latitudes = data['latitude']
    longitudes = data['longitude']

    num_stations = len(station_names)
    # # 定义高斯核的参数 σ
    # sigma = 10.0  # 调整这个值以控制权重分布的宽度
    # 初始化邻接矩阵
    adj_matrix = np.zeros((num_stations, num_stations))

    # 计算距离的标准差
    distances = np.zeros((num_stations, num_stations))
    for i in range(num_stations):
        for j in range(i + 1, num_stations):
            coords_1 = (latitudes[i], longitudes[i])
            coords_2 = (latitudes[j], longitudes[j])
            distance = geodesic(coords_1, coords_2).kilometers
            distances[i, j] = distance
            distances[j, i] = distance
    # 高斯核
    sigma = np.std(distances)

    # 遍历每对站点计算距离并更新邻接矩阵
    for i in range(num_stations):
        for j in range(i + 1, num_stations):
            coords_1 = (latitudes[i], longitudes[i])
            coords_2 = (latitudes[j], longitudes[j])
            distance = geodesic(coords_1, coords_2).kilometers
            weight = np.exp(-((distance ** 2) / (2 * sigma ** 2)))
            if distance < 50:
                adj_matrix[i, j] = weight
                adj_matrix[j, i] = weight
    # 不需要归一化
    np.fill_diagonal(adj_matrix, 1.0)

    # 保存权重矩阵为.npy文件
    weight_matrix_filename = 'D:/SFstlstm/MGFSTM/data/matrix/dis_matrix.npy'
    np.save(weight_matrix_filename, adj_matrix)

# 这是处理uci数据集的部分
def generate_sim_matrix_uci():
    # 读取补充完整的数据集
    data_file = 'D:/SFstlstm/MGFSTM/data/data_aq-2017/data.csv'
    data = pd.read_csv(data_file, index_col='timestamp')
    # 获取站点名称列表
    station_names = data.iloc[:, 0].unique()
    # 初始化阈值和邻接矩阵
    threshold = 0  # 调整阈值以控制边的生成
    num_stations = len(station_names)
    adj_matrix = np.zeros((num_stations, num_stations))

    # 遍历每对站点计算相似性并更新邻接矩阵
    for i in range(num_stations):
        for j in range(i + 1, num_stations):
            series1 = data[data['station_name'] == station_names[i]]['PM2.5'].values
            series2 = data[data['station_name'] == station_names[j]]['PM2.5'].values
            a = dtw(series1, series2)
            similarity = 1.0 / (1.0 + dtw(series1, series2))
            # similarity = dtw.distance(series1, series2)
            if similarity > threshold:
                adj_matrix[i, j] = similarity
                adj_matrix[j, i] = similarity
    # 归一化
    adj_matrix_normalized = adj_matrix / np.max(adj_matrix)
    # 对角线元素设为 1
    np.fill_diagonal(adj_matrix_normalized, 1.0)

    # 保存邻接矩阵为.npy文件
    adj_matrix_filename = 'D:/SFstlstm/MGFSTM/data/matrix/sim_matrix.npy'
    np.save(adj_matrix_filename, adj_matrix_normalized)


def generate_sim_matrix():

    data_files = '/home/zxh/code/MGFSTM-Remote/data/beijing_data-2018/beijing_aq-2018.csv'
    data = pd.read_csv(data_files)
    # 除去前两列，取出其他列名
    station_columns = data.columns[2:]
    stations = list(station_columns)

    threshold = 0.5  # 可以设置
    # 选择type类型的数据
    aqi_data = data[data['type'] == 'PM2.5'][stations].values

    num_stations = len(stations)
    adj_matrix = np.zeros((num_stations, num_stations))

    for i in range(num_stations):
        for j in range(i+1, num_stations):
            series1 = aqi_data[:, i]
            series2 = aqi_data[:, j]
            # a = dtw(series1, series2)
            # similarity = 1.0/(1.0 + dtw(series1, series2))
            # if similarity > threshold:
            #     adj_matrix[i, j] = similarity
            #     adj_matrix[j, i] = similarity
            similarity, _ = pearsonr(series1, series2)
            if similarity > threshold:
                adj_matrix[i, j] = similarity
                adj_matrix[j, i] = similarity

    # 归一化
    adj_matrix_normalized = adj_matrix / (np.max(adj_matrix))
    np.fill_diagonal(adj_matrix_normalized, 1.0)

    adj_matrix_filename = '/home/zxh/code/MGFSTM-Remote/data/matrix/func.npy'
    np.save(adj_matrix_filename, adj_matrix_normalized)

# 生成区域相似图
def generation_poi_matrix():
    poi_data = pd.read_csv('/home/zxh/code/dataprocessing/PoiData/poi_one_hot_vectors_1.csv', header=None, index_col=0)
    num_stations = 34
    adjacency_matrix = np.zeros((num_stations, num_stations))

    for i in range(num_stations):
        for j in range(i+1, num_stations):
            series1 = poi_data.iloc[i]
            series2 = poi_data.iloc[j]
            corr = np.corrcoef(series1, series2)[0, 1]
            adjacency_matrix[i, j] = corr
            adjacency_matrix[j, i] = corr

    np.fill_diagonal(adjacency_matrix, 1.0)
    adjacency_matrix_file = '/home/zxh/code/MGFSTM-Remote/data/matrix/poi.npy'
    np.save(adjacency_matrix_file, adjacency_matrix)


# ******这是生成其邻居矩阵的部分 ******

def generate_nerghi_matrix():
    # 读取包含站点信息的CSV文件
    data = pd.read_csv('D:/SFstlstm/MGFSTM/data/beijing_data-2018/station_aq.csv')

    # 获取站点名称、纬度和经度列
    station_names = data['station_name'].tolist()
    latitudes = data['latitude'].tolist()
    longitudes = data['longitude'].tolist()

    # 初始化邻接矩阵
    num_stations = len(station_names)
    adjacency_matrix = np.zeros((num_stations, num_stations))

    for i in range(num_stations):
        coords_1 = (latitudes[i], longitudes[i])
        min_distance = float('inf')
        for j in range(i + 1, num_stations):
            coords_2 = (latitudes[j], longitudes[j])
            distance = geodesic(coords_1, coords_2).kilometers

            if distance <= min_distance:
                min_distance = distance
                adjacency_matrix[i, j] = 1.0
                adjacency_matrix[j, i] = 1.0

    adjacency_matrix_filename = 'D:/SFstlstm/MGFSTM/data/matrix/neigh.npy'
    np.save(adjacency_matrix_filename, adjacency_matrix)
    # 输出邻接矩阵
    print("Adjacency Matrix:")
    print(adjacency_matrix)

if __name__ == "__main__":
    # generate_dis_matrix()  # 调用生成邻接矩阵的函数
    # generate_sim_matrix_uci()
    # generate_sim_matrix()
    generation_poi_matrix()
    # generate_nerghi_matrix()