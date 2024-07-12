from typing import Any, Callable, Optional
import os
import torch
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp
import pandas as pd
from geopy.distance import geodesic
from pytorch_lightning.metrics.metric import Metric
from torch_geometric.utils import dense_to_sparse, get_laplacian, to_dense_adj


def get_L(W):
    edge_index, edge_weight = dense_to_sparse(W)
    edge_index, edge_weight = get_laplacian(edge_index, edge_weight)
    adj = to_dense_adj(edge_index, edge_attr=edge_weight)[0]
    return adj


def get_L_ASTGCN(W):
    edge_index, edge_weight = dense_to_sparse(W)
    edge_index, edge_weight = get_laplacian(edge_index, edge_weight, normalization="rw")
    adj = to_dense_adj(edge_index, edge_attr=edge_weight)[0]
    return adj

def calculate_random_walk_matrix(adj_mx):
    # Assuming adj_mx is a PyTorch tensor
    adj_mx = adj_mx.cpu().numpy()  # Convert to NumPy array
    adj_mx = sp.coo_matrix(adj_mx)  # Convert to a sparse matrix
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx

def get_L_d(W, device):
    # dynamic L_matrix 计算
    degree_matrix = 1 / (torch.sum(W, dim=2) + 0.0001)  # 有向图出度矩阵
    D = torch.diag_embed(degree_matrix)
    A = torch.matmul(D, W)
    l_d = torch.eye(D.shape[1], device=device) - A  # L = I − D~−1 W
    l_d = F.dropout(l_d, 0.5)
    return l_d


def get_L_d1(W, device):
    # dynamic L_matrix 计算
    degree_matrix = torch.sum(W, dim=2)  # 有向图出度矩阵
    D = torch.diag_embed(degree_matrix)
    l_d = D - W  # 拉普拉斯矩阵
    l_d_norm = F.normalize(l_d, p=2, dim=(1, 2))

    return l_d


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))

def masked_huber(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss_h = torch.abs(preds - labels)
    rho = 1
    loss = torch.where(loss_h > rho, rho * loss_h - 0.5 * (rho ** 2), 0.5 * (loss_h ** 2))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return np.round(mae, 4), np.round(mape, 4), np.round(rmse, 4)


def load_long_static_matrix(device):

    A_dist = torch.from_numpy(np.float32(np.load(os.path.join('data/matrix', 'dist.npy')))).to(device)
    A_func = torch.from_numpy(np.float32(np.load(os.path.join('data/matrix', 'func.npy')))).to(device)
    A_poi = torch.from_numpy(np.float32(np.load(os.path.join('data/matrix', 'poi.npy')))).to(device)
    matrix = A_poi + A_func + A_dist

    A_poi1 = indicator_function(A_poi, device)
    A_func1 = indicator_function(A_func, device)
    A_dist1 = indicator_function(A_dist, device)
    stacked = A_dist1 + A_poi1 + A_func1
    stacked[stacked == 0] = 1

    stack_matrix = matrix / stacked

    return stack_matrix


def indicator_function(matrix, device):
    # 检查矩阵中的元素是否为零，得到指示函数矩阵
    indicator_matrix = torch.where(matrix != 0, torch.tensor(1).to(device), torch.tensor(0).to(device))
    return indicator_matrix

def load_station_coordinates(file_path):
    # 从 Excel 文件中读取站点坐标信息
    station_data = pd.read_excel(file_path)
    latitudes = station_data['latitude'].values
    longitudes = station_data['longitude'].values
    return latitudes, longitudes


def compute_position_vectors(DEVICE):
    latitudes, longitudes = load_station_coordinates(
        os.path.join('data/beijing_data', 'station_aq.xlsx'))
    num_vertices = len(latitudes)
    position_vectors = torch.zeros(num_vertices, num_vertices, 2).to(DEVICE)
    distance_matrix = torch.zeros(num_vertices, num_vertices).to(DEVICE)

    for i in range(num_vertices):
        for j in range(num_vertices):
            if i != j:
                corrd_1 = (latitudes[i], longitudes[i])
                corrd_2 = (latitudes[j], longitudes[j])
                distance = geodesic(corrd_1, corrd_2).kilometers
                distance_matrix[i, j] = distance
                if distance < 30:
                    y = latitudes[j] - latitudes[i]
                    x = longitudes[j] - longitudes[i]
                    angle_rad = np.arctan2(y, x)
                    angle_deg = np.degrees(angle_rad) if angle_rad >= 0 else np.degrees(angle_rad) + 360
                    position_vectors[i, j] = torch.tensor(
                        [np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))]).to(DEVICE)
                else:
                    position_vectors[i, j] = 0
        distance_matrix[i, i] = 1

    return position_vectors, distance_matrix


class LightningMetric(Metric):

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("y_true", default=[], dist_reduce_fx=None)
        self.add_state("y_pred", default=[], dist_reduce_fx=None)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        self.y_pred.append(preds)
        self.y_true.append(target)

    def compute(self):
        """
        Computes explained variance over state.
        """
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)

        feature_dim = y_pred.shape[-1]
        pred_len = y_pred.shape[1]
        # (16, 12, 38, 1)

        y_pred = torch.reshape(y_pred.permute((0, 2, 1, 3)), (-1, pred_len, feature_dim))
        y_true = torch.reshape(y_true.permute((0, 2, 1, 3)), (-1, pred_len, feature_dim))

        # TODO: feature_dim, for multi-variable prediction, not only one.
        y_pred = y_pred[..., 0]
        y_true = y_true[..., 0]

        metric_dict = {}
        rmse_avg = []
        mae_avg = []
        mape_avg = []

        group_size = 6  # revise

        for i in range(pred_len):
            mae, mape, rmse = metric(y_pred[:, i], y_true[:, i])
            idx = i + 1

            metric_dict.update({'rmse_%s' % idx: rmse})
            metric_dict.update({'mae_%s' % idx: mae})
            metric_dict.update({'mape_%s' % idx: mape})

            group_idx = (idx - 1) % group_size  # 计算当前时间步属于哪个组
            if group_idx == 0:
                # 新的一组，初始化组内列表
                group_mae = []
                group_mape = []
                group_rmse = []
            group_mae.append(mae)
            group_mape.append(mape)
            group_rmse.append(rmse)

            if group_idx == group_size - 1 or idx == pred_len:  # 6个为一组，计算平均值
                # 当前组的最后一个时间步，或者是最后一个时间步
                group_num = (idx - 1) // group_size + 1
                metric_dict.update({'rmse_group_%s' % group_num: np.round(np.mean(group_rmse), 4)})
                metric_dict.update({'mae_group_%s' % group_num: np.round(np.mean(group_mae), 4)})
                metric_dict.update({'mape_group_%s' % group_num: np.round(np.mean(group_mape), 4)})

            rmse_avg.append(rmse)
            mae_avg.append(mae)
            mape_avg.append(mape)

        metric_dict.update({'rmse_avg': np.round(np.mean(rmse_avg), 4)})
        metric_dict.update({'mae_avg': np.round(np.mean(mae_avg), 4)})
        metric_dict.update({'mape_avg': np.round(np.mean(mape_avg), 4)})

        return metric_dict


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


if __name__ == '__main__':

    lightning_metric = LightningMetric()
    batches = 10
    for i in range(batches):
        preds = torch.randn(32, 24, 38, 1)
        target = preds + 0.15

        rmse_batch = lightning_metric(preds, target)
        print(f"Metrics on batch {i}: {rmse_batch}")

    rmse_epoch = lightning_metric.compute()
    print(f"Metrics on all data: {rmse_epoch}")

    lightning_metric.reset()
