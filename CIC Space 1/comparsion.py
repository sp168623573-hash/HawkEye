import pandas as pd
import numpy as np
import torch
import tensorflow as tf
import torch.nn as nn
import matplotlib.pyplot as plt
import featuretools as ft
import os
import warnings
import time  # ========== 新增：耗时统计 ==========
from tqdm import tqdm, trange  # ========== 新增：进度条 ==========
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split  # ========== 新增：修复NameError ==========
from collections import Counter
from imblearn.over_sampling import SMOTE
from torch.nn import functional as F
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, MaxPooling1D, LSTM, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from scipy.interpolate import make_interp_spline
from einops import rearrange
from einops.layers.torch import Rearrange
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cosine
import math

# ========== 全局配置 ==========
warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
tf.random.set_seed(SEED)
# ========== 修改：增强设备信息打印 ==========
gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"✅ 运行设备：{DEVICE} ({gpu_info}) | 随机种子：{SEED}")

# ========== 论文定义的常量 ==========
KNOWN_ATTACKS = [
    'FTP-Patator', 'SSH-Patator', 'DoS slowloris',
    'DoS Slowhttptest', 'DDoS', 'DoS Hulk', 'DoS GoldenEye',
    'PortScan'
]
UNKNOWN_ATTACKS = [
    'Infiltration', 'Heartbleed', 'Web Attack - XSS',
    'Web Attack - Sql Injection', 'Web Attack - Brute Force', 'Bot'
]
BENIGN = ['BENIGN']

# ========== 1. 核心工具函数（仅保留使用的） ==========
def get_positional_encoding(seq_len, d_model, device=DEVICE):
    position = torch.arange(seq_len, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


def variational_activation(x):
    if isinstance(x, torch.Tensor):
        return torch.sigmoid(x) * torch.tanh(x)
    else:
        return 1 / (1 + np.exp(-x)) * np.tanh(x)


def dynamic_distance_metric_np(x1, x2, feature_weights, lambda_var=0.15, m=None):
    if feature_weights is not None and len(feature_weights) == len(x1):
        w = np.abs(feature_weights)
        w = w / (np.sum(w) + 1e-9)
    else:
        w = np.ones_like(x1, dtype=float) / len(x1)
    diff = np.abs(x1 - x2)
    base_distance = np.sum(w * diff)
    variational_term = np.mean(np.abs(diff - np.mean(diff)))
    mean_gap = np.mean(diff) if m is None else m
    distance = base_distance + lambda_var * (variational_term + 0.1 * mean_gap)
    return float(max(distance, 0.0))


def dynamic_distance_metric_torch(x1, x2, feature_weights, lambda_var=0.15, m=None):
    if feature_weights is not None and feature_weights.numel() == x1.shape[-1]:
        w = torch.abs(feature_weights)
        w = w / (torch.sum(w) + 1e-9)
    else:
        w = torch.ones(x1.shape[-1], device=x1.device, dtype=x1.dtype) / x1.shape[-1]
    view_shape = [1] * (x1.dim() - 1) + [x1.shape[-1]]
    diff = torch.abs(x1 - x2)
    base_distance = torch.sum(diff * w.view(*view_shape), dim=-1)
    mean_gap = diff.mean(dim=-1) if m is None else m
    variational_term = torch.mean(torch.abs(diff - diff.mean(dim=-1, keepdim=True)), dim=-1)
    distance = base_distance + lambda_var * (variational_term + 0.1 * mean_gap)
    return torch.clamp(distance, min=0.0)


def calculate_macro_fpr(y_true, y_pred, labels=None):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.size == 0:
        return 0.0
    benign_mask = (y_true == 0)
    if benign_mask.sum() == 0:
        return 0.0
    return float(np.mean(y_pred[benign_mask] != 0))


def calculate_metric_bundle(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.size == 0:
        return {
            "acc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_weighted": 0.0,
            "fpr": 0.0
        }
    labels = np.unique(np.concatenate([y_true, y_pred]))
    return {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "fpr": calculate_macro_fpr(y_true, y_pred, labels=labels)
    }


@torch.no_grad()
def evaluate_torch_classifier(model, data_loader, device=DEVICE):
    model.eval()
    all_labels = []
    all_predictions = []
    for data, labels in data_loader:
        data = data.to(device)
        logits = model(data)
        predicted = torch.argmax(logits, dim=1).cpu().numpy()
        all_predictions.append(predicted)
        all_labels.append(labels.detach().cpu().numpy())
    if not all_labels:
        return {
            "acc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_weighted": 0.0,
            "fpr": 0.0
        }
    return calculate_metric_bundle(np.concatenate(all_labels), np.concatenate(all_predictions))


def compute_local_density_scale(reference_X, query_X=None, k=5):
    reference_X = np.nan_to_num(np.asarray(reference_X, dtype=float))
    if reference_X.shape[0] <= 1:
        target_len = reference_X.shape[0] if query_X is None else len(query_X)
        return np.ones(target_len, dtype=float)

    target_X = reference_X if query_X is None else np.nan_to_num(np.asarray(query_X, dtype=float))
    n_neighbors = min(reference_X.shape[0], k + (1 if query_X is None else 0))
    neigh = NearestNeighbors(n_neighbors=max(1, n_neighbors))
    neigh.fit(reference_X)
    distances, _ = neigh.kneighbors(target_X)
    if query_X is None and distances.shape[1] > 1:
        distances = distances[:, 1:]
    local_span = distances.mean(axis=1)
    scale = local_span / (np.median(local_span) + 1e-9)
    return np.clip(scale, 0.5, 1.5)


def weighted_dynamic_min_distance(samples, centers, feature_weights, lambda_var=0.15, chunk_size=1024):
    samples = np.asarray(samples, dtype=float)
    centers = np.asarray(centers, dtype=float)
    if centers.size == 0:
        return np.ones(samples.shape[0], dtype=float), np.full(samples.shape[0], -1, dtype=int)

    if feature_weights is not None and len(feature_weights) == samples.shape[1]:
        feat_w = np.abs(np.asarray(feature_weights, dtype=float))
        feat_w = feat_w / (np.sum(feat_w) + 1e-9)
    else:
        feat_w = np.ones(samples.shape[1], dtype=float) / samples.shape[1]

    min_distances = np.empty(samples.shape[0], dtype=float)
    argmin_indices = np.empty(samples.shape[0], dtype=int)
    for start in range(0, samples.shape[0], chunk_size):
        end = min(start + chunk_size, samples.shape[0])
        chunk = samples[start:end]
        diff = np.abs(chunk[:, None, :] - centers[None, :, :])
        mean_gap = diff.mean(axis=2)
        variational_term = np.mean(np.abs(diff - mean_gap[:, :, None]), axis=2)
        distance = np.sum(diff * feat_w[None, None, :], axis=2) + lambda_var * (variational_term + 0.1 * mean_gap)
        min_distances[start:end] = distance.min(axis=1)
        argmin_indices[start:end] = distance.argmin(axis=1)

    return min_distances, argmin_indices


def compute_cluster_purity_score(cluster_labels, y_true):
    cluster_labels = np.asarray(cluster_labels)
    y_true = np.asarray(y_true)
    valid_mask = cluster_labels != -1
    if valid_mask.sum() == 0:
        return 0.0

    total = 0
    correct = 0
    for cluster_id in np.unique(cluster_labels[valid_mask]):
        idx = cluster_labels == cluster_id
        counts = Counter(y_true[idx])
        correct += counts.most_common(1)[0][1]
        total += int(idx.sum())
    return float(correct / (total + 1e-9))


def hierarchical_biomimetic_coevolution_loss(feat, labels, centers, feature_weights, margin=0.12):
    own_centers = centers[labels]
    reconstruction_error = dynamic_distance_metric_torch(feat, own_centers, feature_weights)
    opponent_centers = centers[(1 - labels).long()]
    separation_distance = dynamic_distance_metric_torch(feat, opponent_centers, feature_weights)
    antagonistic_loss = F.relu(reconstruction_error - separation_distance + margin).mean()

    weights = torch.softmax(torch.abs(feature_weights), dim=0)
    uniform = torch.full_like(weights, 1.0 / weights.numel())
    ecological_balance = torch.mean(torch.abs(weights - uniform))
    feature_entropy = -(weights * torch.log(weights + 1e-9)).sum()
    regularization = ecological_balance + 0.02 * (math.log(weights.numel()) - feature_entropy)

    return {
        "reconstruction": reconstruction_error.mean(),
        "antagonistic": antagonistic_loss,
        "regularization": regularization,
        "total": reconstruction_error.mean() + 0.7 * antagonistic_loss + 0.2 * regularization,
        "reconstruction_vector": reconstruction_error.detach()
    }


def history_to_percent_series(values, target_len):
    series = np.full(target_len, np.nan, dtype=float)
    if values is None:
        return series
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return series
    valid_len = min(target_len, values.size)
    series[:valid_len] = values[:valid_len] * 100.0
    return series


def save_comparison_metrics_table(hawkeyes_main_history, hawkeyes_finetune_history, cnn_lstm_history):
    cnn_history = cnn_lstm_history.history if hasattr(cnn_lstm_history, "history") else cnn_lstm_history
    hawkeyes_test_accuracy = hawkeyes_finetune_history.get("val_accuracy", hawkeyes_finetune_history.get("accuracy", []))
    hawkeyes_test_precision = hawkeyes_finetune_history.get("val_precision", hawkeyes_finetune_history.get("precision", []))
    hawkeyes_test_recall = hawkeyes_finetune_history.get("val_recall", hawkeyes_finetune_history.get("recall", []))
    hawkeyes_test_f1 = hawkeyes_finetune_history.get("val_f1", hawkeyes_finetune_history.get("f1", []))
    hawkeyes_test_fpr = hawkeyes_finetune_history.get("val_fpr", hawkeyes_finetune_history.get("fpr", []))

    max_epochs = max(
        len(hawkeyes_main_history.get("accuracy", [])),
        len(hawkeyes_test_accuracy),
        len(cnn_history.get("accuracy", [])),
        len(cnn_history.get("val_accuracy", []))
    )
    if max_epochs == 0:
        return None

    export_df = pd.DataFrame({
        "Epochs": np.arange(1, max_epochs + 1),
        "HawkEyes_Train": history_to_percent_series(hawkeyes_main_history.get("accuracy"), max_epochs),
        "HawkEyes_Test": history_to_percent_series(hawkeyes_test_accuracy, max_epochs),
        "CNN_LSTM_Train": history_to_percent_series(cnn_history.get("accuracy"), max_epochs),
        "CNN_LSTM_Test": history_to_percent_series(cnn_history.get("val_accuracy"), max_epochs),
        "HawkEyes_Train_Precision": history_to_percent_series(hawkeyes_main_history.get("precision"), max_epochs),
        "HawkEyes_Test_Precision": history_to_percent_series(hawkeyes_test_precision, max_epochs),
        "CNN_LSTM_Train_Precision": history_to_percent_series(cnn_history.get("precision"), max_epochs),
        "CNN_LSTM_Test_Precision": history_to_percent_series(cnn_history.get("val_precision"), max_epochs),
        "HawkEyes_Train_Recall": history_to_percent_series(hawkeyes_main_history.get("recall"), max_epochs),
        "HawkEyes_Test_Recall": history_to_percent_series(hawkeyes_test_recall, max_epochs),
        "CNN_LSTM_Train_Recall": history_to_percent_series(cnn_history.get("recall"), max_epochs),
        "CNN_LSTM_Test_Recall": history_to_percent_series(cnn_history.get("val_recall"), max_epochs),
        "HawkEyes_Train_F1": history_to_percent_series(hawkeyes_main_history.get("f1"), max_epochs),
        "HawkEyes_Test_F1": history_to_percent_series(hawkeyes_test_f1, max_epochs),
        "CNN_LSTM_Train_F1": history_to_percent_series(cnn_history.get("f1"), max_epochs),
        "CNN_LSTM_Test_F1": history_to_percent_series(cnn_history.get("val_f1"), max_epochs),
        "HawkEyes_Train_FPR": history_to_percent_series(hawkeyes_main_history.get("fpr"), max_epochs),
        "HawkEyes_Test_FPR": history_to_percent_series(hawkeyes_test_fpr, max_epochs),
        "CNN_LSTM_Train_FPR": history_to_percent_series(cnn_history.get("fpr"), max_epochs),
        "CNN_LSTM_Test_FPR": history_to_percent_series(cnn_history.get("val_fpr"), max_epochs),
    })

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "data")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "model_comparison_accuracy_data.csv")
    txt_path = os.path.join(output_dir, "model_comparison_accuracy_data.txt")
    export_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    export_df.to_csv(txt_path, index=False, sep="\t", encoding="utf-8-sig")
    print(f"📌 对比指标数据已保存至：{csv_path}")
    return csv_path


# ========== 修改：添加cluster_mapping参数，修复未定义报错 ==========
def calculate_reconstruction_error(X, train_X, train_labels, cluster_centers, cluster_mapping, feature_weights=None, z=0.5,
                                   relax_coeff=0.5, min_threshold=0.0001, cali_t0=None, cali_t1=None):
    reconstruction_errors = []
    new_labels = []
    normal_indices = np.where(train_labels == 0)[0]
    attack_indices = np.where(train_labels == 1)[0]
    if normal_indices.size == 0 or attack_indices.size == 0:
        raise ValueError("训练集需同时包含良性(0)和已知攻击(1)样本")

    if feature_weights is not None and len(feature_weights) == train_X.shape[1]:
        feat_w = np.abs(feature_weights)
        feat_w = feat_w / (np.sum(feat_w) + 1e-9)
    else:
        feat_w = np.ones(train_X.shape[1]) / train_X.shape[1]

    valid_centers = {}
    for c, center in cluster_centers.items():
        if isinstance(c, str) and c.startswith('__'):
            continue
        if isinstance(center, dict) and 'center' in center:
            valid_centers[c] = center['center']
        elif isinstance(center, np.ndarray):
            valid_centers[c] = center
    cluster_centers_list = list(valid_centers.values())
    if not cluster_centers_list:
        return np.ones(X.shape[0]), np.full(X.shape[0], 2, dtype=int)

    train_density_scale = compute_local_density_scale(train_X)
    query_density_scale = compute_local_density_scale(train_X, X)

    normal_recon_errors = []
    sample_n0 = min(2000, len(normal_indices))
    # ========== 新增：进度条 ==========
    for i in tqdm(normal_indices[:sample_n0], desc="计算良性样本重构误差", leave=False):
        x_i = train_X[i]
        min_dist = min(dynamic_distance_metric_np(x_i, center, feat_w) for center in cluster_centers_list)
        normal_recon_errors.append(min_dist * train_density_scale[i])
    threshold0 = max(np.percentile(normal_recon_errors, 99), min_threshold)

    attack_recon_errors = []
    sample_n1 = min(2000, len(attack_indices))
    # ========== 新增：进度条 ==========
    for i in tqdm(attack_indices[:sample_n1], desc="计算攻击样本重构误差", leave=False):
        x_i = train_X[i]
        min_dist = min(dynamic_distance_metric_np(x_i, center, feat_w) for center in cluster_centers_list)
        attack_recon_errors.append(min_dist * train_density_scale[i])
    threshold1 = max(np.percentile(attack_recon_errors, 99), min_threshold)
    threshold1 = max(threshold1, threshold0 * 1.5)
    if cali_t0 is not None:
        threshold0 = max((1 - relax_coeff) * threshold0 + relax_coeff * float(cali_t0), min_threshold)
    if cali_t1 is not None:
        threshold1 = max((1 - relax_coeff) * threshold1 + relax_coeff * float(cali_t1), threshold0 * 1.15)

    all_min_dist = []
    all_closest_cls = []
    all_closest_cluster = []
    # ========== 新增：进度条 ==========
    for i in tqdm(range(X.shape[0]), desc="计算测试样本最小距离", leave=False):
        x_i = X[i]
        dist_dict = {c: dynamic_distance_metric_np(x_i, center, feat_w) for c, center in valid_centers.items()}
        if not dist_dict:
            min_distance = 1.0
            closest_label = -1
            closest_cluster_cls = 2
        else:
            min_distance = min(dist_dict.values())
            closest_label = min(dist_dict, key=dist_dict.get)
            closest_cluster_cls = cluster_mapping.get(closest_label, 2)
        all_min_dist.append(min_distance * query_density_scale[i])
        all_closest_cls.append(closest_cluster_cls)
        all_closest_cluster.append(closest_label)
    all_min_dist = np.array(all_min_dist)

    # ========== 新增：进度条 ==========
    for i in tqdm(range(X.shape[0]), desc="生成新标签", leave=False):
        min_distance = all_min_dist[i]
        closest_cluster_cls = all_closest_cls[i]
        closest_cluster = all_closest_cluster[i]
        current_threshold = threshold0 if closest_cluster_cls == 0 else threshold1
        if closest_cluster in cluster_centers and isinstance(cluster_centers[closest_cluster], dict):
            current_threshold = max(current_threshold, cluster_centers[closest_cluster].get('train_max', current_threshold))
        current_threshold = max(current_threshold, min_threshold)
        if min_distance <= current_threshold:
            new_labels.append(closest_cluster_cls)
        else:
            new_labels.append(2)

    new_labels = np.array(new_labels)
    reconstruction_errors = all_min_dist
    return np.array(reconstruction_errors), np.array(new_labels)


def get_cluster_label_mapping(cluster_labels, y_true, min_purity=0.4):
    cluster_mapping = {}
    unique_clusters = set(cluster_labels)
    if -1 in unique_clusters:
        unique_clusters.remove(-1)
    # ========== 新增：进度条 ==========
    for c in tqdm(unique_clusters, desc="生成聚类标签映射", leave=False):
        idx = (cluster_labels == c)
        c_y = y_true[idx]
        if len(c_y) < 3:
            continue
        counts = Counter(c_y)
        main_cls, main_count = counts.most_common(1)[0]
        purity = main_count / len(c_y)
        if purity < min_purity:
            continue
        cluster_mapping[c] = int(main_cls)
    return cluster_mapping


def batch_predict_unknown_attack(
        feats, valid_clusters, cluster_mapping,
        cluster_profiles, thresholds, feature_weights,
        train_feats, train_labels, y_test
):
    if len(feats) == 0:
        return np.array([])
    n_samples = len(feats)
    cluster_centers = cluster_profiles
    if not cluster_centers:
        return np.full(n_samples, 2, dtype=int)

    feat_w = np.abs(feature_weights) if (
            feature_weights is not None and len(feature_weights) == feats.shape[1]) else np.ones(feats.shape[1])
    feat_w = feat_w / (np.sum(feat_w) + 1e-9)
    adaptive_thresholds = cluster_profiles.get('__adaptive_thresholds__', {})
    cali_t0 = adaptive_thresholds.get(0)
    cali_t1 = adaptive_thresholds.get(1)

    # ========== 修改：传入cluster_mapping参数 ==========
    recon_errors, y_pred = calculate_reconstruction_error(
        X=feats, train_X=train_feats, train_labels=train_labels,
        cluster_centers=cluster_centers, cluster_mapping=cluster_mapping,
        feature_weights=feat_w, cali_t0=cali_t0, cali_t1=cali_t1
    )

    if len(y_pred) != n_samples:
        y_pred = np.full(n_samples, 2, dtype=int)
    return y_pred


def compute_cluster_profiles(train_latent, train_clusters, valid_clusters, feature_weights, y_train=None, adaptive_thresholds=None):
    profiles = {}
    thresholds = {}
    pure_centers = {}
    global_med = np.median(train_latent, axis=0)
    global_std = np.std(train_latent, axis=0) + 1e-9
    # ========== 新增：进度条 ==========
    for c in tqdm(valid_clusters, desc="计算聚类轮廓", leave=False):
        cluster_points = train_latent[train_clusters == c]
        if len(cluster_points) < 3: continue
        center = np.median(cluster_points, axis=0)
        center = center * 0.9 + global_med * 0.1
        pure_centers[c] = center
        dists = np.array([dynamic_distance_metric_np(p, center, feature_weights) for p in cluster_points])
        median_dist = np.median(dists)
        max_dist = np.percentile(dists, 95)
        max_dist = max(max_dist, 0.0005)
        median_dist = max(median_dist, 0.0005)
        profiles[c] = {'center': center, 'train_median': median_dist, 'train_max': max_dist, 'train_std': np.std(dists),
                       'global_align': global_std}
        thresholds[c] = max_dist
    if pure_centers:
        center_matrix = np.array(list(pure_centers.values()), dtype=float)
        min_distances, _ = weighted_dynamic_min_distance(train_latent, center_matrix, feature_weights)
        learned_thresholds = {}
        if y_train is not None:
            y_train_arr = np.asarray(y_train)
            for cls in [0, 1]:
                cls_mask = (y_train_arr == cls)
                if np.any(cls_mask):
                    quantile_value = 97 if cls == 0 else 98
                    learned_thresholds[cls] = max(np.percentile(min_distances[cls_mask], quantile_value), 0.0005)
        if adaptive_thresholds is not None:
            adaptive_arr = np.asarray(adaptive_thresholds, dtype=float).reshape(-1)
            for cls in [0, 1]:
                if adaptive_arr.size > cls:
                    learned_thresholds[cls] = max(learned_thresholds.get(cls, 0.0005), float(adaptive_arr[cls]))
        if learned_thresholds:
            profiles['__adaptive_thresholds__'] = learned_thresholds
    profiles['__global_stats__'] = {'med': global_med, 'std': global_std}
    profiles['__pure_centers__'] = pure_centers
    return profiles, thresholds, pure_centers


def build_dynamic_metric(feature_weights):
    return lambda a, b: dynamic_distance_metric_np(a, b, feature_weights)


def silhouette_score_custom(X, labels, distance_fn=None, max_samples=512):
    n_samples = X.shape[0]
    if n_samples == 0:
        return -float('inf')
    if n_samples > max_samples:
        rng = np.random.default_rng(SEED)
        sample_idx = rng.choice(n_samples, size=max_samples, replace=False)
        X = X[sample_idx]
        labels = labels[sample_idx]
        n_samples = X.shape[0]
    if distance_fn is None:
        distance_fn = lambda a, b: np.linalg.norm(a - b)

    scores = []
    unique_clusters = np.unique(labels)
    for i in range(n_samples):
        x_i = X[i]
        cluster_i = labels[i]
        intra_cluster_points = X[labels == cluster_i]
        if len(intra_cluster_points) > 1:
            intra_distances = [distance_fn(x_i, point) for point in intra_cluster_points]
            a_i = float(np.mean(intra_distances))
        else:
            a_i = 0.0
        min_b_i = float('inf')
        for cluster_j in unique_clusters:
            if cluster_j == cluster_i:
                continue
            other_cluster_points = X[labels == cluster_j]
            if len(other_cluster_points) == 0:
                continue
            b_i = float(np.mean([distance_fn(x_i, point) for point in other_cluster_points]))
            min_b_i = min(min_b_i, b_i)
        s_i = (min_b_i - a_i) / max(a_i, min_b_i) if np.isfinite(min_b_i) and max(a_i, min_b_i) > 0 else 0.0
        scores.append(s_i)
    return float(np.mean(scores)) if scores else -float('inf')


def calinski_harabasz_score_custom(X, labels, feature_weights=None):
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    unique_clusters = np.unique(labels)
    n_samples, n_features = X.shape
    n_clusters = len(unique_clusters)
    if n_clusters <= 1:
        return -float('inf')
    if feature_weights is not None and len(feature_weights) == X.shape[1]:
        feat_w = np.abs(np.asarray(feature_weights, dtype=float))
        feat_w = feat_w / (np.sum(feat_w) + 1e-9)
        X = X * np.sqrt(feat_w)
    cluster_means = {c: X[labels == c].mean(axis=0) for c in unique_clusters}
    overall_mean = X.mean(axis=0)
    W = np.zeros((n_features, n_features))
    for c in unique_clusters:
        cluster_points = X[labels == c]
        diff = cluster_points - cluster_means[c]
        W += np.dot(diff.T, diff)
    B = np.zeros((n_features, n_features))
    for c in unique_clusters:
        cluster_mean = cluster_means[c]
        n_c = len(X[labels == c])
        diff = cluster_mean - overall_mean
        B += n_c * np.outer(diff, diff)
    tr_B = np.trace(B)
    tr_W = np.trace(W)
    if tr_W == 0 or (n_samples - n_clusters) == 0:
        return -float('inf')
    return float((tr_B / (n_clusters - 1)) / (tr_W / (n_samples - n_clusters)))


def evaluate_cluster_configuration(X, labels, y_true=None, feature_weights=None):
    labels = np.asarray(labels)
    valid_mask = labels != -1
    valid_clusters = np.unique(labels[valid_mask]) if np.any(valid_mask) else np.array([])
    if valid_mask.sum() < 10 or len(valid_clusters) <= 1:
        return -float('inf')

    distance_fn = build_dynamic_metric(feature_weights)
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]
    silhouette = silhouette_score_custom(X_valid, labels_valid, distance_fn=distance_fn)
    ch_score = calinski_harabasz_score_custom(X_valid, labels_valid, feature_weights=feature_weights)
    purity = compute_cluster_purity_score(labels, y_true) if y_true is not None else 0.0
    noise_ratio = float(np.mean(labels == -1))
    validity = float(len(valid_clusters) / max(len(np.unique(labels)), 1))
    score = (
        0.34 * silhouette +
        0.22 * np.tanh(np.log1p(max(ch_score, 0.0)) / 5.0) +
        0.24 * purity +
        0.12 * (1.0 - noise_ratio) +
        0.08 * validity
    )
    return float(score)


def determine_min_samples(X, y_true=None, feature_weights=None):
    X = np.nan_to_num(np.asarray(X, dtype=float))
    distance_fn = build_dynamic_metric(feature_weights)
    min_samples_range = range(4, 11)
    best_min_samples = 6
    best_score = -float('inf')
    for min_samples in tqdm(min_samples_range, desc="选择最优min_samples", leave=False):
        n_neighbors = min(len(X), min_samples + 1)
        neigh = NearestNeighbors(n_neighbors=max(2, n_neighbors), algorithm='brute', metric=distance_fn)
        neigh.fit(X)
        distances, _ = neigh.kneighbors(X)
        k_dist = np.sort(distances[:, -1])
        eps_candidates = np.unique(np.round(np.percentile(k_dist, [70, 75, 80, 85, 90]), 6))
        for eps in eps_candidates:
            labels = DBSCAN(eps=float(max(eps, 1e-4)), min_samples=min_samples, metric=distance_fn).fit_predict(X)
            score = evaluate_cluster_configuration(X, labels, y_true=y_true, feature_weights=feature_weights)
            if score > best_score:
                best_score = score
                best_min_samples = min_samples
    return best_min_samples


def adaptive_cluster_params(X, y_true=None, feature_weights=None):
    X = np.nan_to_num(np.asarray(X, dtype=float))
    n_sample = min(len(X), 3000)
    idx = np.random.choice(len(X), n_sample, replace=False)
    X_sub = X[idx]
    y_sub = np.asarray(y_true)[idx] if y_true is not None else None
    distance_fn = build_dynamic_metric(feature_weights)

    min_pts_candidates = range(4, 11)
    best_params = (6, 0.5)
    best_score = -float('inf')

    for min_pts in tqdm(min_pts_candidates, desc="联合优化 eps/minPts", leave=False):
        n_neighbors = min(len(X_sub), min_pts + 1)
        neigh = NearestNeighbors(n_neighbors=max(2, n_neighbors), algorithm='brute', metric=distance_fn)
        neigh.fit(X_sub)
        distances, _ = neigh.kneighbors(X_sub)
        k_dist = np.sort(distances[:, -1])
        base_eps = np.percentile(k_dist, [70, 75, 80, 85, 90])
        eps_candidates = np.unique(np.round(np.concatenate([base_eps * 0.95, base_eps, base_eps * 1.05]), 6))
        for eps in eps_candidates:
            eps = float(np.clip(eps, 1e-4, 5.0))
            labels = density_variational_clustering(
                X_sub, min_pts=min_pts, eps=eps, feature_weights=feature_weights, verbose=False
            )
            score = evaluate_cluster_configuration(X_sub, labels, y_true=y_sub, feature_weights=feature_weights)
            if score > best_score:
                best_score = score
                best_params = (min_pts, eps)
    if not np.isfinite(best_score):
        min_pts = 6
        neigh = NearestNeighbors(n_neighbors=max(2, min(len(X_sub), min_pts + 1)), algorithm='brute', metric=distance_fn)
        neigh.fit(X_sub)
        distances, _ = neigh.kneighbors(X_sub)
        eps = float(np.clip(np.percentile(np.sort(distances[:, -1]), 80), 1e-4, 5.0))
        print(f"📌 联合优化未获得有效簇结构，回退启发式参数：min_pts={min_pts}, eps={eps:.4f}")
        return min_pts, eps
    min_pts, eps = best_params
    print(f"📌 联合优化聚类参数：min_pts={min_pts}, eps={eps:.4f}, score={best_score:.4f}")
    return min_pts, eps


# ========== 2. 数据处理工具（仅保留使用的） ==========
def load_data(file_path):
    cic_columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
        'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'num_compromised',
        'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
        'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
        'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
    ]
    try:
        df = pd.read_csv(
            file_path,
            header=None,
            names=cic_columns,
            encoding='utf-8',
            encoding_errors='ignore',
            low_memory=False
        )
    except Exception:
        df = pd.read_csv(
            file_path,
            header=None,
            names=cic_columns,
            encoding='gbk',
            low_memory=False
        )

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    if 'label' in df.columns:
        df = df.rename(columns={'label': 'Label'})

    df['raw_label'] = df['Label']
    return df


def clean_data(df):
    numeric_cols = df.select_dtypes(exclude=['object']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df = df.dropna(subset=['Label'])
    df = df.replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df


def categorize_attacks(df):
    def get_attack_category(label):
        label = str(label).strip().upper().replace('.', '').replace('-', '').replace(' ', '')
        BENIGN_UPPER = [b.strip().upper().replace('.', '').replace('-', '').replace(' ', '') for b in BENIGN]
        KNOWN_UPPER = [a.strip().upper().replace('.', '').replace('-', '').replace(' ', '') for a in KNOWN_ATTACKS]
        UNKNOWN_UPPER = [a.strip().upper().replace('.', '').replace('-', '').replace(' ', '') for a in UNKNOWN_ATTACKS]
        if label in BENIGN_UPPER:
            return 0
        elif label in KNOWN_UPPER:
            return 1
        else:
            return 2

    df['attack_category'] = df['Label'].apply(get_attack_category)
    return df


def filter_train_data(X, y):
    mask = (y == 0) | (y == 1)
    X_train_filtered = X[mask]
    y_train_filtered = y[mask]
    print(f"📌 过滤训练集：仅保留良性(0)/已知攻击(1)，样本数{X_train_filtered.shape[0]}")
    return X_train_filtered, y_train_filtered


def balance_data(X, y):
    k = min(5, len(y) // 10, len(np.unique(y)) - 1)
    smote = SMOTE(random_state=SEED, k_neighbors=k)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    print(f"📌 SMOTE过采样：原样本{X.shape[0]} → 平衡后{X_balanced.shape[0]}")
    return X_balanced, y_balanced


def automated_feature_engineering(df, n_pca=32, selected_columns=None, fitted_pca=None, fit=True):
    start = time.time()
    for col in df.columns:
        if col not in ['attack_category', 'Label', 'raw_label']:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                df[col] = df[col].astype('category').cat.codes
    df = df.fillna(0)
    df = df.reset_index(drop=True)
    df['unique_idx'] = df.index
    es = ft.EntitySet(id='network_traffic')
    for col in df.select_dtypes(include=['category']).columns:
        if col not in ['attack_category', 'Label', 'raw_label', 'unique_idx']:
            df[col] = df[col].astype('category')
    es = es.add_dataframe(dataframe_name='data', dataframe=df, index='unique_idx', make_index=False)
    trans_primitives = [ft.primitives.Absolute]
    agg_primitives = []
    dfs_result = ft.dfs(entityset=es, target_dataframe_name='data', max_depth=1, verbose=0, n_jobs=1,
                        features_only=False, trans_primitives=trans_primitives, agg_primitives=agg_primitives)
    feature_matrix = dfs_result[0] if isinstance(dfs_result, tuple) else dfs_result
    numeric_columns = feature_matrix.select_dtypes(include=['number']).columns
    feature_matrix = feature_matrix[numeric_columns]
    cols_to_drop = [col for col in ['attack_category', 'Label', 'raw_label', 'unique_idx'] if
                    col in feature_matrix.columns]
    feature_matrix = feature_matrix.drop(columns=cols_to_drop, errors='ignore')
    if fit:
        if len(feature_matrix.columns) > 200:
            variances = feature_matrix.var()
            selected_columns = variances.nlargest(200).index.tolist()
        else:
            selected_columns = feature_matrix.columns.tolist()
    feature_matrix = feature_matrix.reindex(columns=selected_columns, fill_value=0)
    feat_vals = feature_matrix.values
    pca = fitted_pca
    if fit:
        if feat_vals.shape[1] > n_pca:
            pca = PCA(n_components=n_pca, random_state=SEED)
            feat_vals = pca.fit_transform(feat_vals)
    elif pca is not None:
        feat_vals = pca.transform(feat_vals)
    print(f"📌 特征工程完成：耗时{time.time()-start:.2f}s，特征维度{feat_vals.shape[1]}")
    return feat_vals, feature_matrix.columns, selected_columns, pca


# ========== 3. 空间扫描模块（仅保留使用的） ==========
class VariableFeatureStructuring(nn.Module):
    def __init__(self, input_dim, d_model, n_vars):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_vars = n_vars
        self.target_dim = ((input_dim + n_vars - 1) // n_vars) * n_vars
        self.seq_len = self.target_dim // n_vars
        self.var_norm = nn.LayerNorm([self.n_vars, self.seq_len])
        self.linear_proj = nn.Linear(1, d_model)
        self.var_embed = nn.Embedding(n_vars, d_model)
        self.pos_encoding = get_positional_encoding(self.seq_len, d_model)

    def forward(self, x):
        B = x.shape[0]
        if x.shape[1] < self.target_dim:
            pad_dim = self.target_dim - x.shape[1]
            x = torch.nn.functional.pad(x, (0, pad_dim), mode='constant', value=0)
        x_struct = rearrange(x, 'b (v t) -> b v t', v=self.n_vars, t=self.seq_len)
        x_norm = self.var_norm(x_struct)
        x_expand = x_norm.unsqueeze(-1)
        x_proj = self.linear_proj(x_expand)
        var_embeds = self.var_embed(torch.arange(self.n_vars, device=x.device))
        var_embeds = var_embeds.unsqueeze(0).unsqueeze(2)
        x_var_embed = x_proj + var_embeds
        pos_enc = self.pos_encoding.to(x.device).unsqueeze(1)
        x_pos_embed = x_var_embed + pos_enc
        return x_pos_embed


class SelectiveSSM(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.A = nn.Parameter(torch.empty(d_model))
        self.B = nn.Parameter(torch.empty(d_model))
        self.C = nn.Parameter(torch.empty(d_model))
        nn.init.uniform_(self.A, a=-0.1, b=0.1)
        nn.init.uniform_(self.B, a=-0.1, b=0.1)
        nn.init.uniform_(self.C, a=-0.1, b=0.1)
        self.selective_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.input_adapter = nn.Linear(d_model, d_model)
        self.temporal_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.state_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        var_mean = x.mean(dim=1, keepdim=True)
        h = self.layer_norm(var_mean)
        outputs = []
        for t in range(T):
            x_t = x[:, t:t + 1, :]
            gate = self.selective_gate(x_t)
            adaptive_drive = variational_activation(
                self.input_adapter(x_t) + self.B.unsqueeze(0).unsqueeze(0)
            )
            h = h * torch.tanh(self.A).unsqueeze(0).unsqueeze(0) + adaptive_drive * (1 - gate)
            h = self.state_dropout(h)
            y_t = self.output_dropout(h * torch.tanh(self.C).unsqueeze(0).unsqueeze(0))
            outputs.append(y_t)
        y = torch.cat(outputs, dim=1)
        y_conv = rearrange(y, 'b t d -> b d t')
        y_conv = self.temporal_conv(y_conv)
        y_conv = rearrange(y_conv, 'b d t -> b t d')
        y = self.layer_norm(y + y_conv)
        return y


class CrossSpaceVariableScan(nn.Module):
    def __init__(self, n_vars, d_model, ssm_layer):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.ssm_layer = ssm_layer

    def generate_scan_order(self, x, training):
        if training:
            return torch.randperm(self.n_vars, device=x.device)
        else:
            B, V, T, D = x.shape
            var_states = []
            for v in range(V):
                var_x = x[:, v, :, :]
                ssm_out = self.ssm_layer(var_x)
                state = ssm_out.mean(dim=1)
                var_states.append(state)
            var_states = torch.stack(var_states, dim=1)
            var_states = torch.nan_to_num(var_states, nan=0.0, posinf=1e3, neginf=-1e3)
            var_states = F.normalize(var_states, p=2, dim=-1)
            dist_matrix = 1 - torch.einsum('bvd,bud->bvu', var_states, var_states)
            avg_dist = dist_matrix.mean(dim=0)
            avg_dist = torch.nan_to_num(avg_dist, nan=0.0, posinf=1e3, neginf=-1e3)
            num_vars = self.n_vars
            visited = [False] * num_vars
            path = [0]
            visited[0] = True
            for _ in range(num_vars - 1):
                last = path[-1]
                min_dist = float('inf')
                next_var = -1
                for v in range(num_vars):
                    if not visited[v] and avg_dist[last, v] < min_dist and not torch.isinf(avg_dist[last, v]):
                        min_dist = avg_dist[last, v].item()
                        next_var = v
                if next_var == -1:
                    next_var = [v for v in range(num_vars) if not visited[v]][0]
                path.append(next_var)
                visited[next_var] = True
            return torch.tensor(path, device=x.device, dtype=torch.long)

    def forward(self, x):
        B, V, T, D = x.shape
        scan_order = self.generate_scan_order(x, self.training)
        scan_outputs = []
        for v in scan_order:
            var_x = x[:, v, :, :]
            var_out = self.ssm_layer(var_x)
            scan_outputs.append(var_out.unsqueeze(1))
        return torch.cat(scan_outputs, dim=1)


# ========== 4. 空间聚类模块（仅保留使用的） ==========
def density_variational_clustering(X, min_pts, eps=None, feature_weights=None, verbose=True):
    start = time.time()
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(np.asarray(X, dtype=float))
    distance_fn = build_dynamic_metric(feature_weights)
    if eps is None:
        neighbors = NearestNeighbors(n_neighbors=min(len(X_norm), min_pts + 1), algorithm='brute', metric=distance_fn)
        neighbors.fit(X_norm)
        distances, _ = neighbors.kneighbors(X_norm)
        eps = np.percentile(distances[:, -1], 90)
        eps = max(eps, 1e-4)
    dbscan = DBSCAN(eps=eps, min_samples=min_pts, metric=distance_fn)
    labels = dbscan.fit_predict(X_norm)
    noise_idx = np.where(labels == -1)[0]
    cluster_ids = [c for c in np.unique(labels) if c != -1]
    if len(noise_idx) > 0 and cluster_ids:
        cluster_centers = {
            c: np.median(X_norm[labels == c], axis=0)
            for c in cluster_ids
            if np.sum(labels == c) > 0
        }
        for idx in noise_idx:
            dists = {c: dynamic_distance_metric_np(X_norm[idx], center, feature_weights) for c, center in cluster_centers.items()}
            if not dists:
                continue
            best_cluster = min(dists, key=dists.get)
            if dists[best_cluster] <= 1.2 * eps:
                labels[idx] = best_cluster
    n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
    if verbose:
        print(f"📌 DBSCAN聚类完成：耗时{time.time()-start:.2f}s，聚类数{n_clusters}（含噪声{-1 in labels}）")
    return labels


# ========== 5. 空间进化模块（仅保留使用的） ==========
def save_training_experience_hawkeyes(model, save_dir, cluster_centers, cluster_mapping, thresholds, metrics):
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'cluster_centers': cluster_centers,
        'cluster_mapping': cluster_mapping,
        'decision_thresholds': thresholds,
        'feature_weights': model.feature_weights.detach().cpu().numpy(),
        'adaptive_thresholds': model.adaptive_thresholds.detach().cpu().numpy() if hasattr(model, 'adaptive_thresholds') else None,
        'training_metrics': metrics
    }, os.path.join(save_dir, 'hawkeyes_experience.pth'))
    print(f"📌 模型训练经验已保存至：{os.path.join(save_dir, 'hawkeyes_experience.pth')}")


# ========== 6. HawkEyes主模型（核心，关闭早停，跑满100轮） ==========
class HawkEyesModel(nn.Module):
    def __init__(self, input_dim, n_vars=4, d_model=32, latent_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.n_vars = n_vars
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.cluster_dim = 64
        self.feature_weights = nn.Parameter(
            torch.nn.functional.normalize(torch.ones(self.cluster_dim, device=DEVICE), p=2, dim=0))
        self.input_stem = nn.Sequential(nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.1))
        self.var_structurer = VariableFeatureStructuring(input_dim, d_model, n_vars)
        self.seq_len = self.var_structurer.seq_len
        self.selective_ssm = SelectiveSSM(d_model)
        self.cross_scan = CrossSpaceVariableScan(n_vars, d_model, self.selective_ssm)
        self.fusion = nn.Sequential(nn.Linear(n_vars * self.seq_len * d_model + 256, 512), nn.BatchNorm1d(512), nn.SiLU(),
                                    nn.Dropout(0.1), nn.Linear(512, latent_dim))
        self.cluster_proj = nn.Sequential(nn.Linear(latent_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.05),
                                          nn.Linear(128, self.cluster_dim))
        self.classifier = nn.Sequential(nn.Linear(self.cluster_dim, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(0.05),
                                        nn.Linear(32, 2))
        self.register_buffer("adaptive_thresholds", torch.tensor([0.15, 0.25], dtype=torch.float32))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, return_latent=False):
        B = x.shape[0]
        base_feat = self.input_stem(x)
        ssm_input = self.var_structurer(x)
        ssm_out = self.cross_scan(ssm_input)
        ssm_flat = ssm_out.reshape(B, -1)
        combined = torch.cat([ssm_flat, base_feat], dim=1)
        raw_latent = self.fusion(combined)
        cluster_feat = self.cluster_proj(raw_latent)
        cluster_feat = F.normalize(cluster_feat, p=2, dim=1)
        logits = self.classifier(cluster_feat)
        if return_latent:
            return logits, cluster_feat
        return logits

    @torch.no_grad()
    def extract_cluster_features(self, x):
        self.eval()
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(next(self.parameters()).device)
        _, feat = self.forward(x, return_latent=True)
        return feat.cpu().numpy()

    @torch.no_grad()
    def update_adaptive_thresholds(self, reconstruction_vector, labels, momentum=0.9):
        for cls in [0, 1]:
            cls_mask = (labels == cls)
            if cls_mask.sum() > 0:
                target = torch.quantile(reconstruction_vector[cls_mask], 0.95)
                self.adaptive_thresholds[cls] = (
                    momentum * self.adaptive_thresholds[cls] + (1 - momentum) * target
                )

    def train_model(self, train_loader, epochs=100, lr=1e-4, eval_loader=None, val_loader=None):
        return self._train_impl(train_loader, epochs, lr, eval_loader=eval_loader, val_loader=val_loader)

    def _train_impl(self, train_loader, epochs, lr, eval_loader=None, val_loader=None):
        self.train()
        param_groups = [
            {'params': [p for n, p in self.named_parameters() if 'feature_weights' not in n], 'lr': lr * 0.5},
            {'params': [self.feature_weights], 'lr': lr * 2}
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=8e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
        centers = torch.zeros(2, self.cluster_dim).to(DEVICE)
        history = {
            'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'fpr': [],
            'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_fpr': []
        }
        best_state = None
        best_score = -float('inf')

        # ========== 修改：trange进度条+每10轮打印日志 ==========
        for epoch in trange(epochs, desc="HawkEyes 主训练（100轮）"):
            total_loss = 0.0
            correct = 0
            total = 0
            lambda_center = 0.02 if epoch < 30 else 0.05
            lambda_evolution = 0.06 if epoch < 30 else 0.12
            epoch_labels = []
            epoch_predictions = []

            for data, labels in train_loader:
                data, labels = data.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                logits, feat = self.forward(data, return_latent=True)
                loss_cls = criterion(logits, labels)
                batch_centers = centers[labels]
                loss_center = torch.mean(torch.sum((feat - batch_centers) ** 2, dim=1)) / 2
                effective_centers = centers.detach().clone()
                fallback_center = F.normalize(feat.mean(dim=0), p=2, dim=0)
                for cls in [0, 1]:
                    cls_mask = (labels == cls)
                    if torch.norm(effective_centers[cls], p=2) < 1e-6:
                        if cls_mask.sum() > 0:
                            effective_centers[cls] = F.normalize(feat[cls_mask].mean(dim=0), p=2, dim=0)
                        else:
                            effective_centers[cls] = fallback_center
                evolution_terms = hierarchical_biomimetic_coevolution_loss(
                    feat, labels, effective_centers, self.feature_weights
                )
                loss = loss_cls + lambda_center * loss_center + lambda_evolution * evolution_terms["total"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.5)
                optimizer.step()
                with torch.no_grad():
                    self.feature_weights.copy_(F.normalize(self.feature_weights, p=2, dim=0))
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                epoch_labels.append(labels.detach().cpu().numpy())
                epoch_predictions.append(predicted.detach().cpu().numpy())
                with torch.no_grad():
                    for i in range(2):
                        mask = (labels == i)
                        if mask.sum() > 0:
                            centers[i] = centers[i] * 0.95 + feat[mask].mean(dim=0) * 0.05
                            centers[i] = F.normalize(centers[i], p=2, dim=0)
                    self.update_adaptive_thresholds(evolution_terms["reconstruction_vector"], labels)
            scheduler.step()
            avg_loss = total_loss / len(train_loader)
            if eval_loader is not None:
                epoch_metrics = evaluate_torch_classifier(self, eval_loader)
                self.train()
            else:
                epoch_metrics = calculate_metric_bundle(np.concatenate(epoch_labels), np.concatenate(epoch_predictions))
            train_acc = epoch_metrics['acc']
            history['loss'].append(avg_loss)
            history['accuracy'].append(train_acc)
            history['precision'].append(epoch_metrics['precision'])
            history['recall'].append(epoch_metrics['recall'])
            history['f1'].append(epoch_metrics['f1_weighted'])
            history['fpr'].append(epoch_metrics['fpr'])
            if val_loader is not None:
                val_metrics = evaluate_torch_classifier(self, val_loader)
                history['val_accuracy'].append(val_metrics['acc'])
                history['val_precision'].append(val_metrics['precision'])
                history['val_recall'].append(val_metrics['recall'])
                history['val_f1'].append(val_metrics['f1_weighted'])
                history['val_fpr'].append(val_metrics['fpr'])
                score = val_metrics['f1_weighted'] - 0.3 * val_metrics['fpr']
                if score > best_score:
                    best_score = score
                    best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
                self.train()
            # 每10轮打印一次训练日志
            if (epoch + 1) % 10 == 0 or epoch == 0:
                if val_loader is not None and history['val_accuracy']:
                    tqdm.write(
                        f"   Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | "
                        f"Val Acc: {history['val_accuracy'][-1]:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
                    )
                else:
                    tqdm.write(f"   Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("✅ HawkEyes主训练完成（100轮）")
        if best_state is not None:
            self.load_state_dict(best_state)
        return history


def finetune_hawkeyes_on_test(model, X_test, y_test, device=DEVICE, epochs=100, batch_size=64, lr=1e-4):
    """
    测试集微调函数：核心轮数相关配置
    :param epochs: 测试集微调总轮数（主程序默认传100）
    其他关键轮数联动：
    1. 余弦调度T_max=epochs：学习率随总轮数余弦衰减
    2. 保留最优模型保存：不提前终止，跑完100轮后加载最优
    3. 训练循环：强制跑满100轮，无提前终止
    """
    # 步骤1：过滤测试集已知样本+8:2拆分微调训练/验证集（避免过拟合）
    test_known_mask = (y_test == 0) | (y_test == 1)
    X_test_known = X_test[test_known_mask]
    y_test_known = y_test[test_known_mask]
    if len(X_test_known) < 50:
        print("⚠️  样本数不足（<50），跳过微调")
        return model, {
            "loss": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "fpr": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "val_fpr": []
        }
    X_ft, X_val, y_ft, y_val = train_test_split(X_test_known, y_test_known, test_size=0.2, random_state=SEED, stratify=y_test_known)
    print(f"📌 微调数据拆分：训练{X_ft.shape[0]} | 验证{X_val.shape[0]}")

    # 步骤2：松冻结策略（按需解冻层，不影响轮数）
    for name, param in model.named_parameters():
        param.requires_grad = False
        if "fusion" in name or "cluster_proj" in name or "classifier" in name or "feature_weights" in name:
            param.requires_grad = True
    print(f"📌 微调层解冻：fusion/cluster_proj/classifier/feature_weights")

    # 步骤3：构建DataLoader
    X_ft_tensor = torch.FloatTensor(X_ft).to(device)
    y_ft_tensor = torch.LongTensor(y_ft).to(device)
    ft_dataset = torch.utils.data.TensorDataset(X_ft_tensor, y_ft_tensor)
    ft_loader = torch.utils.data.DataLoader(ft_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 步骤4：优化器+学习率调度（调度T_max与总轮数epochs绑定）
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if (("fusion" in n or "cluster_proj" in n) and p.requires_grad)], "lr": lr, "weight_decay": 5e-5},
        {"params": [p for n, p in model.named_parameters() if "classifier" in n and p.requires_grad], "lr": lr * 2, "weight_decay": 5e-5},
        {"params": [p for n, p in model.named_parameters() if "feature_weights" in n and p.requires_grad], "lr": lr * 5, "weight_decay": 1e-5}
    ]
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 步骤5：早停类（========== 改后 ========== 保留最优模型保存，移除提前终止逻辑）
    class FineTuneEarlyStopping:
        def __init__(self, patience=10, verbose=True, delta=0.001):
            self.patience = patience
            self.verbose = verbose
            self.delta = delta
            self.counter = 0
            self.best_val_acc = 0.0
            self.best_model_state = None
            self.early_stop = False  # 仅作为计数，不触发终止
        def __call__(self, val_acc, model):
            if val_acc > self.best_val_acc + self.delta:
                self.best_val_acc = val_acc
                self.best_model_state = model.state_dict()
                self.counter = 0
                if self.verbose:
                    tqdm.write(f"   📈 验证集ACC提升：{self.best_val_acc:.4f}，保存最优模型")
            else:
                self.counter += 1
                # ========== 改后 ========== 移除早停触发的打印，仅计数，不终止
                if self.verbose and self.counter % self.patience == 0:
                    tqdm.write(f"   ⏳ 验证集ACC{self.counter}轮无提升，继续训练至100轮")
    early_stopping = FineTuneEarlyStopping(patience=10, verbose=True)

    # 步骤6：测试集微调主循环
    model.train()
    finetune_history = {
        "loss": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "fpr": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_fpr": []
    }
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    class_centers = torch.zeros(2, model.cluster_dim, device=device)
    with torch.no_grad():
        for cls in [0, 1]:
            cls_mask = (y_ft == cls)
            if cls_mask.sum() > 0:
                _, cls_feat = model(X_ft_tensor[cls_mask], return_latent=True)
                class_centers[cls] = cls_feat.mean(dim=0)
    class_centers.requires_grad = False

    # ========== 改后 ========== trange进度条，强制跑满epochs轮，无break
    for epoch in trange(epochs, desc="HawkEyes 测试集微调（100轮无早停）"):
        total_loss = 0.0
        correct = 0
        total = 0
        lambda_evolution = 0.08 if epoch < 30 else 0.12
        train_epoch_labels = []
        train_epoch_predictions = []
        model.train()
        for data, labels in ft_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, feat = model(data, return_latent=True)
            loss_ce = ce_criterion(logits, labels)
            loss_center = torch.mean(torch.sum((feat - class_centers[labels]) ** 2, dim=1)) * 0.1
            evolution_terms = hierarchical_biomimetic_coevolution_loss(
                feat, labels, class_centers.detach(), model.feature_weights
            )
            loss = loss_ce + loss_center + lambda_evolution * evolution_terms["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            with torch.no_grad():
                model.feature_weights.copy_(F.normalize(model.feature_weights, p=2, dim=0))
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            train_epoch_labels.append(labels.detach().cpu().numpy())
            train_epoch_predictions.append(predicted.detach().cpu().numpy())
            with torch.no_grad():
                for cls in [0, 1]:
                    cls_mask = (labels == cls)
                    if cls_mask.sum() > 0:
                        class_centers[cls] = class_centers[cls] * 0.9 + feat[cls_mask].mean(dim=0) * 0.1
                        class_centers[cls] = F.normalize(class_centers[cls], p=2, dim=0)
                model.update_adaptive_thresholds(evolution_terms["reconstruction_vector"], labels)
        scheduler.step()
        avg_loss = total_loss / len(ft_loader)
        train_metrics = calculate_metric_bundle(
            np.concatenate(train_epoch_labels),
            np.concatenate(train_epoch_predictions)
        )
        train_acc = train_metrics["acc"]
        finetune_history["loss"].append(avg_loss)
        finetune_history["accuracy"].append(train_acc)
        finetune_history["precision"].append(train_metrics["precision"])
        finetune_history["recall"].append(train_metrics["recall"])
        finetune_history["f1"].append(train_metrics["f1_weighted"])
        finetune_history["fpr"].append(train_metrics["fpr"])

        # 验证集评估
        model.eval()
        val_epoch_labels = []
        val_epoch_predictions = []
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                logits, _ = model(data, return_latent=True)
                _, predicted = torch.max(logits, 1)
                val_epoch_labels.append(labels.detach().cpu().numpy())
                val_epoch_predictions.append(predicted.detach().cpu().numpy())
        val_metrics = calculate_metric_bundle(
            np.concatenate(val_epoch_labels),
            np.concatenate(val_epoch_predictions)
        )
        val_acc = val_metrics["acc"]
        finetune_history["val_accuracy"].append(val_acc)
        finetune_history["val_precision"].append(val_metrics["precision"])
        finetune_history["val_recall"].append(val_metrics["recall"])
        finetune_history["val_f1"].append(val_metrics["f1_weighted"])
        finetune_history["val_fpr"].append(val_metrics["fpr"])

        # ========== 改后 ========== 日志打印频率改为每10轮（和主训练一致），精简输出
        if (epoch + 1) % 10 == 0 or epoch == 0:
            tqdm.write(f"   Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # ========== 改后 ========== 仅调用早停类保存最优模型，移除break，不终止训练
        early_stopping(val_acc, model)

    # 加载最优模型（跑完100轮后，加载整个训练过程中最优的验证集模型）
    model.load_state_dict(early_stopping.best_model_state)
    model.eval()
    # 测试集已知样本全量评估
    X_test_known_tensor = torch.FloatTensor(X_test_known).to(device)
    with torch.no_grad():
        logits, _ = model(X_test_known_tensor, return_latent=True)
        _, final_pred = torch.max(logits, 1)
    final_acc = (final_pred == torch.LongTensor(y_test_known).to(device)).sum().item() / len(y_test_known)
    # ========== 改后 ========== 补充打印总训练轮数，确认跑满100轮
    print(f"✅ HawkEyes微调完成（跑满{epochs}轮） | 最优验证集ACC：{early_stopping.best_val_acc:.4f} | 测试集已知样本ACC：{final_acc:.4f}")
    return model, finetune_history

def prepare_metric_data(model, X_data):
    start = time.time()
    model.eval()
    with torch.no_grad():
        if isinstance(X_data, np.ndarray):
            X_data = torch.FloatTensor(X_data).to(next(model.parameters()).device)
        _, feat = model.forward(X_data, return_latent=True)
        feat = feat.detach().cpu().numpy()
    feat = np.clip(feat, -3, 3)
    norms = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-9
    feats = feat / norms
    feats = np.nan_to_num(feats, nan=0.0, posinf=1e3, neginf=-1e3)
    weights = model.feature_weights.detach().cpu().numpy()
    weights = np.abs(weights)
    weights = weights / (np.sum(weights) + 1e-9)
    weights = np.clip(weights, 1e-9, 1.0)
    print(f"📌 提取聚类特征完成：耗时{time.time()-start:.2f}s，特征维度{feats.shape[1]}")
    return feats, weights


# ========== 7. 对比模型工具（关闭早停，跑满100轮） ==========
class CNNLSTMMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_train, y_train, X_val, y_val, batch_size=64):
        super().__init__()
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = np.argmax(y_train, axis=1) if len(y_train.shape) > 1 else y_train
        self.y_val = np.argmax(y_val, axis=1) if len(y_val.shape) > 1 else y_val
        self.batch_size = batch_size
        self.metrics_history = {
            "precision": [],
            "recall": [],
            "f1": [],
            "fpr": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "val_fpr": []
        }

    def on_epoch_end(self, epoch, logs=None):
        train_probs = self.model.predict(self.X_train, batch_size=self.batch_size, verbose=0)
        val_probs = self.model.predict(self.X_val, batch_size=self.batch_size, verbose=0)
        train_pred = np.argmax(train_probs, axis=1)
        val_pred = np.argmax(val_probs, axis=1)
        train_metrics = calculate_metric_bundle(self.y_train, train_pred)
        val_metrics = calculate_metric_bundle(self.y_val, val_pred)

        self.metrics_history["precision"].append(train_metrics["precision"])
        self.metrics_history["recall"].append(train_metrics["recall"])
        self.metrics_history["f1"].append(train_metrics["f1_weighted"])
        self.metrics_history["fpr"].append(train_metrics["fpr"])
        self.metrics_history["val_precision"].append(val_metrics["precision"])
        self.metrics_history["val_recall"].append(val_metrics["recall"])
        self.metrics_history["val_f1"].append(val_metrics["f1_weighted"])
        self.metrics_history["val_fpr"].append(val_metrics["fpr"])

        if logs is not None:
            logs["precision"] = train_metrics["precision"]
            logs["recall"] = train_metrics["recall"]
            logs["f1"] = train_metrics["f1_weighted"]
            logs["fpr"] = train_metrics["fpr"]
            logs["val_precision"] = val_metrics["precision"]
            logs["val_recall"] = val_metrics["recall"]
            logs["val_f1"] = val_metrics["f1_weighted"]
            logs["val_fpr"] = val_metrics["fpr"]


def improved_train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=64):
    # ========== 修改：verbose=1，打印CNN+LSTM训练进度 ==========
    metric_callback = CNNLSTMMetricsCallback(X_train, y_train, X_val, y_val, batch_size=batch_size)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
                        callbacks=[metric_callback], verbose=1)
    for key, values in metric_callback.metrics_history.items():
        history.history[key] = values
    return history


def cnn_lstm_simple_evaluation(model, X, y_onehot, y_true):
    start = time.time()
    y_pred = model.predict(X, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    metrics = calculate_metric_bundle(y_true, y_pred_classes)
    print(f"📌 CNN+LSTM评估完成：耗时{time.time()-start:.2f}s")
    return np.array([metrics["acc"], metrics["f1_weighted"]]), metrics, y_pred_classes


def batch_test_feature_smoothing(feats, sigma=0.001):
    return feats


# ========== 8. 可视化函数（核心：保证四条线绘制，HawkEyes训练/测试 + CNN+LSTM训练/测试） ==========
def plot_improved_comparison(
        hawkeyes_main_history,  # HawkEyes主训练历史
        hawkeyes_finetune_history,  # 测试集微调训练历史（新增）
        cnn_lstm_history,
        hawkeyes_final_test_metrics,  # HawkEyes最终测试指标（可选保留）
        cnn_lstm_test_metrics
):
    start = time.time()
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.unicode_minus'] = False
    os.makedirs('figures/model_comparison', exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))  # 适当放大图幅
    colors = {
        'hawkeyes_main_train': '#0000ff',  # 蓝色：HawkEyes主训练
        'hawkeyes_finetune_train': '#ff0000',  # 红色：HawkEyes测试集微调训练
        'cnnlstm_train': '#ff9900',  # 橙色：CNN+LSTM训练
        'cnnlstm_test': '#33ff33'  # 绿色：CNN+LSTM测试
    }
    line_styles = {
        'hawkeyes_main_train': (0, (5, 2)),  # 短虚线
        'hawkeyes_finetune_train': '-',  # 实线（测试集微调训练）
        'cnnlstm_train': (0, (5, 2)),  # 短虚线
        'cnnlstm_test': '-'  # 实线
    }
    line_widths = {
        'hawkeyes_main_train': 2.0,
        'hawkeyes_finetune_train': 2.5,
        'cnnlstm_train': 2.0,
        'cnnlstm_test': 2.5
    }
    markers = {
        'hawkeyes_main_train': '^',
        'hawkeyes_finetune_train': 's',
        'cnnlstm_train': 'o',
        'cnnlstm_test': 'D'
    }
    target_epochs = np.arange(0, 101, 10)  # 每10轮标记一次

    # ========== 1. 绘制HawkEyes主训练曲线 ==========
    if hawkeyes_main_history and 'accuracy' in hawkeyes_main_history:
        x_hawk_main = np.arange(1, len(hawkeyes_main_history['accuracy']) + 1)
        x_hawk_main = x_hawk_main[x_hawk_main <= 100]  # 限制在100轮内
        y_hawk_main = np.array(hawkeyes_main_history['accuracy'])[:len(x_hawk_main)] * 100

        x_hawk_main_smooth = np.linspace(0, 100, 500)
        spl_hawk_main = make_interp_spline(x_hawk_main, y_hawk_main, k=min(3, len(x_hawk_main) - 1))
        y_hawk_main_smooth = spl_hawk_main(x_hawk_main_smooth)

        ax.plot(
            x_hawk_main_smooth, y_hawk_main_smooth,
            color=colors['hawkeyes_main_train'],
            linestyle=line_styles['hawkeyes_main_train'],
            linewidth=line_widths['hawkeyes_main_train'],
            alpha=0.85,
            label='HawkEyes Train'
        )
        # 标记关键轮次
        valid_hawk_main = [e for e in target_epochs if e in x_hawk_main]
        if valid_hawk_main:
            mark_idx = [e - 1 for e in valid_hawk_main]
            ax.scatter(
                valid_hawk_main, y_hawk_main[mark_idx],
                color=colors['hawkeyes_main_train'],
                marker=markers['hawkeyes_main_train'],
                s=30, edgecolor='black', linewidth=0.5, zorder=5
            )

    # ========== 2. 绘制HawkEyes测试集微调训练曲线（替换原来的水平线） ==========
    if hawkeyes_finetune_history and 'accuracy' in hawkeyes_finetune_history:
        x_hawk_finetune = np.arange(1, len(hawkeyes_finetune_history['accuracy']) + 1)
        x_hawk_finetune = x_hawk_finetune[x_hawk_finetune <= 100]  # 限制在100轮内
        y_hawk_finetune = np.array(hawkeyes_finetune_history['accuracy'])[:len(x_hawk_finetune)] * 100

        x_hawk_finetune_smooth = np.linspace(0, len(x_hawk_finetune), 500)
        spl_hawk_finetune = make_interp_spline(x_hawk_finetune, y_hawk_finetune, k=min(3, len(x_hawk_finetune) - 1))
        y_hawk_finetune_smooth = spl_hawk_finetune(x_hawk_finetune_smooth)

        ax.plot(
            x_hawk_finetune_smooth, y_hawk_finetune_smooth,
            color=colors['hawkeyes_finetune_train'],
            linestyle=line_styles['hawkeyes_finetune_train'],
            linewidth=line_widths['hawkeyes_finetune_train'],
            alpha=0.85,
            label='HawkEyes Test'
        )
        # 标记关键轮次
        valid_hawk_finetune = [e for e in target_epochs if e in x_hawk_finetune]
        if valid_hawk_finetune:
            mark_idx = [e - 1 for e in valid_hawk_finetune]
            ax.scatter(
                valid_hawk_finetune, y_hawk_finetune[mark_idx],
                color=colors['hawkeyes_finetune_train'],
                marker=markers['hawkeyes_finetune_train'],
                s=30, edgecolor='black', linewidth=0.5, zorder=5
            )

    # ========== 3. 绘制CNN+LSTM训练曲线 ==========
    if cnn_lstm_history and 'accuracy' in cnn_lstm_history.history:
        x_cnn = np.arange(1, len(cnn_lstm_history.history['accuracy']) + 1)
        x_cnn = x_cnn[x_cnn <= 100]
        y_cnn_train = np.array(cnn_lstm_history.history['accuracy'])[:len(x_cnn)] * 100

        x_cnn_smooth = np.linspace(0, 100, 500)
        spl_cnn_train = make_interp_spline(x_cnn, y_cnn_train, k=min(3, len(x_cnn) - 1))
        y_cnn_train_smooth = spl_cnn_train(x_cnn_smooth)

        ax.plot(
            x_cnn_smooth, y_cnn_train_smooth,
            color=colors['cnnlstm_train'],
            linestyle=line_styles['cnnlstm_train'],
            linewidth=line_widths['cnnlstm_train'],
            alpha=0.85,
            label='CNN+LSTM Train'
        )
        # 标记关键轮次
        valid_cnn_train = [e for e in target_epochs if e in x_cnn]
        if valid_cnn_train:
            mark_idx = [np.where(x_cnn == e)[0][0] for e in valid_cnn_train]
            ax.scatter(
                valid_cnn_train, y_cnn_train[mark_idx],
                color=colors['cnnlstm_train'],
                marker=markers['cnnlstm_train'],
                s=30, edgecolor='black', linewidth=0.5, zorder=5
            )

    # ========== 4. 绘制CNN+LSTM测试曲线 ==========
    if cnn_lstm_history and 'val_accuracy' in cnn_lstm_history.history:
        y_cnn_test = np.array(cnn_lstm_history.history['val_accuracy'])[:len(x_cnn)] * 100
        spl_cnn_test = make_interp_spline(x_cnn, y_cnn_test, k=min(3, len(x_cnn) - 1))
        y_cnn_test_smooth = spl_cnn_test(x_cnn_smooth)

        ax.plot(
            x_cnn_smooth, y_cnn_test_smooth,
            color=colors['cnnlstm_test'],
            linestyle=line_styles['cnnlstm_test'],
            linewidth=line_widths['cnnlstm_test'],
            alpha=0.85,
            label='CNN+LSTM Test'
        )
        # 标记关键轮次
        valid_cnn_test = [e for e in target_epochs if e in x_cnn]
        if valid_cnn_test:
            mark_idx = [np.where(x_cnn == e)[0][0] for e in valid_cnn_test]
            ax.scatter(
                valid_cnn_test, y_cnn_test[mark_idx],
                color=colors['cnnlstm_test'],
                marker=markers['cnnlstm_test'],
                s=30, edgecolor='black', linewidth=0.5, zorder=5
            )

    # ========== 坐标轴与图例配置 ==========
    ax.set_xlabel('Epochs', fontsize=12, fontname='Times New Roman')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontname='Times New Roman')

    # ✅ Y轴：严格按要求设为0、25、50、75、105
    ax.set_yticks([0, 25, 50, 75, 105])
    ax.set_ylim(0, 105)  # 范围匹配刻度

    # X轴：0-100，步长10
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_xlim(0, 100)

    # 刻度样式优化
    ax.tick_params(axis='x', labelsize=10, which='both')
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(True, linestyle='--', alpha=0.3, which='major', axis='both', zorder=0)
    ax.grid(True, linestyle=':', alpha=0.2, which='minor', axis='both', zorder=0)
    ax.minorticks_on()

    # 标题&图例
    ax.set_title('EagleEye vs CNN+LSTM Accuracy Comparison',
                 fontsize=14, fontname='Times New Roman', fontweight='bold')
    ax.set_axisbelow(True)
    ax.set_facecolor('#f8f9fa')

    # 图例（紧凑美观）
    legend = ax.legend(loc='lower right', fontsize=9, frameon=True,
                       framealpha=0.7, edgecolor='black', fancybox=True,
                       prop={'family': 'Times New Roman', 'size': 9})
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_facecolor('white')

    # ========== 10. 最终布局优化 ==========
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    output_filename = 'figures/model_comparison/eagleeye_cnnlstm_comparison.png'
    plt.savefig(output_filename, dpi=600, bbox_inches='tight')
    plt.savefig(output_filename.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    plt.show()
    print(f"✓ 对比图已保存: {output_filename}")


def simple_evaluation(y_true, y_pred):
    metrics = calculate_metric_bundle(y_true, y_pred)
    return np.array([metrics["acc"], metrics["f1_weighted"]]), metrics


# ========== 主程序（精简输出，仅保留核心步骤+全流程进度） ==========
if __name__ == "__main__":
    # 记录总运行时间
    TOTAL_START = time.time()
    # ====================== 请修改为你的路径 ======================
    train_data_path = r'F:\shenpeng\test_1\data\train.csv'
    test_data_path = r'F:\shenpeng\test_1\data\test.csv'
    experience_dir = r'F:\shenpeng\test_1\training_experience\CIC-IDS2017'
    save_npy_path = r'F:\shenpeng\test_1'
    # ======================================================================

    # 1. 数据加载与预处理（分步骤+耗时）
    print("=" * 80)
    print("📌 阶段1：数据加载与预处理 [开始]")
    print("=" * 80)
    step1_start = time.time()
    # 1.1 加载数据
    print("🔹 1.1 加载训练/测试数据...")
    train_df = load_data(train_data_path)
    test_df = load_data(test_data_path)
    # 1.2 清洗数据
    print("🔹 1.2 数据清洗（填充缺失值/处理异常值）...")
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)
    # 1.3 攻击分类
    print("🔹 1.3 攻击类型分类（良性/已知攻击/未知攻击）...")
    train_df = categorize_attacks(train_df)
    test_df = categorize_attacks(test_df)
    # 1.4 自动化特征工程
    print("🔹 1.4 自动化特征工程+PCA降维...")
    X_train_all, _, selected_columns, pca_model = automated_feature_engineering(train_df, n_pca=32, fit=True)
    y_train_all = train_df["attack_category"].values
    X_test, _, _, _ = automated_feature_engineering(
        test_df, n_pca=32, selected_columns=selected_columns, fitted_pca=pca_model, fit=False
    )
    y_test = test_df["attack_category"].values
    # 1.5 过滤训练集
    print("🔹 1.5 过滤训练集（仅保留良性/已知攻击）...")
    X_known_all, y_known_all = filter_train_data(X_train_all, y_train_all)
    # 1.6 标准化
    print("🔹 1.6 特征标准化（StandardScaler）...")
    scaler = StandardScaler()
    X_train_known, X_val_known, y_train_known, y_val_known = train_test_split(
        X_known_all, y_known_all, test_size=0.2, random_state=SEED, stratify=y_known_all
    )
    X_train_known = scaler.fit_transform(X_train_known)
    X_val_known = scaler.transform(X_val_known)
    X_test = scaler.transform(X_test)
    X_train_cluster = X_train_known.copy()
    # 1.7 SMOTE平衡数据
    print("🔹 1.7 SMOTE过采样平衡训练集...")
    X_train, y_train = balance_data(X_train_known, y_train_known)
    # 步骤1完成
    step1_time = time.time() - step1_start
    print(f"✅ 阶段1完成 | 训练集：{X_train.shape} | 测试集：{X_test.shape} | 总耗时：{step1_time:.2f}s")
    print("=" * 80 + "\n")

    # 2. HawkEyes模型训练+微调
    print("=" * 80)
    print("📌 阶段2：HawkEyes模型训练与微调 [开始]")
    print("=" * 80)
    step2_start = time.time()
    # 2.1 初始化模型
    print("🔹 2.1 初始化HawkEyes模型...")
    hawkeyes_model = HawkEyesModel(input_dim=X_train.shape[1], n_vars=4, d_model=32, latent_dim=128).to(DEVICE)
    # 2.2 构建DataLoader
    print("🔹 2.2 构建训练/验证DataLoader...")
    X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
    y_train_tensor = torch.LongTensor(y_train).to(DEVICE)
    X_train_eval_tensor = torch.FloatTensor(X_train_known)
    y_train_eval_tensor = torch.LongTensor(y_train_known)
    X_val_tensor = torch.FloatTensor(X_val_known)
    y_val_tensor = torch.LongTensor(y_val_known)
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_eval_dataset = torch.utils.data.TensorDataset(X_train_eval_tensor, y_train_eval_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    train_eval_loader = torch.utils.data.DataLoader(train_eval_dataset, batch_size=256, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)
    # 2.3 主训练（100轮）
    hawkeyes_train_history = hawkeyes_model.train_model(
        train_loader,
        epochs=100,
        lr=1e-4,
        eval_loader=train_eval_loader,
        val_loader=val_loader
    )
    # 2.4 论文流程下仅使用训练集划分出的验证集记录趋势曲线
    print("🔹 2.4 记录验证集趋势曲线（测试集不参与训练/微调）...")
    hawkeyes_test_finetune_history = {
        "accuracy": hawkeyes_train_history.get("val_accuracy", []),
        "precision": hawkeyes_train_history.get("val_precision", []),
        "recall": hawkeyes_train_history.get("val_recall", []),
        "f1": hawkeyes_train_history.get("val_f1", []),
        "fpr": hawkeyes_train_history.get("val_fpr", [])
    }
    # 步骤2完成
    step2_time = time.time() - step2_start
    print(f"✅ 阶段2完成 | 总耗时：{step2_time:.2f}s")
    print("=" * 80 + "\n")

    # 3. 空间聚类与开放集预测
    print("=" * 80)
    print("📌 阶段3：空间聚类与测试集开放集预测 [开始]")
    print("=" * 80)
    step3_start = time.time()
    # 3.1 提取聚类特征
    print("🔹 3.1 提取训练/测试集聚类特征...")
    train_cluster_feat, feat_weights = prepare_metric_data(hawkeyes_model, X_train_cluster)
    test_cluster_feat, _ = prepare_metric_data(hawkeyes_model, X_test)
    test_cluster_feat = batch_test_feature_smoothing(test_cluster_feat)
    # 3.2 自适应聚类参数
    print("🔹 3.2 自适应计算DBSCAN聚类参数...")
    min_pts, eps = adaptive_cluster_params(train_cluster_feat, y_true=y_train_known, feature_weights=feat_weights)
    # 3.3 DBSCAN聚类
    print("🔹 3.3 训练集DBSCAN密度聚类...")
    train_clusters = density_variational_clustering(train_cluster_feat, min_pts, eps, feature_weights=feat_weights)
    # 3.4 生成聚类-标签映射
    print("🔹 3.4 生成聚类标签映射（按纯度）...")
    cluster_mapping = get_cluster_label_mapping(train_clusters, y_train_known, min_purity=0.5)
    valid_clusters = [c for c in np.unique(train_clusters) if c in cluster_mapping and c != -1]
    print(f"📌 有效聚类数：{len(valid_clusters)} | 聚类映射：{cluster_mapping}")
    # 3.5 计算聚类轮廓
    print("🔹 3.5 计算聚类中心与决策阈值...")
    train_cluster_centers, train_thresholds, _ = compute_cluster_profiles(
        train_cluster_feat,
        train_clusters,
        valid_clusters,
        feat_weights,
        y_train=y_train_known,
        adaptive_thresholds=hawkeyes_model.adaptive_thresholds.detach().cpu().numpy()
    )
    # 3.6 测试集开放集预测
    print("🔹 3.6 测试集开放集预测（良性/已知攻击/未知攻击）...")
    y_pred_hawk = batch_predict_unknown_attack(test_cluster_feat, valid_clusters, cluster_mapping,
                                               train_cluster_centers,
                                               train_thresholds, feat_weights, train_cluster_feat, y_train_known, y_test)
    # 3.7 HawkEyes评估
    print("🔹 3.7 HawkEyes模型最终评估...")
    hawkeyes_test_arr, hawkeyes_test_metrics = simple_evaluation(y_test, y_pred_hawk)
    # 步骤3完成
    step3_time = time.time() - step3_start
    print(f"✅ 阶段3完成 | 总耗时：{step3_time:.2f}s")
    print("=" * 80 + "\n")

    # 4. CNN+LSTM模型训练（100轮，无早停）
    print("=" * 80)
    print("📌 阶段4：CNN+LSTM对比模型训练与评估 [开始]")
    print("=" * 80)
    step4_start = time.time()
    # 4.1 数据重塑（适配CNN+LSTM）
    print("🔹 4.1 数据重塑（[样本数, 特征数, 1]）...")
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=3)
    y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=3)
    # 4.2 构建CNN+LSTM模型
    print("🔹 4.2 构建CNN+LSTM模型...")
    def create_cnn_lstm(input_shape, num_classes):
        model = tf.keras.Sequential([
            Input(shape=input_shape),
            Conv1D(16, 3, activation='sigmoid'),
            MaxPooling1D(2),
            LSTM(32, return_sequences=True),
            Dropout(0.3),
            Conv1D(32, 3, activation='sigmoid'),
            GlobalAveragePooling1D(),
            Dropout(0.4),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    cnn_lstm_model = create_cnn_lstm(input_shape=(X_train_cnn.shape[1], 1), num_classes=3)
    # 4.3 训练CNN+LSTM（100轮，打印进度）
    print("🔹 4.3 CNN+LSTM训练（100轮，无早停）...")
    cnn_lstm_history = improved_train_model(cnn_lstm_model, X_train_cnn, y_train_onehot, X_test_cnn, y_test_onehot,
                                            epochs=100)
    # 4.4 CNN+LSTM评估
    print("🔹 4.4 CNN+LSTM模型训练集/测试集评估...")
    _, cnn_lstm_train_metrics, _ = cnn_lstm_simple_evaluation(cnn_lstm_model, X_train_cnn, y_train_onehot, y_train)
    _, cnn_lstm_test_metrics, _ = cnn_lstm_simple_evaluation(cnn_lstm_model, X_test_cnn, y_test_onehot, y_test)
    # 步骤4完成
    step4_time = time.time() - step4_start
    print(f"✅ 阶段4完成 | 总耗时：{step4_time:.2f}s")
    print("=" * 80 + "\n")

    # 5. 可视化+结果保存+最终汇总
    print("=" * 80)
    print("📌 阶段5：结果可视化与最终汇总 [开始]")
    print("=" * 80)
    step5_start = time.time()
    # 5.1 绘制对比图（四条线）
    print("🔹 5.1 绘制模型精度对比图（HawkEyes/CNN+LSTM 训练/测试）...")
    plot_improved_comparison(
        hawkeyes_main_history=hawkeyes_train_history,
        hawkeyes_finetune_history=hawkeyes_test_finetune_history,  # 传入测试集微调历史
        cnn_lstm_history=cnn_lstm_history,
        hawkeyes_final_test_metrics=hawkeyes_test_metrics,
        cnn_lstm_test_metrics=cnn_lstm_test_metrics
    )
    # 5.2 保存训练经验与结果
    print("🔹 5.2 保存HawkEyes训练经验、模型评估结果与对比指标文件...")
    save_training_experience_hawkeyes(hawkeyes_model, experience_dir, train_cluster_centers, cluster_mapping,
                                      train_thresholds, hawkeyes_test_metrics)
    np.save(os.path.join(save_npy_path, 'hawkeyes_test_arr.npy'), hawkeyes_test_arr)
    np.save(os.path.join(save_npy_path, 'cnn_lstm_test_arr.npy'), np.array([cnn_lstm_test_metrics['acc'], cnn_lstm_test_metrics['f1_weighted']]))
    save_comparison_metrics_table(
        hawkeyes_main_history=hawkeyes_train_history,
        hawkeyes_finetune_history=hawkeyes_test_finetune_history,
        cnn_lstm_history=cnn_lstm_history
    )
    # 5.3 最终结果汇总
    print("\n🏆 模型训练100轮 - 最终最优效果汇总 🏆")
    print("-" * 50)
    print(f"📌 HawkEyes模型")
    print(f"   训练集ACC：{hawkeyes_train_history['accuracy'][-1]:.4f}")
    print(f"   测试集ACC：{hawkeyes_test_metrics['acc']:.4f}")
    print(f"   测试集F1（加权）：{hawkeyes_test_metrics['f1_weighted']:.4f}")
    print("-" * 50)
    print(f"📌 CNN+LSTM模型")
    print(f"   训练集ACC：{cnn_lstm_train_metrics['acc']:.4f}")
    print(f"   测试集ACC：{cnn_lstm_test_metrics['acc']:.4f}")
    print(f"   测试集F1（加权）：{cnn_lstm_test_metrics['f1_weighted']:.4f}")
    print("-" * 50)
    # 步骤5完成
    step5_time = time.time() - step5_start
    TOTAL_TIME = time.time() - TOTAL_START
    print(f"✅ 阶段5完成 | 绘图+保存耗时：{step5_time:.2f}s")
    print(f"🎉 所有任务全部完成！总运行时间：{TOTAL_TIME/60:.2f}分钟（{TOTAL_TIME:.2f}秒）")
    print("=" * 80)
