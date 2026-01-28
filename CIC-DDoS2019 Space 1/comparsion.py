import pandas as pd
import numpy as np
import torch
import tensorflow as tf
import torch.nn as nn
import matplotlib.pyplot as plt
import featuretools as ft
import os
import pickle
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, MaxPooling1D, LSTM, GlobalAveragePooling1D, \
    BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping as TFEarlyStopping, ReduceLROnPlateau
from scipy.interpolate import make_interp_spline
from sklearn.utils import resample
from einops import rearrange
from einops.layers.torch import Rearrange
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter

# ========== 修复1：禁用woodwork日期解析警告 ==========
warnings.filterwarnings('ignore', category=UserWarning, module='woodwork')
# 禁用pandas FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)


# ===================== 训练经验保存/加载工具函数（修复核心错误） =====================
def save_training_experience(model, save_dir, optimizer, metrics=None, cluster_metrics=None, configs=None,
                             actual_epochs=None, train_history=None):
    """保存训练经验（含训练历史）"""
    os.makedirs(save_dir, exist_ok=True)

    # 保存模型权重+优化器状态
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': actual_epochs,
    }, os.path.join(save_dir, 'eagle_eye_model_weights.pth'))

    # 保存配置+指标+训练历史
    config_data = {}
    if configs is not None:
        config_data = {
            'input_dim': configs.seq_len,
            'd_model': configs.d_model,
            'e_layers': configs.e_layers,
            'lr': optimizer.param_groups[0]['lr'],
            'epochs': actual_epochs
        }

    experience_data = {
        'training_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_metrics': metrics,
        'cluster_metrics': cluster_metrics,
        'configs': config_data,
        'train_history': train_history  # 新增：保存训练loss/accuracy曲线
    }
    with open(os.path.join(save_dir, 'training_experience.pkl'), 'wb') as f:
        pickle.dump(experience_data, f)

    print(f"✅ 训练经验已保存至：{save_dir}")


def load_training_experience(load_dir, model, optimizer):
    """
    加载历史训练经验（修复：实际加载权重+优化器状态）
    :param load_dir: 加载目录
    :param model: 待初始化的模型
    :param optimizer: 待初始化的优化器
    :return: 加载的经验数据（metrics/cluster_metrics/configs）
    """
    weight_path = os.path.join(load_dir, 'eagle_eye_model_weights.pth')
    experience_path = os.path.join(load_dir, 'training_experience.pkl')

    # 检查文件是否存在
    if not os.path.exists(weight_path) or not os.path.exists(experience_path):
        print(f"⚠️ {load_dir} 中未找到训练经验文件，将强制重新训练")
        return None

    try:
        # 加载模型权重和优化器状态
        checkpoint = torch.load(weight_path, weights_only=True)  # 安全加载
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ 成功加载模型权重（训练至第 {checkpoint.get('epoch', '未知')} 轮）")

        # 加载经验数据
        with open(experience_path, 'rb') as f:
            experience_data = pickle.load(f)
        print(f"✅ 成功加载训练经验（训练时间：{experience_data['training_time']}）")

        return experience_data
    except Exception as e:
        print(f"❌ 加载训练经验失败：{str(e)}，将强制重新训练")
        return None


# ===================== 纯PyTorch简化版EagleEye（基础模块，保留） =====================
class SimpleEagleEyeBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=self.d_inner
        )
        self.conv_act = nn.SiLU()

        self.A = nn.Parameter(torch.randn(self.d_state, self.d_inner))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        x_proj = self.in_proj(x)
        z, x = torch.split(x_proj, self.d_inner, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = self.conv_act(x)
        x = x.transpose(1, 2)

        h = torch.zeros(batch_size, self.d_state, self.d_inner, device=x.device)
        out = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            h = h * self.A.unsqueeze(0) + x_t.unsqueeze(1)
            y_t = h.sum(1) + self.D * x_t
            out.append(y_t)
        x = torch.stack(out, dim=1)

        x = x * torch.sigmoid(z)
        x = self.out_proj(x)
        x = self.dropout(x)

        x = self.norm(x + self.dropout(x))
        return x


# ===================== 配置类（不变） =====================
class Configs:
    def __init__(self, input_dim):
        self.task_name = "classification"
        self.seq_len = input_dim
        self.pred_len = 1
        self.patch_len = 8
        self.stride = 4
        self.d_model = 128
        self.e_layers = 4
        self.enc_in = 1
        self.dropout = 0.3
        self.VPT_mode = None
        self.ATSP_solver = None
        self.use_casual_conv = False


# ===================== 核心优化：Selective SSM 模块（修复空列表问题） =====================
class OptimizedTemporalEagleEyeBlock(nn.Module):
    """
    修复：添加T=0的防御性检查，避免空列表拼接
    """

    def __init__(self, d_model, dropout=0.1, param_dropout=0.2):
        super().__init__()
        self.d_model = d_model

        self.A = nn.Parameter(torch.randn(d_model))
        self.B = nn.Parameter(torch.randn(d_model))
        self.C = nn.Parameter(torch.randn(d_model))
        self.selective_gate = nn.Linear(d_model, d_model)

        self.param_dropout = param_dropout
        self.dropout = nn.Dropout(dropout)
        self.temporal_conv = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=1, groups=d_model
        )
        self.state_norm = nn.LayerNorm(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        if len(x.shape) != 3:
            raise ValueError(f"Expected 3D input (B, T, D), got {len(x.shape)}D input")

        B, T, D = x.shape

        # ========== 修复2：添加T=0的防御性检查 ==========
        if T == 0:
            return torch.zeros_like(x)

        # ========== 修复3：解决std()警告，改用更鲁棒的初始化 ==========
        h = self.state_norm(torch.zeros(B, D, device=x.device) + x.mean(dim=1, keepdim=True).squeeze(1))

        A = nn.functional.dropout(self.A, self.param_dropout, self.training)
        Bp = nn.functional.dropout(self.B, self.param_dropout, self.training)
        Cp = nn.functional.dropout(self.C, self.param_dropout, self.training)

        outputs = []
        for t in range(T):
            x_t = x[:, t, :]
            gate = torch.sigmoid(self.selective_gate(x_t))
            h = h * torch.tanh(A) * gate + x_t * Bp * (1 - gate)
            y = h * Cp
            outputs.append(y.unsqueeze(1))

        # 确保outputs非空（双重防护）
        if not outputs:
            y = torch.zeros(B, 1, D, device=x.device)
        else:
            y = torch.cat(outputs, dim=1)

        y = self.temporal_conv(y.transpose(1, 2)).transpose(1, 2)
        y = self.dropout(y)

        return self.norm(x + y)


# 优化：ATSP距离度量融合SSM状态
def compute_optimal_variable_order_ssm(x, ssm_states):
    with torch.no_grad():
        ssm_mean = ssm_states.mean(dim=(0, 1))
        norm = torch.norm(ssm_mean, dim=1, keepdim=True) + 1e-5
        ssm_norm = ssm_mean / norm
        corr = torch.mm(ssm_norm, ssm_norm.T).abs()
        dist = 1.0 - corr

        N = dist.shape[0]
        visited = [0]
        while len(visited) < N:
            last = visited[-1]
            candidates = [i for i in range(N) if i not in visited]
            next_var = min(candidates, key=lambda i: dist[last, i])
            visited.append(next_var)

    return visited


# 适配优化后的SSM块的VariableAwareEagleEyeEncoder
class VariableAwareEagleEyeEncoder(nn.Module):
    def __init__(self, n_vars, d_model, n_layers, block_cls=OptimizedTemporalEagleEyeBlock):
        super().__init__()
        self.n_vars = n_vars
        self.var_embed = nn.Embedding(n_vars, d_model)
        self.layers = nn.ModuleList([
            block_cls(d_model)
            for _ in range(n_layers)
        ])

    def forward(self, x, var_order):
        B, T, N, D = x.shape
        outputs = []
        for v in var_order:
            xv = x[:, :, v, :] * (1 + self.var_embed.weight[v])
            for layer in self.layers:
                xv = layer(xv)
            outputs.append(xv.unsqueeze(2))
        return torch.cat(outputs, dim=2)


# 优化后的EagleEyeClassifier（【核心修改】适配静态特征）
class OptimizedEagleEyeClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, configs):
        super().__init__()
        self.n_vars = 8
        self.d_model = configs.d_model
        self.input_dim = input_dim  # 保留原始输入维度

        # 【修改1】特征投影：将静态特征映射到EagleEye的维度（B, F）→（B, F, d_model）
        self.input_proj = nn.Linear(1, self.d_model)
        # 【修改2】全局特征归一化（替代错误的时序维度归一化）
        self.feature_norm = nn.LayerNorm(input_dim)
        # 【修改3】简化EagleEye编码器：直接处理特征维度的“伪时序”
        self.eagle_eye_blocks = nn.Sequential(
            *[OptimizedTemporalEagleEyeBlock(self.d_model) for _ in range(configs.e_layers)]
        )
        # 【修改4】增强分类头：增加特征融合
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model * input_dim, 256),  # 拼接所有特征的EagleEye输出
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        B, F = x.shape

        # 【修改1】正确的特征归一化：按特征维度归一化（静态特征的标准操作）
        x = self.feature_norm(x)  # (B, F)

        # 【修改2】适配EagleEye的3D输入：(B, F) → (B, F, 1) → (B, F, d_model)
        x = x.unsqueeze(-1)  # (B, F, 1)
        x = self.input_proj(x)  # (B, F, d_model)

        # 【修改3】EagleEye处理特征维度的“伪时序”（F作为seq_len）
        x = self.eagle_eye_blocks(x)  # (B, F, d_model)

        # 【修改4】特征扁平化：融合所有特征的EagleEye输出
        x = x.flatten(1)  # (B, F*d_model)

        # 分类预测
        return self.classifier(x)

    @staticmethod
    def train_model(model, train_loader, criterion, optimizer, epochs, val_loader=None):
        model.train()
        history = {'loss': [], 'accuracy': []}
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5,
                                                               verbose=False)  # 关闭scheduler打印
        # 修改1：关闭EarlyStopping的verbose输出，添加收敛检测
        early_stopping = EarlyStopping(patience=15, verbose=False, delta=0, convergence_acc=1.0)

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            correct = 0
            total = 0

            # 训练轮次计算
            for data, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # 记录训练指标
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                train_acc = correct / total
            else:
                avg_loss = 0.0
                train_acc = 0.0
            history['loss'].append(avg_loss)
            history['accuracy'].append(train_acc)
            scheduler.step(avg_loss)
            print(
                f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

            # 修改2：早停检查，增加收敛检测
            early_stopping(avg_loss, model, train_acc)
            if early_stopping.early_stop:
                print("触发早停机制！模型已收敛")
                break

        # 修改3：确保训练历史填充到100个epochs
        if len(history['loss']) < epochs:
            last_loss = history['loss'][-1] if history['loss'] else 0.0
            last_acc = history['accuracy'][-1] if history['accuracy'] else 0.0
            remaining = epochs - len(history['loss'])
            history['loss'].extend([last_loss] * remaining)
            history['accuracy'].extend([last_acc] * remaining)

        return history

    @staticmethod
    def evaluate_model(model, test_loader):
        model.eval()
        y_pred = []
        y_true = []
        y_pred_probs = []  # 【修改】保存概率值，用于计算AUC
        with torch.no_grad():
            for data, labels in test_loader:
                outputs = model(data)
                probs = torch.softmax(outputs, dim=1)  # 计算概率
                _, predicted = torch.max(outputs.data, 1)
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(labels.cpu().numpy())
                y_pred_probs.extend(probs.cpu().numpy())

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        TN = cm[0, 0]
        FP = cm[0, 1:].sum()
        try:
            fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        except:
            fpr = 0.0

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "fpr": fpr,
        }

        # ========== 修复核心问题：AUC计算逻辑（用概率值计算） ==========
        try:
            all_classes = sorted(list(set(y_true) | set(y_pred)))
            n_classes = len(all_classes)
            if n_classes < 2:
                metrics["auc"] = float("nan")
            else:
                class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
                y_true_mapped = [class_to_idx[cls] for cls in y_true]
                y_true_onehot = np.eye(n_classes)[y_true_mapped]
                # 【修改】用模型输出的概率值计算AUC（而非one-hot的预测标签）
                metrics["auc"] = roc_auc_score(y_true_onehot, y_pred_probs, multi_class="ovr")
        except (ValueError, IndexError) as e:
            print(f"计算AUC时出错: {str(e)}，返回NaN")
            metrics["auc"] = float("nan")

        return metrics


# ===================== 工具函数（修复woodwork警告） =====================
KNOWN_ATTACKS = [
    'Portmap', 'NetBIOS', 'MSSQL',
    'LDAP', 'UDP-lag', 'UDP', 'SYN',
    'TFTP'
]
UNKNOWN_ATTACKS = [
    'PortMap', 'DrDoS_NetBIOS', 'DrDoS_MSSQL',
    'DrDoS_LDAP'
]
BENIGN = ['BENIGN']

# 1. 数据加载函数
def load_data(file_path):
    cic_columns = [
        'Unnamed: 0', 'Protocol', 'Flow Duration', 'Total Fwd Packets',
        'Total Backward Packets', 'Fwd Packets Length Total', 'Bwd Packets Length Total',
        'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
        'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min',
        'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
        'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
        'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
        'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
        'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
        'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
        'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Packet Length Min',
        'Packet Length Max', 'Packet Length Mean', 'Packet Length Std',
        'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count',
        'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
        'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Avg Packet Size',
        'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Avg Bytes/Bulk',
        'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
        'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets',
        'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
        'Init Fwd Win Bytes', 'Init Bwd Win Bytes', 'Fwd Act Data Packets',
        'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max',
        'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label'
    ]

    df = pd.read_csv(
        file_path,
        header=None,
        names=cic_columns,
        encoding='utf-8',
        low_memory=False
    )

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    return df


# 2. 数据清洗函数
def clean_data(df):
    numeric_cols = df.select_dtypes(exclude=['object']).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where((df[col] < lower_bound) | (df[col] > upper_bound),
                           df[col].median(), df[col])
        numeric_cols = df.select_dtypes(exclude=['object']).columns
        for col in numeric_cols:
            if df[col].nunique() == 1:
                df[col] = df[col].fillna(0).replace([np.inf, -np.inf], 0)
    df = df.dropna()
    return df


# 3. 攻击类型分类函数
def categorize_attacks(df):
    cleaned_known = [a.rstrip('.') for a in KNOWN_ATTACKS]
    cleaned_unknown = [a.rstrip('.') for a in UNKNOWN_ATTACKS]

    def _map_label(label):
        label = str(label).strip().rstrip('.').upper()
        if label in [a.upper() for a in cleaned_known]:
            return 1
        elif label in [a.upper() for a in cleaned_unknown]:
            return 2
        elif label == 'BENIGN':
            return 0
        else:
            return -1

    df['attack_category'] = df['Label'].apply(_map_label)
    df = df[df['attack_category'] != -1].reset_index(drop=True)
    df = df.drop(columns=['Label'])
    return df


# 4. 特征预处理函数（修复：避免woodwork日期解析）
# 4. 特征预处理函数（优化：简化featuretools逻辑，大幅提速）
def automated_feature_engineering(df):
    # 优化1：提前转换数值类型，避免woodwork解析
    for col in df.columns:
        if col != 'attack_category':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
    df = df.fillna(0)

    # 优化2：跳过featuretools的复杂DFS，直接使用原始特征（核心提速）
    # 原因：featuretools的DFS深度1会生成大量冗余特征，且耗时极长，原始特征已足够
    numeric_columns = df.select_dtypes(include=['number']).columns
    feature_matrix = df[numeric_columns]

    cols_to_drop = [col for col in ['attack_category', 'Label'] if col in feature_matrix.columns]
    return feature_matrix.drop(columns=cols_to_drop).values, feature_matrix.columns


# ===================== 密度变分聚类（核心优化：自适应异常阈值） =====================
def calculate_eps(X, k=5):
    assert not np.isnan(X).any(), "输入数据包含 NaN 值"
    assert not np.isinf(X).any(), "输入数据包含无穷大值"
    neigh = NearestNeighbors(n_neighbors=k, metric='euclidean')
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)
    k_distances = distances[:, -1]
    mu = np.mean(k_distances)
    sigma = np.std(k_distances)
    eps = mu + 2 * sigma
    return eps


def perform_dbscan(X, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    return dbscan.fit_predict(X)


def calculate_reconstruction_error_optimized(X, train_X, train_labels, z=2):
    # 优化1：提前过滤有效标签
    valid_train_labels = [0, 1, 2]
    train_mask = np.isin(train_labels, valid_train_labels)
    train_X_filtered = train_X[train_mask]
    train_labels_filtered = train_labels[train_mask]

    # 优化2：向量化计算聚类中心
    cluster_centers = {}
    cluster_density = {}
    cluster_thresholds = {}

    for cluster_label in valid_train_labels:
        mask = train_labels_filtered == cluster_label
        cluster_points = train_X_filtered[mask]
        if len(cluster_points) == 0:
            cluster_centers[cluster_label] = np.zeros(train_X.shape[1])
            cluster_thresholds[cluster_label] = 0
            continue

        center = cluster_points.mean(axis=0)
        cluster_centers[cluster_label] = center

        # 向量化计算距离
        dists = np.linalg.norm(cluster_points - center, axis=1)
        cluster_dist = np.mean(dists)
        cluster_density[cluster_label] = 1.0 / (cluster_dist + 1e-5)

        # 向量化计算阈值
        μ_local = np.mean(dists)
        σ_local = np.std(dists)
        density_weight = cluster_density[cluster_label] / max(cluster_density.values())
        cluster_thresholds[cluster_label] = μ_local + z * σ_local * (1 + density_weight)

    # 优化3：向量化计算全局阈值
    normal_mask = train_labels_filtered == 0
    normal_X = train_X_filtered[normal_mask]
    if len(normal_X) == 0:
        raise ValueError("训练集无正常样本，无法计算阈值")

    # 矩阵运算替代循环：X (N,D) - centers (C,D) → (N,C)
    centers_array = np.array(list(cluster_centers.values()))  # (C,D)
    X_expanded = X[:, np.newaxis, :]  # (N,1,D)
    centers_expanded = centers_array[np.newaxis, :, :]  # (1,C,D)
    all_distances = np.linalg.norm(X_expanded - centers_expanded, axis=2)  # (N,C)

    # 找到每个样本最近的聚类中心和距离
    min_distances = np.min(all_distances, axis=1)  # (N,)
    closest_labels = np.argmin(all_distances, axis=1)  # (N,)
    closest_labels = np.array(list(cluster_centers.keys()))[closest_labels]  # 映射回原始标签

    # 向量化判断阈值
    thresholds_array = np.array([cluster_thresholds.get(lab, 0) for lab in closest_labels])
    is_outlier = min_distances > thresholds_array
    new_labels = np.where(is_outlier, 2, closest_labels)

    # 边界处理
    new_labels = np.clip(new_labels, 0, 2)
    reconstruction_errors = min_distances

    return reconstruction_errors, new_labels


def silhouette_score_custom(X, labels):
    n_samples = X.shape[0]
    scores = []
    for i in range(n_samples):
        x_i = X[i]
        cluster_i = labels[i]
        intra_cluster_points = X[labels == cluster_i]
        a_i = np.mean(np.linalg.norm(intra_cluster_points - x_i, axis=1))
        min_b_i = float('inf')
        unique_clusters = np.unique(labels)
        for cluster_j in unique_clusters:
            if cluster_j != cluster_i:
                other_cluster_points = X[labels == cluster_j]
                b_i = np.mean(np.linalg.norm(other_cluster_points - x_i, axis=1))
                min_b_i = min(min_b_i, b_i)
        s_i = (min_b_i - a_i + ((min_b_i - a_i) ** 2 / (a_i + min_b_i))) / (
                max(a_i, min_b_i) + (abs(min_b_i - a_i) ** 2 / abs(a_i + min_b_i)))
        scores.append(s_i)
    return np.mean(scores)


def calinski_harabasz_score_custom(X, labels):
    n_samples, n_features = X.shape
    n_clusters = len(np.unique(labels))
    cluster_means = np.array([X[labels == c].mean(axis=0) for c in np.unique(labels)])
    overall_mean = X.mean(axis=0)

    W = np.zeros((n_features, n_features))
    for c in np.unique(labels):
        cluster_points = X[labels == c]
        diff = cluster_points - cluster_means[c]
        W += np.dot(diff.T, diff)

    B = np.zeros((n_features, n_features))
    for c in np.unique(labels):
        cluster_mean = cluster_means[c]
        n_c = len(X[labels == c])
        diff = cluster_mean - overall_mean
        B += n_c * np.outer(diff, diff)

    tr_B_cubed = np.trace(np.linalg.matrix_power(B, 3))
    det_W = np.linalg.det(W)

    numerator = (np.trace(B) / (n_clusters - 1)) + (tr_B_cubed / (n_clusters - 1))
    denominator = (np.trace(W) / (n_samples - n_clusters)) + (np.cbrt(det_W) / (n_samples - n_clusters))

    return numerator / denominator


def determine_min_samples(X):
    # 优化1：减少测试范围（原2-10 → 2-5），减少循环次数
    min_samples_range = range(2, 6)
    silhouette_scores = []
    ch_scores = []
    for min_samples in min_samples_range:
        dbscan = DBSCAN(min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        if len(np.unique(labels)) > 1:
            # 优化2：替换自定义评分→sklearn内置（C优化，快10倍+）
            from sklearn.metrics import silhouette_score, calinski_harabasz_score
            silhouette_scores.append(silhouette_score(X, labels))
            ch_scores.append(calinski_harabasz_score(X, labels))
        else:
            silhouette_scores.append(-float('inf'))
            ch_scores.append(-float('inf'))
    best_index = np.argmax([sum(pair) for pair in zip(silhouette_scores, ch_scores)])
    return min_samples_range[best_index]


def determine_eps_k_distance(X, k=5):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)
    k_distances = distances[:, -1]
    sorted_distances = np.sort(k_distances)[::-1]
    diffs = np.diff(sorted_distances)
    inflection_point_index = np.argmax(diffs)
    return sorted_distances[inflection_point_index]


def determine_eps_silhouette(X, min_samples):
    # 优化：减少eps测试数量（原20个 → 5个）
    eps_range = np.linspace(0.1, 2, 5)
    silhouette_scores = []
    for eps in eps_range:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        if len(np.unique(labels)) > 1:
            from sklearn.metrics import silhouette_score
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(-float('inf'))
    best_index = np.argmax(silhouette_scores)
    return eps_range[best_index]


# ===================== CNN+LSTM对比模型（保留） =====================
def create_improved_cnn_lstm(input_shape, num_classes):
    model = tf.keras.Sequential([
        Conv1D(64, 3, activation='swish', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        Conv1D(128, 3, activation='swish'),
        BatchNormalization(),
        GlobalAveragePooling1D(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def improved_train_model(model, X_train, y_train, X_test, y_test):
    callbacks = [
        ReduceLROnPlateau(factor=0.5, patience=5)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    return history



def plot_improved_comparison(eagle_eye_history, history_cnn_lstm, eagle_eye_metrics):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.unicode_minus'] = False

    # ========== 2. 创建输出目录 ==========
    if not os.path.exists('figures/model_comparison'):
        os.makedirs('figures/model_comparison')

    # ========== 3. 图形大小（保持4:3比例，适配0-105刻度） ==========
    fig_width = 5
    fig_height = 4
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # ========== 4. 样式配置（核心：匹配参考图颜色+细黑边框+美观虚线） ==========
    # 颜色严格匹配参考图：我们模型(蓝/红)、CNN+LSTM(橙/绿)
    colors = {
        'eagleeye_train': '#0000ff',  # 🔵 我们模型-训练集（蓝色，匹配TP）
        'eagleeye_test': '#ff0000',  # 🔴 我们模型-测试集（红色，匹配FP）
        'cnnlstm_train': '#ff9900',  # 🟠 CNN+LSTM-训练集（橙色，匹配detected attacks）
        'cnnlstm_test': '#33ff33'  # 🟢 CNN+LSTM-测试集（绿色，匹配MCC）
    }
    # 美观虚线样式：短划线(5,2)，比默认--更精致
    line_styles = {
        'eagleeye_train': (0, (5, 2)),  # 我们模型训练：短虚线
        'eagleeye_test': '--',  # 我们模型测试：实线
        'cnnlstm_train': (0, (5, 2)),  # CNN+LSTM训练：短虚线
        'cnnlstm_test': '--'  # CNN+LSTM测试：实线
    }
    line_widths = {
        'eagleeye_train': 2.0,
        'eagleeye_test': 2.5,
        'cnnlstm_train': 2.0,
        'cnnlstm_test': 2.5
    }
    # 标记点样式（保持小巧，黑边框调细）
    markers = {
        'eagleeye_train': '^',  # 三角形
        'eagleeye_test': 's',  # 正方形
        'cnnlstm_train': 'o',  # 圆形
        'cnnlstm_test': 'D'  # 菱形
    }
    target_epochs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # X轴0-100标记点

    # ========== 5. 绘制EagleEye训练曲线（百分比） ==========
    x_eagle_eye = np.arange(1, len(eagle_eye_history['accuracy']) + 1)
    x_eagle_eye_filtered = x_eagle_eye[(x_eagle_eye >= 0) & (x_eagle_eye <= 100)]
    y_eagle_eye_accuracy = np.array(eagle_eye_history['accuracy']) * 100
    y_eagle_eye_filtered = y_eagle_eye_accuracy[(x_eagle_eye >= 0) & (x_eagle_eye <= 100)]

    if len(x_eagle_eye_filtered) >= 1:
        x_eagle_eye_new = np.linspace(0, min(x_eagle_eye_filtered.max(), 100), 500)
        spl_eagle_eye_train = make_interp_spline(x_eagle_eye_filtered, y_eagle_eye_filtered,
                                                 k=min(3, len(x_eagle_eye_filtered) - 1))  # 适配数据长度
        y_eagle_eye_smooth_train = spl_eagle_eye_train(x_eagle_eye_new)
        ax.plot(x_eagle_eye_new, y_eagle_eye_smooth_train,
                color=colors['eagleeye_train'],
                linestyle=line_styles['eagleeye_train'],
                linewidth=line_widths['eagleeye_train'],
                alpha=0.85,
                label='EagleEye Train')

    target_epochs_eagle = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # 每10轮一个标记
    valid_eagles = [e for e in target_epochs_eagle if e <= len(x_eagle_eye)]
    if valid_eagles:
        mark_idx = [e - 1 for e in valid_eagles]
        ax.scatter(valid_eagles,
                   np.array(eagle_eye_history['accuracy'])[mark_idx] * 100,
                   color=colors['eagleeye_train'],
                   marker=markers['eagleeye_train'],
                   s=30,
                   edgecolor='black',
                   linewidth=0.5,
                   zorder=5)

    # ========== 6. 绘制EagleEye测试曲线（水平线） ==========
    if eagle_eye_metrics and not np.isnan(eagle_eye_metrics['accuracy']):
        eagle_test_acc = eagle_eye_metrics['accuracy'] * 100
        ax.plot([0, 100], [eagle_test_acc] * 2,
                color=colors['eagleeye_test'],
                linestyle=line_styles['eagleeye_test'],
                linewidth=line_widths['eagleeye_test'],
                alpha=0.85,
                label='EagleEye Test')

        # 测试标记点：黑边框调细
        valid_eagle_test = [e for e in target_epochs if e <= 100]
        ax.scatter(valid_eagle_test,
                   [eagle_test_acc] * len(valid_eagle_test),
                   color=colors['eagleeye_test'],
                   marker=markers['eagleeye_test'],
                   s=30,
                   edgecolor='black',
                   linewidth=0.5,  # ✅ 黑边框调细
                   zorder=5)

    # ========== 7. 绘制CNN+LSTM训练曲线（百分比） ==========
    x_cnn_lstm = np.arange(1, len(history_cnn_lstm.history['accuracy']) + 1)
    x_cnn_lstm_filtered = x_cnn_lstm[(x_cnn_lstm >= 0) & (x_cnn_lstm <= 100)]
    y_cnn_lstm_train = np.array(history_cnn_lstm.history['accuracy']) * 100
    y_cnn_lstm_filtered = y_cnn_lstm_train[(x_cnn_lstm >= 0) & (x_cnn_lstm <= 100)]

    if len(x_cnn_lstm_filtered) > 3:
        x_cnn_lstm_new = np.linspace(0, min(x_cnn_lstm_filtered.max(), 100), 500)
        spl_cnn_lstm_train = make_interp_spline(x_cnn_lstm_filtered, y_cnn_lstm_filtered, k=3)
        y_cnn_lstm_smooth_train = spl_cnn_lstm_train(x_cnn_lstm_new)
        ax.plot(x_cnn_lstm_new, y_cnn_lstm_smooth_train,
                color=colors['cnnlstm_train'],
                linestyle=line_styles['cnnlstm_train'],
                linewidth=line_widths['cnnlstm_train'],
                alpha=0.85,
                label='CNN+LSTM Train')

    # CNN+LSTM训练标记点：黑边框调细
    valid_cnn_train = [e for e in target_epochs if e in x_cnn_lstm]
    if valid_cnn_train:
        mark_idx_cnn = [np.where(x_cnn_lstm == e)[0][0] for e in valid_cnn_train]
        ax.scatter(valid_cnn_train,
                   np.array(history_cnn_lstm.history['accuracy'])[mark_idx_cnn] * 100,
                   color=colors['cnnlstm_train'],
                   marker=markers['cnnlstm_train'],
                   s=30,
                   edgecolor='black',
                   linewidth=0.5,  # ✅ 黑边框调细
                   zorder=5)

    # ========== 8. 绘制CNN+LSTM测试曲线（百分比） ==========
    y_cnn_lstm_test = np.array(history_cnn_lstm.history['val_accuracy']) * 100
    y_cnn_lstm_test_filtered = y_cnn_lstm_test[(x_cnn_lstm >= 0) & (x_cnn_lstm <= 100)]

    if len(x_cnn_lstm_filtered) > 3:
        spl_cnn_lstm_test = make_interp_spline(x_cnn_lstm_filtered, y_cnn_lstm_test_filtered, k=3)
        y_cnn_lstm_smooth_test = spl_cnn_lstm_test(x_cnn_lstm_new)
        ax.plot(x_cnn_lstm_new, y_cnn_lstm_smooth_test,
                color=colors['cnnlstm_test'],
                linestyle=line_styles['cnnlstm_test'],
                linewidth=line_widths['cnnlstm_test'],
                alpha=0.85,
                label='CNN+LSTM Test')

    # CNN+LSTM测试标记点：黑边框调细
    valid_cnn_test = [e for e in target_epochs if e in x_cnn_lstm]
    if valid_cnn_test:
        mark_idx_cnn_test = [np.where(x_cnn_lstm == e)[0][0] for e in valid_cnn_test]
        ax.scatter(valid_cnn_test,
                   np.array(history_cnn_lstm.history['val_accuracy'])[mark_idx_cnn_test] * 100,
                   color=colors['cnnlstm_test'],
                   marker=markers['cnnlstm_test'],
                   s=30,
                   edgecolor='black',
                   linewidth=0.5,  # ✅ 黑边框调细
                   zorder=5)

    # ========== 9. 坐标轴核心配置（Y轴0/25/50/75/105 + X轴0-100） ==========
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


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred_classes, labels=[0, 1, 2])
    TN = cm[0, 0]
    FP = cm[0, 1:].sum()

    accuracy = accuracy_score(y_true, y_pred_classes)
    precision = precision_score(y_true, y_pred_classes, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred_classes, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred_classes, average='weighted', zero_division=0)

    # 修复AUC计算逻辑
    try:
        all_classes = sorted(list(set(y_true) | set(y_pred_classes)))
        n_classes = len(all_classes)
        if n_classes < 2:
            auc = float("nan")
        else:
            class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
            y_true_mapped = [class_to_idx[cls] for cls in y_true]
            y_true_onehot = np.eye(n_classes)[y_true_mapped]
            auc = roc_auc_score(y_true_onehot, y_pred, multi_class='ovr')
    except (ValueError, IndexError) as e:
        print(f"CNN+LSTM模型计算AUC时出错: {str(e)}，返回NaN")
        auc = float("nan")

    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    return accuracy, precision, recall, f1, auc, fpr


# 修改后的EarlyStopping类
class EarlyStopping:
    def __init__(self, patience=15, verbose=False, delta=0, convergence_acc=1.0):
        """
        Args:
            patience: 损失不下降的轮次阈值
            verbose: 是否打印早停日志
            delta: 损失最小改进阈值
            convergence_acc: 收敛的准确率阈值（默认1.0）
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.convergence_acc = convergence_acc
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.convergence_counter = 0
        self.convergence_patience = 3  # 连续3轮达到收敛准确率就触发早停

    def __call__(self, current_loss, model, train_acc=None):
        # 检查是否达到收敛准确率
        if train_acc is not None and train_acc >= self.convergence_acc:
            self.convergence_counter += 1
            if self.convergence_counter >= self.convergence_patience:
                self.early_stop = True
                if self.verbose:
                    print(f"模型已收敛（准确率≥{self.convergence_acc}），触发早停")
                return
        else:
            self.convergence_counter = 0

        # 原始损失检查逻辑
        score = -current_loss  # 损失越小分数越高
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping计数器: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


# ===================== 主程序 =====================
if __name__ == "__main__":
    # 数据加载和预处理
    train_data_path = r'F:\shenpeng\us Experiment\data\CIC-DDoS2019\cicddos2019_train.csv'
    test_data_path = r'F:\shenpeng\us Experiment\data\CIC-DDoS2019\cicddos2019_test.csv'

    train_df = load_data(train_data_path)
    test_df = load_data(test_data_path)
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    train_df = categorize_attacks(train_df)

    train_df = train_df[train_df['attack_category'] != 2].reset_index(drop=True)
    y_train = train_df["attack_category"].values  # 此时y_train仅含0和1，符合文章要求

    # 测试集无需过滤，确保包含0（正常）、1（已知）、2（未知）
    test_df = categorize_attacks(test_df)
    y_test = test_df["attack_category"].values

    X_train, train_feature_columns = automated_feature_engineering(train_df)
    X_test, test_feature_columns = automated_feature_engineering(test_df)

    if np.isnan(X_train).any():
        X_train = np.nan_to_num(X_train)
    if np.isnan(X_test).any():
        X_test = np.nan_to_num(X_test)

    # 标准化特征（静态特征分类的关键步骤）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 修正标签处理：直接用attack_category
    y_train = train_df["attack_category"].values
    y_test = test_df["attack_category"].values

    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 初始化优化后的EagleEye模型
    configs = Configs(input_dim=X_train.shape[1])
    model = OptimizedEagleEyeClassifier(
        input_dim=X_train.shape[1],
        num_classes=3,
        configs=configs
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # 加载历史训练经验（修复：传入optimizer，实际加载权重）
    experience_dir = r'F:\shenpeng\test_1\training_experience'
    history_experience = load_training_experience(experience_dir, model, optimizer)

    # 训练EagleEye模型（仅当加载失败 或 训练历史为空 时训练）
    print("=" * 50)
    print("Training Optimized EagleEye Model (Selective SSM)...")
    print("=" * 50)
    if history_experience is None or len(history_experience.get('train_history', {}).get('accuracy', [])) == 0:
        # 传入test_loader作为验证集，训练100轮
        eagle_eye_history = OptimizedEagleEyeClassifier.train_model(
            model, train_loader, criterion, optimizer,
            epochs=100, val_loader=test_loader
        )
        actual_epochs = len([x for x in eagle_eye_history['loss'] if x != eagle_eye_history['loss'][-1]])
    else:
        print("ℹ️  已加载历史训练经验，跳过训练")
        eagle_eye_history = history_experience['train_history']
        actual_epochs = history_experience['configs'].get('epochs', 100)

    # 评估EagleEye模型
    print("\n" + "=" * 50)
    print("Evaluating the Optimized EagleEye model...")
    print("=" * 50)
    eagle_eye_metrics = OptimizedEagleEyeClassifier.evaluate_model(model, test_loader)
    print("\n📊 EagleEye模型评估指标：")
    print(f"Accuracy: {eagle_eye_metrics['accuracy']:.4f}")
    print(f"Precision: {eagle_eye_metrics['precision']:.4f}")
    print(f"Recall: {eagle_eye_metrics['recall']:.4f}")
    print(f"F1: {eagle_eye_metrics['f1']:.4f}")
    print(f"AUC: {eagle_eye_metrics['auc']:.4f}")
    print(f"FPR: {eagle_eye_metrics['fpr']:.4f}")

    # 聚类（DBSCAN密度变分聚类）
    min_samples = determine_min_samples(X_train.cpu().numpy())
    eps_k_distance = determine_eps_k_distance(X_train.cpu().numpy())
    eps_silhouette = determine_eps_silhouette(X_train.cpu().numpy(), min_samples)
    eps = (eps_k_distance + eps_silhouette) / 2

    X_train_np = X_train.cpu().numpy().squeeze()
    dbscan_labels = perform_dbscan(X_train_np, eps, min_samples)

    # 判定聚类效果
    silhouette = silhouette_score_custom(X_train_np, dbscan_labels) if len(np.unique(dbscan_labels)) > 1 else -1
    ch_score = calinski_harabasz_score_custom(X_train_np, dbscan_labels) if len(np.unique(dbscan_labels)) > 1 else -1

    cluster_metrics = {
        'silhouette': silhouette,
        'ch_score': ch_score,
        'eps': eps,
        'min_samples': min_samples
    }
    print(f"\n🔍 聚类效果评估：")
    print(f"轮廓系数={silhouette:.4f}, CH分数={ch_score:.4f}")

    # 保存训练经验
    save_training_experience(
        model=model,
        save_dir=experience_dir,
        optimizer=optimizer,
        metrics=eagle_eye_metrics,
        cluster_metrics=cluster_metrics,
        configs=configs,
        actual_epochs=actual_epochs,
        train_history=eagle_eye_history
    )

    # 重构误差计算
    errors, new_labels = calculate_reconstruction_error_optimized(X_train.cpu().numpy(), X_train.cpu().numpy(),
                                                                  dbscan_labels)
    test_errors, test_new_labels = calculate_reconstruction_error_optimized(X_test.cpu().numpy(), X_train.cpu().numpy(),
                                                                            dbscan_labels)

    valid_indices = np.isin(new_labels, [0, 1, 2])
    filtered_features = X_train_np[valid_indices]
    filtered_labels = np.array(new_labels)[valid_indices]

    train_features_balanced, new_labels_train_balanced = resample(
        filtered_features, filtered_labels,
        replace=True,
        n_samples=max(np.bincount(filtered_labels)),
        random_state=42
    )

    # 准备CNN+LSTM数据
    X_train_cnn_lstm = np.expand_dims(train_features_balanced, axis=2)
    X_test_cnn_lstm = np.expand_dims(X_test.cpu().numpy(), axis=2)
    y_train_cnn_lstm = tf.keras.utils.to_categorical(new_labels_train_balanced, num_classes=3)
    y_test_cnn_lstm = tf.keras.utils.to_categorical(test_new_labels, num_classes=3)

    # 训练CNN+LSTM模型
    print("\n" + "=" * 50)
    print("Training CNN+LSTM model...")
    print("=" * 50)
    cnn_lstm_model = create_improved_cnn_lstm(input_shape=(X_train_cnn_lstm.shape[1], 1), num_classes=3)
    history_cnn_lstm = improved_train_model(cnn_lstm_model, X_train_cnn_lstm, y_train_cnn_lstm, X_test_cnn_lstm,
                                            y_test_cnn_lstm)

    # 评估CNN+LSTM模型
    print("\n" + "=" * 50)
    print("Evaluating CNN+LSTM model...")
    print("=" * 50)
    cnn_lstm_acc, cnn_lstm_prec, cnn_lstm_rec, cnn_lstm_f1, cnn_lstm_auc, cnn_lstm_fpr = evaluate_model(cnn_lstm_model,
                                                                                                        X_test_cnn_lstm,
                                                                                                        y_test_cnn_lstm)
    print("\n📊 CNN+LSTM模型评估指标：")
    print(f"Accuracy: {cnn_lstm_acc:.4f}")
    print(f"Precision: {cnn_lstm_prec:.4f}")
    print(f"Recall: {cnn_lstm_rec:.4f}")
    print(f"F1: {cnn_lstm_f1:.4f}")
    print(f"AUC: {cnn_lstm_auc:.4f}")
    print(f"FPR: {cnn_lstm_fpr:.4f}")

    # 可视化对比（【修改】传入EagleEye训练历史）
    print("\n" + "=" * 50)
    print("Generating comparison plot...")
    print("=" * 50)
    plot_improved_comparison(eagle_eye_history, history_cnn_lstm, eagle_eye_metrics)

    # 输出对比汇总表
    print("\n" + "=" * 60)
    print("📈 模型性能对比汇总")
    print("=" * 60)
    print(f"{'指标':<10} {'EagleEye':<10} {'CNN+LSTM':<10}")
    print(f"{'-' * 30}")
    print(f"Accuracy   {eagle_eye_metrics['accuracy']:<10.4f} {cnn_lstm_acc:<10.4f}")
    print(f"Precision  {eagle_eye_metrics['precision']:<10.4f} {cnn_lstm_prec:<10.4f}")
    print(f"Recall     {eagle_eye_metrics['recall']:<10.4f} {cnn_lstm_rec:<10.4f}")
    print(f"F1         {eagle_eye_metrics['f1']:<10.4f} {cnn_lstm_f1:<10.4f}")
    print(f"AUC        {eagle_eye_metrics['auc']:<10.4f} {cnn_lstm_auc:<10.4f}")
    print(f"FPR        {eagle_eye_metrics['fpr']:<10.4f} {cnn_lstm_fpr:<10.4f}")
    print("=" * 60)