import os
from time import time

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader

from feature import SampleEntity

from TreeConvolution.tcnn import (BinaryTreeConv, DynamicPooling,
                                  TreeActivation, TreeLayerNorm)
from TreeConvolution.util import prepare_trees

# CUDA = torch.cuda.is_available()
CUDA = False
GPU_LIST = [0, 1, 2, 3, 4, 5, 6, 7]
# GPU_LIST = [0]
torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device("cuda:0" if CUDA else "cpu")


def _nn_path(base):
    return os.path.join(base, "nn_weights")

def _feature_generator_path(base):
    return os.path.join(base, "feature_generator")

def _input_feature_dim_path(base):
    return os.path.join(base, "input_feature_dim")

def collate_fn(x):
    trees = []
    targets = []

    for tree, target in x:
        trees.append(tree)
        targets.append(target)

    targets = torch.tensor(targets)
    return trees, targets

def collate_pairwise_fn(x):
    return zip(*x)


def transformer(x: SampleEntity):
    return x.get_feature()

def left_child(x: SampleEntity):
    return x.get_left()

def right_child(x: SampleEntity):
    return x.get_right()


class LeroNet_ori(nn.Module):
    def __init__(self, input_feature_dim) -> None:
        super(LeroNet, self).__init__()
        self.input_feature_dim = input_feature_dim
        self._cuda = False
        self.device = None

        self.tree_conv = nn.Sequential(
            BinaryTreeConv(self.input_feature_dim, 256),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            DynamicPooling(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, trees):
        return self.tree_conv(trees)

    def build_trees(self, feature):
        return prepare_trees(feature, transformer, left_child, right_child, cuda=self._cuda, device=self.device)

    def cuda(self, device):
        self._cuda = True
        self.device = device
        return super().cuda()


class LeroModel():
    def __init__(self, feature_generator) -> None:
        self._net = None
        self._feature_generator = feature_generator
        self._input_feature_dim = None
        self._model_parallel = None

    def load(self, path):
        with open(_input_feature_dim_path(path), "rb") as f:
            self._input_feature_dim = joblib.load(f)

        self._net = LeroNet(self._input_feature_dim)
        if CUDA:
            self._net.load_state_dict(torch.load(_nn_path(path)))
        else:
            self._net.load_state_dict(torch.load(
                _nn_path(path), map_location=torch.device('cpu')))
        self._net.eval()

        with open(_feature_generator_path(path), "rb") as f:
            self._feature_generator = joblib.load(f)

    def save(self, path):
        os.makedirs(path, exist_ok=True)

        if CUDA:
            torch.save(self._net.module.state_dict(), _nn_path(path))
        else:
            torch.save(self._net.state_dict(), _nn_path(path))

        with open(_feature_generator_path(path), "wb") as f:
            joblib.dump(self._feature_generator, f)
        with open(_input_feature_dim_path(path), "wb") as f:
            joblib.dump(self._input_feature_dim, f)

    def fit(self, X, Y, pre_training=False):
        if isinstance(Y, list):
            Y = np.array(Y)
            Y = Y.reshape(-1, 1)

        batch_size = 64
        if CUDA:
            batch_size = batch_size * len(GPU_LIST)

        pairs = []
        for i in range(len(Y)):
            pairs.append((X[i], Y[i]))
        dataset = DataLoader(pairs,
                             batch_size=batch_size,
                             shuffle=True,
                             collate_fn=collate_fn)

        if not pre_training:
            # # determine the initial number of channels
            input_feature_dim = len(X[0].get_feature())
            print("input_feature_dim:", input_feature_dim)

            self._net = LeroNet(input_feature_dim)
            self._input_feature_dim = input_feature_dim
            if CUDA:
                self._net = self._net.cuda(device)
                self._net = torch.nn.DataParallel(
                    self._net, device_ids=GPU_LIST)
                self._net.cuda(device)

        optimizer = None
        if CUDA:
            optimizer = torch.optim.Adam(self._net.module.parameters())
            optimizer = nn.DataParallel(optimizer, device_ids=GPU_LIST)
        else:
            optimizer = torch.optim.Adam(self._net.parameters())

        loss_fn = torch.nn.MSELoss()
        losses = []
        start_time = time()
        for epoch in range(100):
            loss_accum = 0
            for x, y in dataset:
                if CUDA:
                    y = y.cuda(device)

                tree = None
                if CUDA:
                    tree = self._net.module.build_trees(x)
                else:
                    tree = self._net.build_trees(x)

                y_pred = self._net(tree)
                loss = loss_fn(y_pred, y)
                loss_accum += loss.item()

                if CUDA:
                    optimizer.module.zero_grad()
                    loss.backward()
                    optimizer.module.step()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            loss_accum /= len(dataset)
            losses.append(loss_accum)

            print("Epoch", epoch, "training loss:", loss_accum)
        print("training time:", time() - start_time, "batch size:", batch_size)

    def predict(self, x):
        if CUDA:
            self._net = self._net.cuda(device)

        if not isinstance(x, list):
            x = [x]

        tree = None
        if CUDA:
            tree = self._net.module.build_trees(x)
        else:
            tree = self._net.build_trees(x)

        pred = self._net(tree)[0].cpu().detach().numpy()
        return pred


class LeroModelPairWise(LeroModel):
    def __init__(self, feature_generator) -> None:
        super().__init__(feature_generator)

    def fit(self, X1, X2, Y1, Y2, pre_training=False):
        assert len(X1) == len(X2) and len(Y1) == len(Y2) and len(X1) == len(Y1)
        if isinstance(Y1, list):
            Y1 = np.array(Y1)
            Y1 = Y1.reshape(-1, 1)
        if isinstance(Y2, list):
            Y2 = np.array(Y2)
            Y2 = Y2.reshape(-1, 1)

        # # determine the initial number of channels
        if not pre_training:
            input_feature_dim = len(X1[0].get_feature())
            print("input_feature_dim:", input_feature_dim)

            self._net = LeroNet(input_feature_dim)
            self._input_feature_dim = input_feature_dim
            if CUDA:
                self._net = self._net.cuda(device)
                self._net = torch.nn.DataParallel(
                    self._net, device_ids=GPU_LIST)
                self._net.cuda(device)

        pairs = []
        for i in range(len(X1)):
            pairs.append((X1[i], X2[i], 1.0 if Y1[i] >= Y2[i] else 0.0))

        batch_size = 64
        if CUDA:
            batch_size = batch_size * len(GPU_LIST)

        dataset = DataLoader(pairs,
                             batch_size=batch_size,
                             shuffle=True,
                             collate_fn=collate_pairwise_fn)

        optimizer = None
        if CUDA:
            optimizer = torch.optim.Adam(self._net.module.parameters())
            optimizer = nn.DataParallel(optimizer, device_ids=GPU_LIST)
        else:
            optimizer = torch.optim.Adam(self._net.parameters())

        bce_loss_fn = torch.nn.BCELoss()

        losses = []
        sigmoid = nn.Sigmoid()
        start_time = time()
        for epoch in range(100):
            loss_accum = 0
            for x1, x2, label in dataset:

                tree_x1, tree_x2 = None, None
                if CUDA:
                    tree_x1 = self._net.module.build_trees(x1)
                    tree_x2 = self._net.module.build_trees(x2)
                else:
                    tree_x1 = self._net.build_trees(x1)
                    tree_x2 = self._net.build_trees(x2)

                # pairwise
                y_pred_1, inter_fea1 = self._net(tree_x1)
                y_pred_2, inter_fea2 = self._net(tree_x2)
                diff = y_pred_1 - y_pred_2
                prob_y = sigmoid(diff)

                label_y = torch.tensor(np.array(label).reshape(-1, 1))
                if CUDA:
                    label_y = label_y.cuda(device)

                loss = bce_loss_fn(prob_y, label_y)
                loss_accum += loss.item()

                if CUDA:
                    optimizer.module.zero_grad()
                    loss.backward()
                    optimizer.module.step()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            loss_accum /= len(dataset)
            losses.append(loss_accum)

            print("Epoch", epoch, "training loss:", loss_accum)
        print("training time:", time() - start_time, "batch size:", batch_size)

    def get_inter_fea(self, x1, x2):
        tree_x1, tree_x2 = None, None
        if CUDA:
            tree_x1 = self._net.module.build_trees(x1)
            tree_x2 = self._net.module.build_trees(x2)
        else:
            tree_x1 = self._net.build_trees(x1)
            tree_x2 = self._net.build_trees(x2)

        # pairwise
        _, inter_fea1 = self._net(tree_x1)
        _, inter_fea2 = self._net(tree_x2)
        return inter_fea1, inter_fea2


class LeroNet(nn.Module):
    def __init__(self, input_feature_dim) -> None:
        super(LeroNet, self).__init__()
        self.input_feature_dim = input_feature_dim
        self._cuda = False
        self.device = None

        self.tree_conv = nn.Sequential(
            BinaryTreeConv(self.input_feature_dim, 256),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            DynamicPooling(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )


    def forward(self, trees):
        inter_fea = self.tree_conv(trees)
        return self.fc(inter_fea), inter_fea

    def build_trees(self, feature):
        return prepare_trees(feature, transformer, left_child, right_child, cuda=self._cuda, device=self.device)

    def cuda(self, device):
        self._cuda = True
        self.device = device
        return super().cuda()

    
class BayesianNet(nn.Module):
    def __init__(self, input_feature_dim) -> None:
        super(LeroNet, self).__init__()
        self.input_feature_dim = input_feature_dim
        self._cuda = False
        self.device = None

        self.tree_conv = nn.Sequential(
            BinaryTreeConv(self.input_feature_dim, 256),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            DynamicPooling(),
        )


    def forward(self, trees):
        return self.tree_conv(trees)

    def build_trees(self, feature):
        return prepare_trees(feature, transformer, left_child, right_child, cuda=self._cuda, device=self.device)

    def cuda(self, device):
        self._cuda = True
        self.device = device
        return super().cuda()
    
def custom_nll_loss(logits, log_variance, targets, alpha):
    # 将 logits 转换为概率
    probs = torch.sigmoid(logits)
    variance = torch.exp(log_variance)
    # 计算损失
    loss = alpha * (0.5 * log_variance + ((targets - probs) ** 2) / (2 * variance))
    return loss.mean()

class BayesianHead(nn.Modeule):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x1, x2):
        x = torch.concatenate(x1, x2)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x[:, 0], x[:, 1]


class BayesianModelPairWise(LeroModel):
    def __init__(self, feature_generator, gamma=0.2, delta_threshold=0.1) -> None:
        super().__init__(feature_generator)
        self.gamma = gamma
        self.delta_threshold = delta_threshold
        self.head = BayesianHead()

    def fit(self, X1, X2, Y1, Y2, pre_training=False):
        assert len(X1) == len(X2) and len(Y1) == len(Y2) and len(X1) == len(Y1)
        if isinstance(Y1, list):
            Y1 = np.array(Y1)
            Y1 = Y1.reshape(-1, 1)
        if isinstance(Y2, list):
            Y2 = np.array(Y2)
            Y2 = Y2.reshape(-1, 1)
        

        # # determine the initial number of channels
        if not pre_training:
            input_feature_dim = len(X1[0].get_feature())
            print("input_feature_dim:", input_feature_dim)

            self.net = BayesianNet(input_feature_dim)
            self._input_feature_dim = input_feature_dim
            if CUDA:
                self.net = self.net.cuda(device)
                self.net = torch.nn.DataParallel(
                    self.net, device_ids=GPU_LIST)
                self.head = torch.nn.DataParallel(self.head, device_ids=GPU_LIST)
                self.net.cuda(device)
                self.head.cuda(device)

        pairs = []
        for i in range(len(X1)):
            pairs.append((X1[i], X2[i], 1.0 if Y1[i] >= Y2[i] else 0.0, self.gamma if abs(Y1[i]-Y2[i]) < self.delta_threshold else 1. ))

        batch_size = 64
        if CUDA:
            batch_size = batch_size * len(GPU_LIST)

        dataset = DataLoader(pairs,
                             batch_size=batch_size,
                             shuffle=True,
                             collate_fn=collate_pairwise_fn)

        optimizer = None
        if CUDA:
            optimizer = torch.optim.Adam(list(self.net.module.parameters()) + list(self.head.module.parameters()))
            optimizer = nn.DataParallel(optimizer, device_ids=GPU_LIST)
        else:
            optimizer = torch.optim.Adam(self.net.parameters())

        losses = []
        start_time = time()
        for epoch in range(100):
            loss_accum = 0
            for x1, x2, label, alpha in dataset:

                tree_x1, tree_x2 = None, None
                if CUDA:
                    tree_x1 = self.net.module.build_trees(x1)
                    tree_x2 = self.net.module.build_trees(x2)
                else:
                    tree_x1 = self.net.build_trees(x1)
                    tree_x2 = self.net.build_trees(x2)

                # pairwise
                inter_fea1 = self.net(tree_x1)
                inter_fea2 = self.net(tree_x2)
                prob, log_variance = self.head(inter_fea1, inter_fea2)

                label_y = torch.tensor(np.array(label).reshape(-1, 1))
                if CUDA:
                    label_y = label_y.cuda(device)

                loss = custom_nll_loss(prob, log_variance, label_y, alpha)
                loss_accum += loss.item()

                if CUDA:
                    optimizer.module.zero_grad()
                    loss.backward()
                    optimizer.module.step()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            loss_accum /= len(dataset)
            losses.append(loss_accum)

            print("Epoch", epoch, "training loss:", loss_accum)
        print("training time:", time() - start_time, "batch size:", batch_size)

    def get_inter_fea(self, x1, x2):
        tree_x1, tree_x2 = None, None
        if CUDA:
            tree_x1 = self.net.module.build_trees(x1)
            tree_x2 = self.net.module.build_trees(x2)
        else:
            tree_x1 = self.net.build_trees(x1)
            tree_x2 = self.net.build_trees(x2)

        # pairwise
        _, inter_fea1 = self.net(tree_x1)
        _, inter_fea2 = self.net(tree_x2)
        return inter_fea1, inter_fea2
