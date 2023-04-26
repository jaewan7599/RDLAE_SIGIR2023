"""
Define models here
"""
import world
import torch
from dataloader import BasicDataset
import dataloader
from torch import nn
import numpy as np
from scipy import sparse
from time import time


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError


class RLAE(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):
        super(RLAE, self).__init__()

        self.dataset : dataloader.BasicDataset = dataset
        self.config = config
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        
        self.reg_p = config['reg_p']               
        self.xi = config['xi']
        
        self.__init_weight()
    
    def __init_weight(self):
        X = self.dataset.UserItemNet
        self.train_matrix = X

        train_start = time()

        G = np.array(X.T.dot(X).toarray())
        G[np.diag_indices(self.num_items)] += self.reg_p
        P = np.linalg.inv(G)
        diag_P = np.diag(P)
        
        condition = (1 - self.reg_p * diag_P) > self.xi
        assert condition.sum() > 0
        lagrangian = ((1 - self.xi) / diag_P - self.reg_p) * condition.astype(float)

        self.W = P * -(lagrangian + self.reg_p)
        self.W[np.diag_indices(self.num_items)] = 0

        train_end = time()
        self.train_time = train_end - train_start
        print(f"costing {self.train_time}s for training")

    def getUsersRating(self, users):
        users = users.detach().cpu().numpy()

        input_matrix = np.array(self.train_matrix[users].toarray())
        eval_output = input_matrix @ self.W

        return torch.FloatTensor(eval_output)


class RDLAE(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):
        super(RDLAE, self).__init__()

        self.dataset : dataloader.BasicDataset = dataset
        self.config = config
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        
        self.reg_p = config['reg_p']
        self.drop_p = config['drop_p']
        self.xi = config['xi']
        
        self.__init_weight()
    
    def __init_weight(self):
        X = self.dataset.UserItemNet
        self.train_matrix = X
        
        train_start = time()

        G = np.array(X.T.dot(X).toarray())
        gamma = np.diag(G) * self.drop_p / (1 - self.drop_p) + self.reg_p
        G[np.diag_indices(self.num_items)] += gamma
        C = np.linalg.inv(G)
        diag_C = np.diag(C)
                
        condition = (1 - gamma * diag_C) > self.xi
        assert condition.sum() > 0
        lagrangian = ((1 - self.xi) / diag_C - gamma) * condition.astype(float)

        self.W = C * -(gamma + lagrangian)
        self.W[np.diag_indices(self.num_items)] = 0

        train_end = time()
        self.train_time = train_end - train_start
        print(f"costing {self.train_time}s for training")

    def getUsersRating(self, users):
        users = users.detach().cpu().numpy()

        input_matrix = np.array(self.train_matrix[users].toarray())
        eval_output = input_matrix @ self.W

        return torch.FloatTensor(eval_output)


class EASE(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):
        super(EASE, self).__init__()

        self.dataset : dataloader.BasicDataset = dataset
        self.config = config
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        
        self.reg_p = config['reg_p']
        self.diag_const = config['diag_const']
        
        self.__init_weight()
    
    def __init_weight(self):
        X = self.dataset.UserItemNet
        self.train_matrix = X
        
        train_start = time()

        G = np.array(X.T.dot(X).toarray())
        G[np.diag_indices(self.num_items)] += self.reg_p
        P = np.linalg.inv(G)

        if self.diag_const:
            self.W = P / (-np.diag(P))
        else:
            self.W = -P * self.reg_p

        self.W[np.diag_indices(self.num_items)] = 0

        train_end = time()
        self.train_time = train_end - train_start
        print(f"costing {self.train_time}s for training")

    def getUsersRating(self, users):
        users = users.detach().cpu().numpy()

        input_matrix = np.array(self.train_matrix[users].toarray())
        eval_output = input_matrix @ self.W

        return torch.FloatTensor(eval_output)


class EDLAE(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):
        super(EDLAE, self).__init__()

        self.dataset : dataloader.BasicDataset = dataset
        self.config = config
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        
        self.reg_p = config['reg_p']
        self.drop_p = config['drop_p']
        self.diag_const = config['diag_const']
        
        self.__init_weight()
    
    def __init_weight(self):
        X = self.dataset.UserItemNet
        self.train_matrix = X
        
        train_start = time()

        G = np.array(X.T.dot(X).toarray())
        gamma = np.diag(G) * self.drop_p / (1 - self.drop_p) + self.reg_p

        G[np.diag_indices(self.num_items)] += gamma
        C = np.linalg.inv(G)

        if self.diag_const:
            self.W = C / (-np.diag(C))
        else:
            self.W = -C * gamma

        self.W[np.diag_indices(self.num_items)] = 0

        train_end = time()
        self.train_time = train_end - train_start
        print(f"costing {self.train_time}s for training")

    def getUsersRating(self, users):
        users = users.detach().cpu().numpy()

        input_matrix = np.array(self.train_matrix[users].toarray())
        eval_output = input_matrix @ self.W

        return torch.FloatTensor(eval_output)
    

class GFCF(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):
        super(GFCF, self).__init__()

        self.dataset = dataset
        self.adj_mat = dataset.UserItemNet.tolil()
        self.__init_weight()
    
    def __init_weight(self):
        adj_mat = self.adj_mat
        
        train_start = time()
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sparse.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sparse.diags(d_inv)
        self.d_mat_i = d_mat
        self.d_mat_i_inv = sparse.diags(1/d_inv)
        norm_adj = norm_adj.dot(d_mat)
        self.norm_adj = norm_adj.tocsc()
        
        import scipy.sparse.linalg as linalg
        _, _, self.vt = linalg.svds(self.norm_adj, 256)
        
        train_end = time()
        self.train_time = train_end - train_start
        print(f"costing {self.train_time}s for training")
        
    def getUsersRating(self, users):
        users = users.detach().cpu().numpy()

        norm_adj = self.norm_adj
        adj_mat = self.adj_mat
        batch_test = np.array(adj_mat[users,:].todense())
        U_2 = batch_test @ norm_adj.T @ norm_adj

        if world.dataset == 'abook':
            ret = U_2
        else:
            U_1 = batch_test @ self.d_mat_i @ self.vt.T @ self.vt @ self.d_mat_i_inv
            ret = U_2 + 0.3 * U_1
        
        return torch.FloatTensor(ret)
        