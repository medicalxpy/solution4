"""
早停机制

实现模型训练的早停机制
"""

import numpy as np
import torch
import os


class EarlyStopping:
    """
    早停类：当验证损失不再改善时停止训练
    """

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        初始化早停。
        
        参数:
            patience: 在停止前等待改善的epoch数量
            verbose: 是否打印消息
            delta: 被认为是改善的最小变化量
            path: 保存模型的路径
            trace_func: 用于打印信息的函数
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        """
        调用实例，基于验证损失评估模型表现。
        
        参数:
            val_loss: 验证损失
            model: 模型实例
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        保存模型检查点。
        
        参数:
            val_loss: 验证损失
            model: 模型实例
        """
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
            
        directory = os.path.dirname(self.path)
        if not os.path.exists(directory) and directory != '':
            os.makedirs(directory)
            
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss