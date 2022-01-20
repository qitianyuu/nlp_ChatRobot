"""
# File       :  settings.py
# Time       :  2022/1/20 4:23 下午
# Author     : Qi
# Description: 配置文件
"""
import torch


config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'train_frac': 0.85
}