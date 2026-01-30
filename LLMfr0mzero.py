import torch.nn as nn


class Config:
    def __init__(self):
        self.__n_dimension = 128
        self.__n_head = 4
        self.__n_layer = 4
        self.__context_len = 32
        self.__dictionary_size = 1000
        self.__dropout = 0.1
class self_attention(nn.Module):
    def __init__(self,config):
        super().__init__()


