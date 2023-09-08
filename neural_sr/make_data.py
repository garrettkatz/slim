import torch
import numpy as np

# class DataSet(torch.nn.data.Dataset):
    # def __init__(self, x, y, w_prev, w_next):
    #     self.x = torch.FloatTensor(np.array(x))
    #     self.y = torch.FloatTensor(np.array(y))
    #     self.w_prev = torch.FloatTensor(np.array(w_prev))
    #     self.w_next = torch.FloatTensor(np.array(w_next))
    #     self.len = self.x.shape[0]
    
    # def __getitem__(self, index):
    #     return self.x[index], self.y[index], self.w_prev[index], self.w_next[index]

    # def __len__(self):
    #     return self.len

class DataSet(torch.nn.data.Dataset):
    def __init__(self, input):
        self.input = torch.FloatTensor(input)
        self.len = self.input.shape[0]
    
    def __getitem__(self, index):
        return self.input[index]
    
    def __len__(self):
        return self.len