import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GraphConvolution(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1)
        self.adj = torch.tensor(get_smpl_adj(), dtype=torch.float32).unsqueeze(0).to(device)
        
    def forward(self, x):
        x = self.linear(x)
        x = torch.matmul(x, self.adj)
        return x
    

class GCN(nn.Module):

    def __init__(self):
        super(GCN, self).__init__()
        self.dim = 256
        self.gcn1 = GraphConvolution(3, self.dim)
        self.gcn2 = GraphConvolution(self.dim, self.dim)
        self.gcn3 = GraphConvolution(self.dim, self.dim)
        
    def forward(self, x):
        x = F.relu(self.gcn1(x))
        x = F.relu(self.gcn2(x))
        x = F.relu(self.gcn3(x))
        return x
    
    
def get_smpl_skeleton():
    return np.array(
        [
            [0, 1], 
            [0, 2], 
            [0, 3], 
            [1, 4], 
            [2, 5], 
            [3, 6], 
            [4, 7], 
            [5, 8], 
            [6, 9], 
            [7, 10], 
            [8, 11], 
            [9, 12], 
            [9, 13], 
            [9, 14], 
            [12, 15], 
            [13, 16], 
            [14, 17], 
            [16, 18],
            [17, 19],
            [18, 20],
            [19, 21],
            [20, 22],
            [21, 23],
        ]
    )

def get_smpl_adj():
    skeleton = get_smpl_skeleton()
    num_joint = len(skeleton)+1
    adj = np.zeros((num_joint, num_joint))
    for r, c in skeleton:
        adj[r, c] = 1.
        adj[c, r] = 1.
    for i in range(len(adj)):
        adj[i, i] = 1.
    adj = adj / (np.sum(adj, axis=0, keepdims=True) + 1e-9)
    return adj