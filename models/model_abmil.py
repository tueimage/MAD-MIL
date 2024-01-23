import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np
import math

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 512, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x
        

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
class ABMIL(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes=2):
        super(ABMIL, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:    
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
            
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        self.n_classes = n_classes
        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)

    def forward(self, h, attention_only=False):
        A, h = self.attention_net(h)  # NxK  
              
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        
        A = F.softmax(A, dim=1)  # softmax over N
        
        M = torch.mm(A, h) 
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        
        return logits, Y_prob, Y_hat, A_raw, {}
    
    





"""
Multihead ABMIL model with seperate attention modules.

Args:
    gate (bool): whether to use gated attention network
    size_arg (str): config for network size
    dropout (bool): whether to use dropout
    n_classes (int): number of classes
    n (int): number of attention heads
    head_size (str): size of each head
"""
            
class ABMIL_Multihead(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=False, n_classes=2, n= 2, head_size="small"):
        super(ABMIL_Multihead, self).__init__()
        self.n_heads = n
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384], "tiny": [1024, 128, 16]}
        self.size = self.size_dict[size_arg]
        
        if self.size[1] % self.n_heads != 0:
            print("The feature dim should be divisible by num_heads!! Do't worry, we will fix it for you.")
            self.size[1] = math.ceil(self.size[1] / self.n_heads) * self.n_heads
                           
        self.step = self.size[1] // self.n_heads  
        
        if head_size == "tiny":
            self.dim = self.step // 4
        elif head_size == "small":  
            self.dim = self.step // 2
        elif head_size == "big":
            self.dim = self.size[2]
        else:
            self.dim = self.step    
        
        fc = [nn.Linear(self.size[0], self.size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
            
        if gate:
            att_net = [Attn_Net_Gated(L=self.step, D=self.dim, dropout=dropout, n_classes=1) for ii in range(self.n_heads)]
        else:    
            att_net = [Attn_Net(L=self.step, D=self.dim, dropout=dropout, n_classes=1) for ii in range(self.n_heads)]

        self.net_general = nn.Sequential(*fc)
        self.attention_net =  nn.ModuleList(att_net)
        self.classifiers = nn.Linear(self.size[1], n_classes) 
        self.n_classes = n_classes
        initialize_weights(self)

    def relocate(self):
        """
        Relocates the model to GPU if available, else to CPU.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net_general = self.net_general.to(device)
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        
    def forward(self, h, attention_only=False):
        """
        Forward pass of the model.

        Args:
            h (torch.Tensor): Input tensor
            attention_only (bool): Whether to return only attention weights

        Returns:
            tuple: Tuple containing logits, predicted probabilities, predicted labels, attention weights, and attention weights before softmax
        """
        device = h.device
        
        h = self.net_general(h)
        N, C = h.shape
        
        # Multihead Input
        h = h.reshape(N, self.n_heads, C // self.n_heads) 
        
        A = torch.empty(N, self.n_heads, 1).float().to(device) 
        for nn in range(self.n_heads):
            a,_ = self.attention_net[nn](h[:,nn,:])
            A [:, nn, :] = a
            
        A = torch.transpose(A, 2, 0)  # KxheadsxN
        if attention_only:
            return A
        A_raw = A
           
        A = F.softmax(A, dim=-1)  # softmax over N     
        
        # Multihead Output
        M = torch.empty(1, self.size[1]).float().to(device) 
        for nn in range(self.n_heads):  
            m = torch.mm(A[:, nn, :], h[:, nn, :])
            M[:, self.step * nn: self.step * nn + self.step] = m
                       
        # Singlehead Classification
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)  
        return logits, Y_prob, Y_hat, A_raw, A 