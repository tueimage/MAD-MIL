
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc as calc_auc
from matplotlib.patches import Rectangle

plt.style.use("ggplot")


"""Reproducible results"""
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



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
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class ABMIL(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes=2):
        super(ABMIL, self).__init__()
        self.size_dict = {"small": [784, 128, 256]}
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
        h = h.view(-1, 28*28)
        
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
        
        return logits, Y_prob, Y_hat, A_raw, A   
    


"""
Multihead ABMIL model with seperate attention modules.

Args:
    gate (bool): whether to use gated attention network
    size_arg (str): config for network size
    dropout (bool): whether to use dropout
    n_classes (int): number of classes
    temp (list): temperature scaling values for each head
    head_size (str): size of each head
"""
            
class ABMIL_Multihead(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=False, n_classes=2, temp=[1, 1], head_size="small"):
        super(ABMIL_Multihead_Sep, self).__init__()
        self.n_heads = len(temp)
        self.size_dict = {"small": [784, 128, 256]}
        self.size = self.size_dict[size_arg]
        self.temp = temp
        
        if self.size[1] % self.n_heads != 0:
            print("The feature dim should be divisible by num_heads!! Do't worry, we will fix it for you.")
            self.size[1] = math.ceil(self.size[1] / self.n_heads) * self.n_heads
                           
        self.step = self.size[1] // self.n_heads  
        
        if head_size == "tiny":
            self.dim = self.step // 4
        elif head_size == "small":  
            self.dim = self.step // 2
        elif head_size == "same":
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
        h = h.view(-1, 28*28)
        
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
        
        # Temperature scaling
        for ii, tt in enumerate(self.temp):
            A[:, ii, :] =  A[:, ii, :] / tt
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
        return logits, Y_prob, Y_hat, A_raw, A[:, 0, :] 
    

        
    
"""
args:
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Max_Pool(nn.Module):
    def __init__(self, size_arg = "small", dropout = False, n_classes=2):
        super(Max_Pool, self).__init__()
        self.size_dict = {"small": [784, 128, 256]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
            
        self.feature = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        self.n_classes = n_classes
        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature= self.feature.to(device)
        self.classifiers = self.classifiers.to(device)

    def forward(self, h):
        h = h.view(-1, 28*28)

        h = self.feature(h)  # Nxsize[1]       
        
        M = torch.max(h, dim= 0, keepdim= True)[0] # 1xsize[1]
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        
        return logits, Y_prob, Y_hat, {}, {}



"""
args:
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Mean_Pool(nn.Module):
    def __init__(self, size_arg = "small", dropout = False, n_classes=2):
        super(Mean_Pool, self).__init__()
        self.size_dict = {"small": [784, 128, 256]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
            
        self.feature = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        self.n_classes = n_classes
        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature= self.feature.to(device)
        self.classifiers = self.classifiers.to(device)

    def forward(self, h):
        h = h.view(-1, 28*28)

        h = self.feature(h)  # Nxsize[1]       
        
        M = torch.mean(h, dim= 0, keepdim= True) # 1xsize[1]
                
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        
        return logits, Y_prob, Y_hat, {}, {}



def create_soft_bags(input_data, input_labels, positive_class, bag_count, 
                     instance_count, seed= None, pos_perc= 0.5, neg_perc= 0.2,
                     pos_label= 1, neg_label= 0):

    # Set up bags.
    bags = []
    bag_labels = []
    num_pos= int(pos_perc*instance_count)
    num_neg= int(neg_perc*instance_count)

    # Normalize input data.
    input_data = np.divide(input_data, 255.0)

    # Count positive samples.
    count = 0
    pos_count= 0
    neg_count= 0
    np.random.seed(seed)
    
    key_instances_idx= np.where(input_labels==positive_class)[0]
    non_key_instances_idx= np.where(input_labels != positive_class)[0]
    
    while count < bag_count:
        
        instance_count_key = np.random.choice([num_pos, num_neg], 1, replace= False)[0]
        instance_count_non_key = instance_count - instance_count_key
                
        # Pick a fixed size random subset of samples.
        index_key = np.random.choice(key_instances_idx, instance_count_key, replace=False)
        index_non_key = np.random.choice(non_key_instances_idx, instance_count_non_key, replace=False)
        
        instances_data_key = input_data[index_key]
        instances_labels_key = input_labels[index_key]

        instances_data_non_key = input_data[index_non_key]
        instances_labels_non_key = input_labels[index_non_key]
        
        instances_data = np.concatenate((instances_data_key, instances_data_non_key)
                                        , axis= 0)
        instances_labels = np.concatenate((instances_labels_key, instances_labels_non_key)
                                          , axis= 0)
        
        instances_data , instances_labels = shuffle(instances_data, instances_labels
                                                    , random_state= seed+1)


        # Check how many positive classese are in the bag.
        if (instances_labels == positive_class).sum() == num_pos:
            bag_label = pos_label
            count += 1
            bags.append(instances_data)
            bag_labels.append(np.array([bag_label]))
            pos_count += 1
            
        elif (instances_labels == positive_class).sum() == num_neg:
            bag_label = neg_label
            count += 1
            bags.append(instances_data)
            bag_labels.append(np.array([bag_label]))
            neg_count += 1
        else:
            pass
        
    print(f"Positive bags: {pos_count}")
    print(f"Negative bags: {neg_count}\n")

    return (np.array(bags), np.reshape(np.array(bag_labels), (-1,)))
    


def print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)



def initialize_weights(module):
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
		
		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)


def get_optim(model, args):
	if args.opt == "adam":
 		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)

	elif args.opt == 'sgd':
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
	else:
		raise NotImplementedError
	return optimizer


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss
        


class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        self.data_all = {'y_true':[],'y_pred':[]}

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

        self.data_all['y_true'].append(Y)
        self.data_all['y_pred'].append(Y_hat)

    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        self.data_all['y_true'].append(Y)
        self.data_all['y_pred'].append(Y_hat)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

    def get_f1(self):
        y_true = np.asarray(self.data_all['y_true']).reshape(-1,)
        y_pred = np.asarray(self.data_all['y_pred']).reshape(-1,)
        f1 = f1_score(y_true,y_pred,average='macro')
        return f1


def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()
	return error


def make_weights_for_balanced_classes(bags, nclasses= 2):
    n_bags = len(bags)
    count_per_class = [0] * nclasses
    for _, bag_class in bags:
        count_per_class[bag_class] += 1
    weight_per_class = [0.] * nclasses
    for i in range(nclasses):
        weight_per_class[i] = float(n_bags) / float(count_per_class[i])
    weights = [0] * n_bags
    for idx, (bag, bag_class) in enumerate(bags):
        weights[idx] = weight_per_class[bag_class]        
    return weights

        
def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur)) 
    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets

    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
      
    if args.model_type == 'abmil':
        model = ABMIL(**model_dict)  
    if args.model_type == 'abmil_multihead':
        model = ABMIL_Multihead(**model_dict, temp= args.temp)               
    elif args.model_type == 'max_pool':
        model = Max_Pool(**model_dict)
    elif args.model_type == 'mean_pool':
        model = Mean_Pool(**model_dict)    
        
    
    model.relocate()
    print('Done!')
    print_network(model)
        
    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    dataset_train,  dataset_val,  dataset_test = datasets
    
    # For unbalanced dataset we create a weighted sampler                       
    weights = make_weights_for_balanced_classes(dataset_train)                                                                
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))                     
                                         
    train_loader = DataLoader(dataset_train, batch_size= 1, sampler= sampler)
    val_loader = DataLoader(dataset_val, batch_size= 1, shuffle= False)
    test_loader = DataLoader(dataset_test, batch_size= 1, shuffle= False)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 5, stop_epoch=10, verbose=True)
    else:
        early_stopping = None
        
    print('Done!')
    for epoch in range(args.max_epochs):
        train_loop(epoch, model, train_loader, optimizer, args.n_classes, loss_fn)
        stop = validate(cur, epoch, model, val_loader, args.n_classes, early_stopping, loss_fn,
                        args.results_dir, args.exp_name)            
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, args.exp_name, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, args.exp_name, "s_{}_checkpoint.pt".format(cur)))
      
    val_error, val_loss, val_auc, val_logger, val_atts, _= summary(model, val_loader, args.n_classes, loss_fn, args.BAG_SIZE, args)
    print('\nVal error: {:.4f}, Val loss: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_loss, val_auc))
    val_f1 = val_logger.get_f1()
    
    test_error, test_loss, test_auc, acc_logger, test_atts, _ = summary(model, test_loader, args.n_classes, loss_fn, args.BAG_SIZE, args)
    print('\nTest error: {:.4f}, Test loss: {:.4f}, ROC AUC: {:.4f} \n'.format(test_error, test_loss, test_auc))
    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
    test_f1 = acc_logger.get_f1() 
    
    return test_auc, val_auc, test_f1, val_f1, test_loss, val_loss, test_atts, val_atts



def train_loop(epoch, model, loader, optimizer, n_classes, loss_fn = None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.

    train_error = 0.
    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        data, label = data.to(torch.float32), label.to(torch.long) 
        
        logits, Y_prob, Y_hat, _, _ = model(data)
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)       
        loss_value = loss.item()
        
        train_loss += loss_value
        
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
       



def validate(cur, epoch, model, loader, n_classes, early_stopping = None, loss_fn = None, results_dir=None, exp_name= None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)

    val_loss = 0.
    val_error = 0.

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
            data, label = data.to(torch.float32), label.to(torch.long) 

            logits, Y_prob, Y_hat, _, _ = model(data)
                
            acc_logger.log(Y_hat, label) 
            loss = loss_fn(logits, label)
            
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)
    
    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
      
    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, exp_name, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False



def summary(model, loader, n_classes, loss_fn, bag_size, args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    all_atts = np.zeros((len(loader), bag_size))
    
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        data, label = data.to(torch.float32), label.to(torch.long) 

        with torch.no_grad():
            logits, Y_prob, Y_hat, _, A = model(data)

        loss = loss_fn(logits, label)
        test_loss += loss.item()
                        
        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        if args.model_type == 'abmil' or args.model_type == 'abmil_multihead' or args.model_type == 'abmil_multihead_sep':
            all_atts[batch_idx] = A.cpu().numpy()
        
        error = calculate_error(Y_hat, label)
        test_error += error
        
    test_loss /= len(loader)
    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    return test_error, test_loss, auc, acc_logger, all_atts, all_probs



def plot(data, bag_class, plot_size=3, args= None, fold_num= None, preds= False):

    """"Utility for plotting bags and attention weights.

    Args:
      data: Input data that contains the bags of instances.
      labels: The associated bag labels of the input data.
      bag_class: String name of the desired bag class.
        The options are: "positive" or "negative".
    """
    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
      
    if args.model_type == 'abmil':
        model = ABMIL(**model_dict)  
    if args.model_type == 'abmil_multihead':
        model = ABMIL_Multihead(**model_dict, temp= args.temp)                  
    elif args.model_type == 'max_pool':
        model = Max_Pool(**model_dict)
    elif args.model_type == 'mean_pool':
        model = Mean_Pool(**model_dict)    
          
    model.relocate()
    model.load_state_dict(torch.load(os.path.join(args.results_dir, args.exp_name, "s_{}_checkpoint.pt".format(fold_num))))
    test_loader = DataLoader(data, batch_size= 1, shuffle= False)
    loss_fn = nn.CrossEntropyLoss()
 
    _, _, _, _, attention_weights, predictions = summary(model, test_loader, args.n_classes, loss_fn, args.BAG_SIZE, args= args)

    
    labels= data.tensors[1]
    labels = np.array(labels).reshape(-1)

    data= data.tensors[0]
    if bag_class == "positive":
        if preds:
            labels = np.where(predictions.argmax(1) == 1)[0]
            bags = np.array(data)[labels[0:plot_size], :]

        else:
            labels = np.where(labels == 1)[0]
            bags = np.array(data)[labels[0:plot_size], :]

    elif bag_class == "negative":
        if preds:
            labels = np.where(predictions.argmax(1) == 0)[0]
            bags = np.array(data)[labels[0:plot_size], :]
        else:
            labels = np.where(labels == 0)[0]
            bags = np.array(data)[labels[0:plot_size], :]

    else:
        print(f"There is no class {bag_class}")
        return
    
    leg= 'com'
    idxes= np.argsort(attention_weights)[:, -8:]
    print(f"The bag class label is {bag_class}")
    for i in range(plot_size):
        figure = plt.figure(figsize=(32, 32))
        print(f"Bag number: {labels[i]}")
        for j in range(args.BAG_SIZE):
            image = bags[i][j]
            idx_ = idxes[labels[i]]
            figure.add_subplot(1, args.BAG_SIZE, j + 1)
            plt.grid(False)
            if attention_weights is not None:
                plt.title(np.around(attention_weights[labels[i]][j], 4))
            plt.imshow(image)
            if j in idx_:    
                plt.gca().add_patch(Rectangle((0,0,28,28),28,28,linewidth=10 ,edgecolor='r',facecolor='none'))
            figure.savefig('{}.png'.format(os.path.join(args.results_dir, args.exp_name, f'{bag_class, labels[i], np.around(predictions[labels[i]][1], 2), leg}')), bbox_inches= 'tight', dpi= 300)  
        plt.show()        