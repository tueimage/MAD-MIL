
from calflops import calculate_flops
from model_abmil import ABMIL, ABMIL_Multihead_Sep
from model_pool import Max_Pool, Mean_Pool
from model_dsmil import *
from model_clam import CLAM_MB, CLAM_SB

model_type= 'abmil_multihead_sep'
temp= [1,1]
drop_out= True
n_classes= 2
subtyping= True
model_size= "small"
head_size= "small"
B= 8


model_dict = {"dropout": drop_out, 'n_classes': n_classes}
if model_type == 'clam' and subtyping:
    model_dict.update({'subtyping': True})

if model_size is not None and model_type != 'mil':
    model_dict.update({"size_arg": model_size})

if model_type in ['clam_sb', 'clam_mb']:
    if subtyping:
        model_dict.update({'subtyping': True})
    
    if B > 0:
        model_dict.update({'k_sample': B})
        instance_loss_fn = nn.CrossEntropyLoss()
        
        from model_clam import CLAM_MB, CLAM_SB

    if model_type =='clam_sb':
        model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
    elif model_type == 'clam_mb':
        model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
    else:
        raise NotImplementedError   

elif model_type == 'abmil':
    model = ABMIL(**model_dict) 
elif model_type == 'abmil_multihead_sep':
    model = ABMIL_Multihead_Sep(**model_dict, temp= temp, head_size= head_size)               
elif model_type == 'max_pool':
    model = Max_Pool(**model_dict)        
elif model_type == 'mean_pool':
    model = Mean_Pool(**model_dict) 
elif model_type == 'dsmil':
    i_classifier = FCLayer(in_size= 1024, out_size= 2)
    b_classifier = BClassifier(input_size= 1024, output_class= 2, dropout_v=0.0)
    model = MILNet(i_classifier, b_classifier) 

input_shape = (120, 1024)
flops, macs, params = calculate_flops(model=model, 
                                      input_shape=input_shape,
                                      output_as_string=True,
                                      output_precision=4)
print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
