import torch
from torch.autograd import grad

def D_lin_efficient(losses, w):
    #shuffle the losses
    permutation = torch.randperm(losses.shape[0])
    losses = losses[permutation]
    
    #create two mini-batches of size losses.shape[0]/2
    #list[a::n] returns a list containing elements indexed by a+k*n from the original list.
    batch_1 = losses[0::2]
    batch_2 = losses[1::2]
    
    grad_1 = grad(batch_1.mean(), w, create_graph=True)[0] 
    grad_2 = grad(batch_2.mean(), w, create_graph=True)[0] 
    
    #As the result grad_1 * grad_2 is a tensor of shape [1], 
    #we use .sum() to "transform" it into scalar 
    return (grad_1 * grad_2).sum() 