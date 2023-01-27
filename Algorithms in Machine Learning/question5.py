import torch
from torch.autograd import grad

def D_lin(losses, dummy_w):
    #shuffle the losses
    permutation = torch.randperm(losses.shape[0])
    losses = losses[permutation]
    
    #create two mini-batches of size losses.shape[0]/2
    #list[a::n] returns a list containing elements indexed by a+k*n from the original list.
    batch_1 = losses[0::2]
    batch_2 = losses[1::2]
    D = 0
    for b1,b2 in tqdm(zip(batch_1, batch_2)):
        grad_1 = grad(b1, dummy_w, create_graph=True, retain_graph=True)[0]
        grad_2 = grad(b2, dummy_w, create_graph=True, retain_graph=True)[0] 
        D += grad_1 * grad_2
    return D