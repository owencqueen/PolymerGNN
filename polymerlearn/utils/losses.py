import torch
import numpy as np

def NIG_NLL(y, gamma, v, alpha, beta, reduce=True):
    '''
    Negative log loss for Normal-Inverse Gamma distribution

    For learning uncertainties with evidential regression
    '''

    O = 2 * beta * (1 + v)

    nll = 0.5 * torch.log(np.pi/v) \
        - alpha * torch.log(O) \
        + (alpha + 0.5) * (torch.log(v * (y - gamma)**2 + O)) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha + 0.5)

    return torch.mean(nll) if reduce else nll

def NIG_reg(y, gamma, v, alpha, reduce = True, *_):
    '''
    Computes regularization loss for evidential regression
    '''
    Phi = (2 * v + alpha)
    L = (torch.abs(y - gamma) * Phi)
    return torch.mean(L) if reduce else L

def evidential_loss(y_true, output_dict, coef = 1.0, reduce = True):
    '''
    Entire loss function for evidential regression
    '''

    gamma = output_dict['gamma']
    v = output_dict['v']
    alpha = output_dict['alpha']
    beta = output_dict['beta']

    loss_nll = NIG_NLL(y_true, gamma, v, alpha, beta, reduce)
    loss_reg = NIG_reg(y_true, gamma, v, alpha, reduce = reduce)

    return loss_nll + coef * loss_reg