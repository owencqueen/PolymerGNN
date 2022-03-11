import torch

def epistemic(outdict):
    '''
    Epistemic uncertainty given the output of evidential model
    '''
    beta = outdict['beta']
    alpha = outdict['alpha']
    v = outdict['v']

    return (beta / (v * (alpha - 1))).item()

def aleatoric(outdict):
    '''
    Aleatoric uncertainty given the output of evidential model
    '''
    beta = outdict['beta']
    alpha = outdict['alpha']

    return (beta / (alpha - 1)).item()
