from .losses import SimSiamLoss, BCELoss

def get_criterion(mode):
    
    if mode == 'CL':
        criterion = SimSiamLoss()
    elif mode == 'MLM':
        criterion = BCELoss()
    else:
        raise NotImplementedError
    
    return criterion
