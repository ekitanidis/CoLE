from torch import optim
from .schedulers import CosineScheduler, LinearScheduler


def get_optim(model, args):
    
    opt_args = {
        'lr': args.pretrain_base_lr,
        'weight_decay': args.pretrain_weight_decay
    }
    
    sched_args = {
        'warmup_iters': args.pretrain_warmup_iters,
        'warmup_lr': args.pretrain_warmup_lr,
        'total_iters': args.pretrain_total_iters,
        'base_lr': args.pretrain_base_lr,
        'final_lr': args.pretrain_final_lr,
        'constant_predictor_lr': True
    }
        
    if args.mode == 'CL':
        parameters = [{'name': 'encoder',
                       'params': [param for name, param in model.named_parameters() if name.startswith('encoder')],
                       'lr': args.pretrain_base_lr},
                      {'name': 'predictor',
                       'params': [param for name, param in model.named_parameters() if name.startswith('predictor')],
                       'lr': args.pretrain_base_lr}]
        optimizer = optim.Adam(parameters, **opt_args)
        if args.scheduler:
            scheduler = CosineScheduler(optimizer, **sched_args)
        else:
            scheduler = None
        
    elif args.mode == 'MLM':
        optimizer = optim.Adam(model.parameters(), **opt_args)
        if args.scheduler:
            scheduler = LinearScheduler(optimizer, **sched_args)   
        else:
            scheduler = None
            
    else:
        raise NotImplementedError
    
    return optimizer, scheduler
