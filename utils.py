import os
import torch
import torch.nn.functional as F

from loss import get_criterion


class AverageMeter():

    def __init__(self, param_name, time_unit='epoch', delta_time=1):
        self.param_name = param_name
        self.time_unit = time_unit
        self.log = dict({time_unit:[], param_name:[]})
        self.delta = delta_time
        self.time = 0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.log[self.time_unit].append(self.time)
        self.log[self.param_name].append(self.avg)
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.time += self.delta

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count        
                
        
def accuracy(model, dataloader, device):

    model.eval()
    num_correct = 0
    
    with torch.no_grad():

        for batch in dataloader:
    
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            output = model(input_ids).squeeze(dim=1)
            pred = (output > 0.5).float()
            num_correct += (pred == labels).sum().item()
            batch_size = input_ids.size(0)

    accuracy = 100 * num_correct / (len(dataloader) * batch_size)

    return accuracy


def get_val_loss(model, dataloader, device, args):
    
    model.eval()
    criterion = get_criterion(mode=args.mode)
    running_loss = 0.0
    num_batches = 0

    with torch.no_grad():
    
        for batch in dataloader:
    
            if args.mode == 'CL':
                data_1, data_2 = batch
                input_ids_1, attn_mask_1 = data_1
                input_ids_2, attn_mask_2 = data_2
                input_ids_1, attn_mask_1 = input_ids_1.to(device, non_blocking=True), attn_mask_1.to(device, non_blocking=True)
                input_ids_2, attn_mask_2 = input_ids_2.to(device, non_blocking=True), attn_mask_2.to(device, non_blocking=True)
                z1, p1 = model.forward(input_ids_1, src_mask=attn_mask_1)
                z2, p2 = model.forward(input_ids_2, src_mask=attn_mask_2)
                loss = criterion(z1, z2, p1, p2)
            
            elif args.mode == 'MLM':
                data, = batch
                input_ids, attn_mask, mlm_mask = data
                input_ids, attn_mask, mlm_mask = input_ids.to(device, non_blocking=True), attn_mask.to(device, non_blocking=True), mlm_mask.to(device, non_blocking=True)
                pred = model.forward(input_ids, src_mask=attn_mask)            
                loss = criterion(pred, mlm_mask)
                
            running_loss += loss.item()
            num_batches += 1
        
        avg_loss = running_loss / num_batches
    
    return avg_loss


def load_from_saved(model, optimizer, scheduler, loss_meter, val_loss_meter, rte_acc_meter, args):

    saved = torch.load(os.path.join(args.checkpoint_dir, args.checkpoint_name))

    model.load_state_dict(saved['model_state_dict'])
    optimizer.load_state_dict(saved['optimizer_state_dict'])
    scheduler.load_state_dict(saved['scheduler_state_dict'])
    loss_meter = saved['loss_meter']
    val_loss_meter = saved['val_loss_meter']
    rte_acc_meter = saved['rte_acc_meter']
    current_iter = saved['current_iter']
    current_epoch = saved['current_epoch']
    original_args = saved['args']

    print('Overriding config args with settings used to generate saved model, for consistency.')
    for key, value in vars(original_args).items():
        if key not in ['checkpoint_dir', 'checkpoint_name']:
            setattr(args, key, value)

    assert (args.pretrain_total_iters - current_iter) >= 0, "This training run is already complete."

    return model, optimizer, scheduler, current_iter, current_epoch, loss_meter, val_loss_meter, rte_acc_meter, args
