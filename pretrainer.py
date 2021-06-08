from config import load_args

from data.process import get_tokenizer
from data.datasets import fetch_datasets, DataLoader

from model import get_model
from optim import get_optim
from loss import get_criterion

from utils import AverageMeter, load_from_saved, get_val_loss
from evals.glue import run_glue_eval

import time
import os

import torch
from torch import nn, optim

import wandb


def unsupervised_train(device, args):
    
    config_defaults = vars(args)
    wandb.init(config=config_defaults)    
    config = wandb.config
    for key, value in config.__dict__.items():
        vars(args)[key] = value
    
    if not os.path.isdir(args.checkpoint_dir):
        print('Checkpoint directory %s not found. Creating local directory.' % args.checkpoint_dir) 
        os.mkdir(args.checkpoint_dir)
            
    tokenizer = get_tokenizer(**vars(args))
    vocab_size = tokenizer.vocab_size
    
    train_set, val_set = fetch_datasets(samples=('train', 'val'), **vars(args))
        
    model = get_model(vocab_size, args)
    model.to(device)
    
    optimizer, scheduler = get_optim(model, args)

    criterion = get_criterion(mode=args.mode)
        
    loss_meter = AverageMeter('loss', time_unit='iters', delta_time=10)
    train_loss_meter = AverageMeter('loss', time_unit='epoch', delta_time=1)
    val_loss_meter = AverageMeter('loss', time_unit='epochs', delta_time=1)
    rte_acc_meter = AverageMeter('accuracy', time_unit='epochs', delta_time=1)

    epochs, iters = 0, 0
    iterate_through_dataset = True
    
    if args.resume_from_checkpoint:
        model, optimizer, scheduler, current_iter, current_epoch, loss_meter, val_loss_meter, rte_acc_meter, args = load_from_saved(model, optimizer, scheduler, loss_meter, vall_loss_meter, rte_acc_meter, args)
        iters = current_iter
        epochs = current_epoch

    while iterate_through_dataset:
                
        print('=== Epoch %s ===' % (epochs + 1))
        
        # Each epoch, we sample new sentences from the same set of documents
        train_loader = DataLoader(train_set, tokenizer, batch_size=args.pretrain_batch_size, shuffle=True, drop_last=True, **vars(args))
        val_loader = DataLoader(val_set, tokenizer, batch_size=args.pretrain_batch_size, shuffle=False, drop_last=True, **vars(args))
        
        for batch_id, batch in enumerate(train_loader):
            
            batch_start = time.time()
            
            model.train()
            
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

            loss.backward()
            wandb.log({'train_loss_per_batch': loss.item()})
            loss_meter.update(loss.item())
            train_loss_meter.update(loss.item())
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            if args.scheduler:
                scheduler.step()
            
            model.zero_grad()

            iters += 1

            if (iters % loss_meter.delta == 0) and (iters > 0):
                batch_end = time.time()
                print('Epoch %s | iters %s-%s took %.2f seconds and had mean loss of %.4f.' % (epochs + 1, iters - loss_meter.delta, iters, batch_end - batch_start, loss_meter.avg))
                loss_meter.reset()

#             if (iters % 500 == 0) or (iters == args.pretrain_total_iters):
#                   torch.save({
#                   'model_state_dict': model.state_dict(),
#                   'optimizer_state_dict': optimizer.state_dict(),
#                   'scheduler_state_dict': scheduler.state_dict(),
#                   'current_iter': iters,
#                   'current_epoch': epochs,
#                   'loss_meter': loss_meter,
#                   'val_loss_meter': val_loss_meter,
#                   'rte_acc_meter': rte_acc_meter,
#                   'args': args,
#                   }, os.path.join(args.checkpoint_dir, args.checkpoint_name))
                
            if iters == args.pretrain_total_iters:
                iterate_through_dataset = False
                break                

        # Get validation loss over epoch
        val_loss = get_val_loss(model, val_loader, device, args)
        wandb.log({'train_loss_per_epoch': train_loss_meter.avg})
        wandb.log({'val_loss_per_epoch': val_loss})
        print('Epoch %s has mean training loss of %.4f and mean validation loss of %.4f.' % (epochs + 1, train_loss_meter.avg, val_loss))
        val_loss_meter.update(val_loss)
        val_loss_meter.reset()
        train_loss_meter.reset()
        
        # Get RTE accuracy (train & val) at this epoch
        print('Running RTE eval suite now...')
        max_val_acc = run_glue_eval('RTE', model, tokenizer, device, args)
        wandb.log({'accuracy_per_epoch': max_val_acc})
        rte_acc_meter.update(max_val_acc)
        rte_acc_meter.reset()
        
        epochs += 1
                            
                
if __name__ == '__main__':

    args = load_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    unsupervised_train(device, args)
