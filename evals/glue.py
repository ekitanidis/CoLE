import os
import random
from copy import deepcopy

from model.finetune import SumLayer, FinetuneModel
from utils import accuracy

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from nlp import load_dataset
import warnings
warnings.filterwarnings('ignore') # To suppress this annoying warning: https://github.com/huggingface/datasets/issues/616


task_dict = dict()

task_dict['RTE'] = {'label_dict': {'not_entailment': 0, 'entailment': 1},
                    'variable_dict': {'premise': 'premise', 'hypothesis': 'hypothesis'},
                    'label_name': 'label',
                    'num_classes': 2,
                    'fn': {'train': 'RTE/train.jsonl', 'val': 'RTE/val.jsonl'}}

task_dict['MNLI'] = {'label_dict': {'contradiction': 0, 'neutral': 1, 'entailment': 2},
                    'variable_dict': {'premise': 'sentence1', 'hypothesis': 'sentence2'},
                    'label_name': 'gold_label',
                    'num_classes': 3,
                    'fn': {'train': 'MNLI/multinli_1.0_train.jsonl', 'val': 'MNLI/multinli_1.0_dev_matched.jsonl'}}


def get_glue_data(task, tokenizer, args):
    
    train_fn = os.path.join(args.glue_dir, task_dict[task]['fn']['train'])
    val_fn = os.path.join(args.glue_dir, task_dict[task]['fn']['val'])

    train_dataset = load_dataset('json', data_files=train_fn)['train']
    val_dataset = load_dataset('json', data_files=val_fn)['train']
    
    def label_encoder(ds):
        label_dict = task_dict[task]['label_dict']
        label_name = task_dict[task]['label_name']
        ds['label'] = label_dict.get(ds[label_name], None)
        return ds

    def smart_encoder(tokenizer, max_length):
        ''' Tokenize and concatenate, with hypothesis always before premise:
            - If total length less than (or equal to) max sequence length, pad with zeros as needed. 
            - Else, concatenate full hypothesis with random consecutive subset of premise. No padding.
            Hypothesis begins with [CLS]. Hypothesis and premise separated with [SEP] token.
            Example 1: "[CLS] This is a hypothesis. [SEP] This is a premise. [SEP] [PAD] [PAD] [PAD]"
            Example 2: "[CLS] This is a hypothesis. [SEP] is a random subset of a premise, and" '''
        def encoder_func(ds):
            p_name, h_name = task_dict[task]['variable_dict']['premise'], task_dict[task]['variable_dict']['hypothesis']
            h, p = ds[h_name], ds[p_name] 
            h_dict = tokenizer(h, max_length=max_length, truncation=True)
            h_ids, h_attn = h_dict['input_ids'], h_dict['attention_mask']
            p_max = max_length - len(h_ids)
            p_dict = tokenizer(p, padding='max_length', max_length=p_max+1, truncation=False)
            p_ids, p_attn = p_dict['input_ids'][1:], p_dict['attention_mask'][1:] # remove extraneous [CLS] token from start of premise
            p_len = len(p_ids)
            if p_len > p_max:
                idx = random.randint(0,p_len-p_max)
                p_ids = p_ids[idx:idx+p_max]
                p_attn = p_attn[idx:idx+p_max]
            h_ids.extend(p_ids)
            h_attn.extend(p_attn)
            return dict({'input_ids': h_ids, 'attention_mask': h_attn})
        return encoder_func

    def remove_unknown_labels(ds):
        if ds['label'] is not None:
            return True
        return False

    train_dataset = train_dataset.map(label_encoder)
    val_dataset = val_dataset.map(label_encoder)

    train_dataset = train_dataset.map(smart_encoder(tokenizer, args.max_seq_len))
    val_dataset = val_dataset.map(smart_encoder(tokenizer, args.max_seq_len))

    train_dataset = train_dataset.filter(remove_unknown_labels)
    val_dataset = val_dataset.filter(remove_unknown_labels)

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    return train_dataset, val_dataset
        
    
def run_glue_eval(task, model, tokenizer, device, args):
    
    train_dataset, val_dataset = get_glue_data(task, tokenizer, args)
    
    train_loader = DataLoader(train_dataset, batch_size=args.eval_batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True, drop_last=True, num_workers=0)
    
    encoder = deepcopy(model.encoder.backbone)
    model = FinetuneModel(encoder, SumLayer())
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.eval_lr)
    
    criterion = nn.BCELoss()
    
    max_val_acc = 0

    for epoch in range(args.eval_epochs):

        model.train()
        running_loss = 0

        for batch_id, batch in enumerate(train_loader):

            model.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            pred = model(input_ids, src_mask=attn_mask).squeeze(dim=1)

            loss = criterion(pred, labels.float())
            running_loss += loss.item()
            loss.backward()

            optimizer.step()

        avg_loss = running_loss / len(train_loader)
        train_acc = accuracy(model, train_loader, device)
        val_acc = accuracy(model, val_loader, device)
        if val_acc > max_val_acc:
            max_val_acc = val_acc

        print('Evaluating on %s | eval epoch %s | average loss: %.4f, train acc: %.2f, val acc: %2f' % (task, epoch, avg_loss, train_acc, val_acc))
        
    return max_val_acc
