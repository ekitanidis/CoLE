import argparse


def load_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', type=str, default='CL')    
    parser.add_argument('--scheduler', type=bool, default=True)    
    
    ### Directories and filenames
    parser.add_argument('--glue_dir', type=str, default='../../SuperGlueSuite/data/')
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--dataset_name', type=str, default='ThePile')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--checkpoint_name', type=str, default='pretrain-cl.pt')
    
    ### Data and augmentation
    parser.add_argument('--hf_tokenizer', type=str, default=None)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--num_sen', type=int, default=3)
    parser.add_argument('--min_sep', type=int, default=1)
    parser.add_argument('--max_sep', type=int, default=None)
    parser.add_argument('--mlm_prob', type=int, default=0.15)
    
    ### Contrastive pre-training
    parser.add_argument('--resume_from_checkpoint', action='store_true')
    # Backbone (Transformer encoder) architecture
    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_encoders', type=int, default=6)
    # Projector architecture
    parser.add_argument('--proj_hidden', type=int, default=2048)
    parser.add_argument('--proj_out', type=int, default=2048)
    # Predictor architecture
    parser.add_argument('--pred_hidden', type=int, default=512)
    parser.add_argument('--pred_out', type=int, default=2048)
    # Learning params
    parser.add_argument('--pretrain_batch_size', type=int, default=128)
    parser.add_argument('--pretrain_warmup_iters', type=int, default=0)
    parser.add_argument('--pretrain_warmup_lr', type=float, default=0)
    parser.add_argument('--pretrain_total_iters', type=int, default=3000) #100000
    parser.add_argument('--pretrain_base_lr', type=float, default=1e-4)
    parser.add_argument('--pretrain_final_lr', type=float, default=0.0)
    parser.add_argument('--pretrain_weight_decay', type=float, default=0.01)
    
    ### Evaluation on RTE & MNLI tasks
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--eval_lr', type=float, default=3e-5)
    parser.add_argument('--eval_weight_decay', type=float, default=0)
    parser.add_argument('--eval_epochs', type=int, default=30)

    args = parser.parse_args()
        
    return args
