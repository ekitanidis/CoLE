from .backbone import FancyEncoder
from .simsiam import Encoder, Predictor, SimSiam
from .mlm import MaskEncoder
from torch import nn


def get_model(vocab_size, config):
    
    backbone = FancyEncoder(vocab_size, d_model=config.embed_size, **vars(config))        
    if config.mode == 'CL':
        encoder = Encoder(backbone, input_dim=config.embed_size, hidden_dim=config.proj_hidden, output_dim=config.proj_out)
        predictor = Predictor(input_dim=config.proj_out, hidden_dim=config.pred_hidden, output_dim=config.pred_out)
        model = SimSiam(encoder, predictor)
    elif config.mode == 'MLM':
        model = MaskEncoder(backbone)
    else:
        raise NotImplementedError
    
    return model
