import torch
import random
from .process import sentence_chunk, get_tokenizer
 
    
class Dataset():
    
    def __init__(self, dataset):            
        self.dataset = dataset
        
    def __iter__(self):
        for x in self.dataset:
            yield x
            
    def shuffle(self, buffer_size=None):
        return RandomShuffler(self, buffer_size)
            
    def sentence_chunk(self, num_sen=1, min_sep=1):
        return SentenceChunker(self, num_sen, min_sep)

    def pair_choose(self, num_sen=1, min_sep=1, max_sep=None):
        return PairChooser(self, num_sen, min_sep, max_sep)

    def single_choose(self, num_sen=1, min_sep=1, max_sep=None):
        return SingleChooser(self, num_sen)

    def tokenize(self, tokenizer=None, max_length=50):
        return Tokenizer(self, tokenizer, max_length)

    def random_mask(self, mlm_prob=0.15):
        return RandomMasker(self, mlm_prob)
            
    def batch(self, batch_size, drop_last=True):
        return BatchSampler(self, batch_size, drop_last)

            
class RandomShuffler(Dataset):
    
    def __init__(self, dataset, buffer_size=None):
        super().__init__(dataset)
        self.buffer_size = buffer_size

    def __iter__(self):
        chunk = []
        if self.buffer_size is None:
            for x in self.dataset:
                chunk.append(x)
            random.shuffle(chunk)
            yield from chunk
        else:
            for x in self.dataset:
                chunk.append(x)
                if len(chunk) == self.buffer_size:
                    random.shuffle(chunk)
                    yield from chunk
                    chunk = []
            if chunk:
                random.shuffle(chunk)
                yield from chunk

            
class SentenceChunker(Dataset):
    ''' Input is string containing full document.
        Output is list of strings corresponding to sentences in document.
    '''
    
    def __init__(self, dataset, num_sen=1, min_sep=1):
        super().__init__(dataset)
        self.num_sen = num_sen
        self.min_sep = min_sep

    def __iter__(self):
        for x in self.dataset:
            sentences = sentence_chunk(x)
            if len(sentences) >= (2 * self.num_sen + self.min_sep): # document must have enough sentences for at least one pair
                yield sentences
            else:
                pass      

            
class PairChooser(Dataset):
    ''' Used in CL mode only:
            Input is list of strings (output of SentenceChunker).
            Output is list of two strings.
    '''

    def __init__(self, dataset, num_sen=1, min_sep=1, max_sep=None):
        super().__init__(dataset)
        self.num_sen = num_sen
        self.min_sep = min_sep
        self.max_sep = max_sep

    def __iter__(self):
        for x in self.dataset:
            if self.max_sep is None:
                self.max_sep = len(x) - 2 * self.num_sen
            domain = list(range(len(x) - self.num_sen + 1))
            i = random.choice(domain)
            min_d = self.min_sep + self.num_sen
            max_d = self.max_sep + self.num_sen
            domain = [k for k in domain if (abs(i-k) >= min_d) and (abs(i-k) <= max_d)]
            if domain:
                j = random.choice(domain)
                t1, t2 = ' '.join(x[i:i+self.num_sen]), ' '.join(x[j:j+self.num_sen])
                yield [t1, t2]
            else:
                pass
            

class SingleChooser(Dataset):
    ''' Used in MLM mode only:
            Input is list of strings (output of SentenceChunker).
            Output is single string.
    '''
    
    def __init__(self, dataset, num_sen=1):
        super().__init__(dataset)
        self.num_sen = num_sen

    def __iter__(self):
        for x in self.dataset:
            domain = list(range(len(x) - self.num_sen + 1))
            if domain:
                i = random.choice(domain)
                t = ' '.join(x[i:i+self.num_sen])
                yield t
            else:
                pass


class Tokenizer(Dataset):
    ''' In CL mode:
            Input is list of two strings (output of PairChooser).
            Output is torch.Tensor of shape (2, 2, max_seq_len).
        In MLM mode:
            Input is string (output of SingleChooser).
            Output is torch.Tensor of shape (2, 1, max_seq_len).
    '''
    
    def __init__(self, dataset, tokenizer=None, max_length=50):
        super().__init__(dataset)
        if tokenizer is None:
            self.tokenizer = get_tokenizer()
        else: 
            self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.max_length = max_length

    def __iter__(self):
        for x in self.dataset:
            tokenized = self.tokenizer(x, padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")
            input_ids = tokenized['input_ids']
            attention_mask = tokenized['attention_mask']
            output = torch.stack((input_ids, attention_mask), dim=0)
            yield output
    
    
class RandomMasker(Dataset):
    ''' Used in MLM mode only:
            Input is torch.Tensor of shape (2, 1, max_seq_len) (output of Tokenizer).
            Output is torch.Tensor of shape (3, 1, max_seq_len).
    '''

    def __init__(self, dataset, mlm_prob=0.15):
        super().__init__(dataset)
        self.mlm_prob = mlm_prob
        self.vocab_size = dataset.vocab_size
        self.tokenizer = dataset.tokenizer

    def __iter__(self):
        
        for x in self.dataset:
        
            input_ids = x[0,:,:]
            attn_mask = x[1,:,:]

            special_tokens_mask = self.tokenizer.get_special_tokens_mask(input_ids.squeeze(), already_has_special_tokens=True)
            probs = torch.full(input_ids.size(), self.mlm_prob)
            probs.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool).unsqueeze(0), value=0.0)
            mlm_mask = torch.bernoulli(probs).bool()
            masked_ids = input_ids.clone()
            
            # 0% of the time, masked tokens replaced with the [MASK] token
            spec = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
            spec_mask = torch.bernoulli(torch.full(input_ids.size(), 0.0)).bool()
            masked_ids[spec_mask & mlm_mask] = spec
            
            # 100% of the time, masked tokens replaced with random tokens
            rans = torch.randint(0, self.vocab_size, input_ids.size(), dtype=torch.long)
            rans_mask = torch.bernoulli(torch.full(input_ids.size(), 1.0)).bool()
            masked_ids[rans_mask & mlm_mask & ~spec_mask] = rans[rans_mask & mlm_mask & ~spec_mask]
            
            out = torch.cat((masked_ids, attn_mask, mlm_mask), dim=0).to(torch.long).unsqueeze(1)
            yield out
        

class BatchSampler(Dataset):
    ''' In CL mode:
            Input is torch.Tensor of shape (2, 2, max_seq_len) (output of Tokenizer).
            Output is torch.Tensor of shape (batch_size, 2, 2, max_seq_len).
        In MLM mode:
            Input is torch.Tensor of shape (3, 1, max_seq_len) (output of RandomMasker).
            Output is torch.Tensor of shape (batch_size, 3, 1, max_seq_len).
    '''
    
    def __init__(self, dataset, batch_size, drop_last=True):
        super().__init__(dataset)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for x in self.dataset:
            batch.append(x)
            if len(batch) == self.batch_size:
                yield torch.stack(batch, dim=0)
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield torch.stack(batch, dim=0)

            
class DataLoader():
    ''' In CL mode:
            Returns tuple containing pair of sequences; each sequence is tuple of (input_ids, attention_mask).
        In MLM mode:
            Returns tuple containing single sequence, which is tuple of (input_ids, attention_mask, mlm_mask). 
    '''
    
    def __init__(self, dataset, tokenizer, mode, batch_size, shuffle=True, drop_last=True, num_sen=1, min_sep=1, max_sep=None, max_length=50, mlm_prob=0.15, **args):
        if shuffle:
            dataset = dataset.shuffle()
        dataset = dataset.sentence_chunk(num_sen, min_sep)
        if mode == 'CL':
            dataset = dataset.pair_choose(num_sen, min_sep, max_sep)
            dataset = dataset.tokenize(tokenizer, max_length)
            self.vocab_size = dataset.vocab_size
        elif mode == 'MLM':
            dataset = dataset.single_choose(num_sen)
            dataset = dataset.tokenize(tokenizer, max_length)
            self.vocab_size = dataset.vocab_size
            dataset = dataset.random_mask(mlm_prob)
            self.mlm_prob = mlm_prob
        else:
            raise NotImplementedError
        sampler = dataset.batch(batch_size, drop_last)
        self.dataset = dataset
        self.sampler = sampler
        self.mode = mode
        self.batch_size = batch_size
        self.mode = mode
        self.num_sen = num_sen
        self.min_sep = min_sep
        self.max_sep = max_sep
        self.max_length = max_length
                    
    def __iter__(self):
        for x in self.sampler:
            num_feats, num_seqs = x.size(1), x.size(2)
            output = tuple(tuple(x[:,idx,jdx,:] for idx in range(num_feats)) for jdx in range(num_seqs))
            yield output


def fetch_datasets(dataset_name, data_dir='.data', samples=('train', 'test', 'val'), **args):
    
    readers = dict()

    if dataset_name == 'ThePile':
        from .pile import PileReader
        for sample_select in samples:
            urls = {'train': [data_dir + 'train/{}.jsonl.zst'.format(str(i).zfill(2)) for i in range(30)],
                    'test': data_dir + 'test.jsonl.zst',
                    'val': data_dir + 'val.jsonl.zst'}
    #        readers[sample_select] = PileReader(urls[sample_select])
            readers['train'] = PileReader(data_dir + '00-mini.jsonl.zst')    
            readers['val'] = PileReader(data_dir + 'output.jsonl.zst')
    else:
        raise NotImplementedError
                    
    return tuple(Dataset(readers[sample_select]) for sample_select in samples)
