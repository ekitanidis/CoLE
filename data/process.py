import re
import torch
from transformers import AutoTokenizer


def get_tokenizer(hf_tokenizer, **args):
    
    if hf_tokenizer is None:
        hf_tokenizer = 'bert-base-cased'
    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer)
    
    return tokenizer


def sentence_chunk(text_block):
    
    # Deal with non-fullstop periods (e.g. acronyms, decimals, ellipses)  --- TO DO: Deal with web urls!
    
    digits = "([0-9])"
    alphabets= "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
        
    text_block = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text_block)
    text_block = re.sub(prefixes,"\\1<prd>",text_block)
    if "Ph.D" in text_block: text_block = text_block.replace("Ph.D.","Ph<prd>D<prd>")
    if "M.D" in text_block: text_block = text_block.replace("M.D.","M<prd>D<prd>")
    if "J.D" in text_block: text_block = text_block.replace("J.D.","J<prd>D<prd>") 
    if "e.g." in text_block: text_block = text_block.replace("e.g.","e<prd>g<prd>") 
    if "i.e." in text_block: text_block = text_block.replace("i.e.","i<prd>e<prd>")
    text_block = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text_block)
    text_block = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text_block)
    text_block = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text_block)
    text_block = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text_block)
    text_block = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text_block)
    text_block = re.sub(" "+suffixes+"[.]"," \\1<prd>",text_block)
    text_block = re.sub(" " + alphabets + "[.]"," \\1<prd>",text_block)
    if "..." in text_block: text_block = text_block.replace("...","<prd><prd><prd>")

    # Headers and titles should be treated as full sentences
    
    text_block = text_block.replace("\n\n","<stop>")    

    # Move fullstop punctuation outside quotations and parantheses
    
    for ender in (".", "!", "?"):
        if ")" in text_block: text_block = text_block.replace(ender + ")", ")" + ender)
        if "\"" in text_block: text_block = text_block.replace(ender + "\"", "\"" + ender)
        if "”" in text_block: text_block = text_block.replace(ender + "”", "”" + ender)
        
    # Chunk into sentences and remove empty strings
    
    text_block = text_block.replace(".",".<stop>")
    text_block = text_block.replace("?","?<stop>")
    text_block = text_block.replace("!","!<stop>")
    text_block = text_block.replace("<prd>",".")    
    sentences = text_block.split("<stop>")
    sentences = [s.strip() for s in sentences]
    sentences = list(filter(None, sentences))     
    
    return sentences
