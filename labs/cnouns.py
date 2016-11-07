# -*- coding: utf-8 -*-  

from konlpy.tag import Mecab
import hanja
import re

mecab = Mecab()

def test_special_ch(typ):
    return typ == u'SF' or typ == u'SE' or typ == u'SSO' or typ == u'SSC' or typ == u'SC' or typ == u'SY'

def test_noun(typ):
    return typ == u'NNP' or typ == u'NNG'

def test_compound_noun(pos_tags):
    return sum([test_noun(p) for p in pos_tags]) == len(pos_tags)
def test_pisusic_noun(pos_tags):
    return len(pos_tags) == 3 and (test_noun(pos_tags[0]) and pos_tags[1] == u'JKG' and test_noun(pos_tags[2]))
def test_jugeck_noun(pos_tags):
    return len(pos_tags) == 3 and (test_noun(pos_tags[0]) and pos_tags[1] == u'JKS' and test_noun(pos_tags[2]))
def test_mokjeck_noun(pos_tags):
    return len(pos_tags) == 3 and (test_noun(pos_tags[0]) and pos_tags[1] == u'JKO' and test_noun(pos_tags[2]))
    
def extract_compound_nouns(inp_pos, n):
    ngrams = zip(*[inp_pos[i:] for i in range(n)])
    results = []
    for ngram in ngrams:
        pos_tags = [e[1] for e in ngram]
        
        if (
            test_compound_noun(pos_tags) or 
            test_pisusic_noun(pos_tags) or 
            test_jugeck_noun(pos_tags) or 
            test_mokjeck_noun(pos_tags)
           ): 
            results.append(''.join(e[0] + e[1] for e in ngram))

    return results
    
def extract_all_compound_nouns(inp_pos, max_n):
    results = []
    for n in range(1, max_n + 1):
        results = results + extract_compound_nouns(inp_pos, n)
    return results

def morphs_ngrams(inp_pos, max_n):
    results = []
    for n in range(1, max_n + 1):
        ngrams = zip(*[inp_pos[i:] for i in range(n)])
        results = results + [''.join(e[0] + e[1] for e in ngram) for ngram in ngrams]
    return results

def morphs_ngrams_without_tag(inp_pos, max_n):
    results = []
    for n in range(1, max_n + 1):
        ngrams = zip(*[inp_pos[i:] for i in range(n)])
        for ngram in ngrams:
            if(all([not test_special_ch(e[1]) for e in ngram])):
                results = results + [''.join(e[0] for e in ngram)]
    return results

def text_cleaning(text):
    text = hanja.translate(text, 'substitution')
    text = re.sub(u'(\[.*\]|\(.*\))', '', text)
    text = re.sub(u'(\(|\)|\[|\])', '', text)
    return text

def text_cleaning_without_special_ch(text):
    text = hanja.translate(text, 'substitution')
    text = re.sub(u'(\[.*\]|\(.*\))', '', text)
    text = re.sub(u'(\(|\)|\[|\])', '', text)
    text = re.sub(u'[^가-힝0-9a-zA-Z\\s]', '', text)
    return text

def tokenize(inp_str):
    return ' '.join(tokenize_arr(inp_str))

def tokenize_arr(inp_str):
    clean_text = text_cleaning(inp_str)
    pos_tags = mecab.pos(clean_text)
    cns = extract_all_compound_nouns(pos_tags, 3)
    morphs = morphs_ngrams(pos_tags, 3)
    return cns + morphs

def tokenize_nouns(inp_str):
    return ' '.join(tokenize_nouns_arr(inp_str))

def tokenize_nouns_arr(inp_str):
    clean_text = text_cleaning(inp_str)
    pos_tags = mecab.pos(clean_text)
    cns = extract_all_compound_nouns(pos_tags, 3)
    return cns

def pos_tags(inp_str):
    return ' '.join(pos_tags_arr(inp_str))

def pos_tags_arr(inp_str):
    clean_text = text_cleaning(inp_str)
    pos_tags = mecab.pos(clean_text)
    return [pt[0] + pt[1] for pt in pos_tags]