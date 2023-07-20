from string import punctuation
import thulac
import jieba
import numpy as np


def format_string(s):
    return s.replace("b", "").replace("\t", " ").replace("t", "")


def punc_delete(fact_list):
    add_punc = '，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥'
    all_punc = punctuation + add_punc

    fact_filtered = []
    for word in fact_list:
        if word not in all_punc:
            fact_filtered.append(word)
    return fact_filtered


def hanzi_to_num(hanzi):
    _hanzi = hanzi.strip().replace('零', '')
    if _hanzi == '':
        return str(int(0))
    d = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '': 0}
    m = {'十': 1e1, '百': 1e2, '千': 1e3, }
    w = {'万': 1e4, '亿': 1e8}
    res = 0
    tmp = 0
    thou = 0
    for i in _hanzi:
        if i not in d.keys() and i not in m.keys() and i not in w.keys():
            return _hanzi

    if (_hanzi[0]) == '十':
        _hanzi = '一' + _hanzi

    for i in range(len(_hanzi)):
        if _hanzi[i] in d:
            tmp += d[_hanzi[i]]
        elif _hanzi[i] in m:
            tmp *= m[_hanzi[i]]
            res += tmp
            tmp = 0
        else:
            thou += (res + tmp) * w[_hanzi[i]]
            tmp = 0
            res = 0
    return int(thou + res + tmp)


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


def get_cutter(dict_path, stopword_path, mode='thulac', stop_words_filtered=True):
    if stop_words_filtered:
        stopwords = stopwordslist(stopword_path)
    else:
        stopwords = []
    if mode == 'jieba':
        jieba.load_userdict(dict_path)
        return lambda x: [a for a in list(jieba.cut(x)) if a not in stopwords]
    elif mode == 'thulac':
        thu = thulac.thulac(user_dict=dict_path, seg_only=True)
        return lambda x: [a for a in thu.cut(x, text=True).split(' ') if a not in stopwords]


def seg_sentence(sentence, cut):
    sentence_seged = cut(sentence)
    outstr = []
    for word in sentence_seged:
        if word != '\t':
            word = str(hanzi_to_num(word))
            outstr.append(word)
    return outstr


def lookup_index_for_sentences(sentences, word2id, doc_len, sent_len):
    item_num = 0
    res = []
    if len(sentences) == 0:
        tmp = [word2id['BLANK']] * sent_len
        res.append(np.array(tmp))
    else:
        for sent in sentences:
            sent = punc_delete(sent)
            tmp = [word2id['BLANK']] * sent_len
            for i in range(len(sent)):
                if i >= sent_len:
                    break
                try:
                    tmp[i] = word2id[sent[i]]
                    item_num += 1
                except KeyError:
                    tmp[i] = word2id['UNK']

            res.append(np.array(tmp))
    if len(res) < doc_len:
        res = np.concatenate(
            [np.array(res), word2id['BLANK'] * np.ones([doc_len - len(res), sent_len], dtype=np.int)], 0)
    else:
        res = np.array(res[:doc_len])

    return res, item_num


def sentence2index_matrix(sentence, word2id, doc_len, sent_len, cut):
    sentence = sentence.replace(' ', '')
    sent_words, sent_n_words = [], []
    for i in sentence.split('。'):
        if i != '':
            sent_words.append((seg_sentence(i, cut)))
    index_matrix, item_num = lookup_index_for_sentences(sent_words, word2id, doc_len, sent_len)
    return index_matrix, item_num, sent_words
