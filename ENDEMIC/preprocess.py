from __future__ import unicode_literals
import collections
import io
import re
import six
import numpy as np
import progressbar
import json
import os
import pickle
from collections import namedtuple
import torch

from config import get_preprocess_args
from torch.nn.utils.rnn import pad_sequence

# from transformers import BertTokenizer

# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('distilbert-base-nli-mean-tokens')

split_pattern = re.compile(r'([.,!?"\':;)(])')
digit_pattern = re.compile(r'\d')

Special_Seq = namedtuple('Special_Seq', ['PAD', 'EOS', 'UNK', 'BOS'])
Vocab_Pad = Special_Seq(PAD=0, EOS=1, UNK=2, BOS=3)


def split_sentence(s, tok=False):
    if tok:
        s = s.lower()
        s = s.replace('\u2019', "'")
        # s = digit_pattern.sub('0', s)
    words = []
    for word in s.strip().split():
        if tok:
            words.extend(split_pattern.split(word))
        else:
            words.append(word)
    words = [w for w in words if w]
    return words


def open_file(path):
    return io.open(path, encoding='utf-8', errors='ignore')


def count_lines(path):
    with open_file(path) as f:
        return sum([1 for _ in f])


def read_file(path, tok=False):
    n_lines = count_lines(path)
    bar = progressbar.ProgressBar()
    with open_file(path) as f:
        for line in bar(f, max_value=n_lines):
            tokens = line.strip().split('\t')
            label = tokens[0]
            text = ' '.join(tokens[1:])
            words = split_sentence(text, tok)
            yield label, words


def count_words(path, max_vocab_size=40000, tok=False):
    counts = collections.Counter()
    for _, words in read_file(path, tok):
        for word in words:
            counts[word] += 1
    vocab = [word for (word, _) in counts.most_common(max_vocab_size)]
    return vocab


def get_label_vocab(path, tok=False):
    vocab_label = set()
    for label, _ in read_file(path, tok):
        vocab_label.add(label)
    return sorted(list(vocab_label))


def make_dataset(path, w2id, tok=False):
    labels = []
    dataset = []
    token_count = 0
    unknown_count = 0
    for label, words in read_file(path, tok):
        labels.append(label)
        array = make_array(w2id, words)
        dataset.append(array)
        token_count += array.size
        unknown_count += (array == Vocab_Pad.UNK).sum()
    print('# of tokens: %d' % token_count)
    print('# of unknown: %d (%.2f %%)' % (unknown_count,
                                          100. * unknown_count / token_count))
    return labels, dataset


def make_array(word_id, words):
    ids = [word_id.get(word, Vocab_Pad.UNK) for word in words]
    return np.array(ids, 'i')

if __name__ == "__main__":
    args = get_preprocess_args()

    print(json.dumps(args.__dict__, indent=4))

    # Vocab Construction
    train_path = os.path.join(args.input, args.train_filename)
    valid_path = os.path.join(args.input, args.dev_filename)
    test_path = os.path.join(args.input, args.test_filename)
    unlabel_path = os.path.join(args.input, args.unlabel_filename)

    train_word_cntr = count_words(train_path, args.vocab_size, args.tok)
    valid_word_cntr = count_words(valid_path, args.vocab_size, args.tok)
    unlabel_word_cntr = count_words(unlabel_path, args.vocab_size, args.tok)
    
    all_words = list(set(train_word_cntr + valid_word_cntr + unlabel_word_cntr))

    vocab = ['<pad>', '<eos>', '<unk>', '<bos>'] + all_words
    w2id = {word: index for index, word in enumerate(vocab)}
    
    # tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    label_list = get_label_vocab(train_path, args.tok)
    label2id = {l: index for index, l in enumerate(label_list)}

    # Unlabelled Dataset
    labels, data = make_dataset(unlabel_path, w2id, args.tok)
    print('Original unlabelled data size: %d' % len(data))
    unlab_r = []
    unlabel_data = []
    for i, (l, s) in enumerate(six.moves.zip(labels, data)):
        if 0 < len(s) < args.max_seq_length:
            unlabel_data.append((-1, s))
        else:
            unlab_r.append(i)
    
    print('Filtered unlabelled data size: %d' % len(unlabel_data))
    print('Removed unlabelled data: %d' % len(unlab_r))

    # Train Dataset
    labels, data = make_dataset(train_path, w2id, args.tok)
    print('Original training data size: %d' % len(data))
    train_r = []
    train_data = []
    for i, (l, s) in enumerate(six.moves.zip(labels, data)):
        if 0 < len(s) < args.max_seq_length:
            train_data.append((label2id[l], s))
        else:
            train_r.append(i)
    
    print('Removed training data: %d' % len(train_r))
    print('Filtered training data size: %d' % len(train_data))
    
    # Valid Dataset
    labels, data = make_dataset(valid_path, w2id, args.tok)
    valid_data = [(label2id[l], s) for l, s in six.moves.zip(labels, data)
                  if 0 < len(s)]
    
    print('Filtered validation data size: %d' % len(valid_data))

    # Test Dataset
    labels, data = make_dataset(test_path, w2id, args.tok)
    test_r = []
    test_data = []
    for i, (l, s) in enumerate(six.moves.zip(labels, data)):
        if 0 < len(s):
            test_data.append((label2id[l], s))
        else:
            test_r.append(i)
    
    print('Filtered testing data size: %d' % len(test_data))
    
    # External Knowledge
    ek = []
    ek_t = []
    ek_u = []
    
    train_ek_d = np.load(args.input+args.data_path+"train_ek_count.npy")
    test_ek_d = np.load(args.input+args.data_path+"test_ek_count.npy")
    unlab_ek_d = np.load(args.input+args.data_path+"unlab_ek_count.npy")
    
    for i in train_ek_d:
        with open(args.input+args.ekdata + "/train/filename" + str(i) + '.pkl', 'rb') as f:
            a = torch.from_numpy(pickle.load(f))
            if(a.shape[0] > args.ek_dim):
                a = a[:args.ek_dim]
            ek.append(a)
    for i in test_ek_d:
        with open(args.input+args.ekdata + "/test/filename" + str(i) + '.pkl', 'rb') as f:
            a = torch.from_numpy(pickle.load(f))
            if(a.shape[0] > args.ek_dim):
                a = a[:args.ek_dim]
            ek_t.append(a)
    for i in unlab_ek_d:
        with open(args.input+args.ekdata + "/unlab/filename" + str(i) + '.pkl', 'rb') as f:
            a = torch.from_numpy(pickle.load(f))
            if(a.shape[0] > args.ek_dim):
                a = a[:args.ek_dim]
            ek_u.append(a)
        
    ek = pad_sequence(ek, batch_first = True)
    ek_t = pad_sequence(ek_t, batch_first = True)
    ek_u = pad_sequence(ek_u, batch_first = True)
    
    ek = np.delete(ek, train_r, 0)
    ek_t = np.delete(ek_t, test_r, 0)
    ek_u = np.delete(ek_u, unlab_r, 0)
    
    print('EK Data: ')
    print(f'Train: {len(ek)}, Test: {len(ek_t)}, Unlab: {len(ek_u)}')
    
    # Addn Dataset
    print('Preparing addn data...')
    addn_data = np.load(args.addndata + "train_features.npy")
    
    print('Train Addn')
    addn_data_1 = np.delete(addn_data, train_r, 0)
    
    addn_data = torch.from_numpy(addn_data_1)
    
    print('Filtered addn train data:', addn_data.shape)
    
    addn_data_t = np.load(args.addndata + "test_features.npy")
    if args.behaviour_test:
        addn_data_fr = torch.from_numpy(np.delete(np.load(args.addndata + "test_features_fr.npy"), test_r, 0))
        addn_data_f = torch.from_numpy(np.delete(np.load(args.addndata + "test_features_f.npy"), test_r, 0))
        addn_data_r = torch.from_numpy(np.delete(np.load(args.addndata + "test_features_r.npy"), test_r, 0))
        
        print('Load and Filtered behaviour testing data:', addn_data_fr.shape, addn_data_f.shape, addn_data_r.shape)
    
    print('Test Addn')
    addn_data_t_1 = np.delete(addn_data_t, test_r, 0)
    
    addn_data_t = torch.from_numpy(addn_data_t_1)
    
    print('Filtered addn testing data:', addn_data_t.shape)

    addn_data_unlab = np.load(args.addndata + "unlab_features.npy")
    
    print('Unlab Addn')
    addn_data_unlab_1 = np.delete(addn_data_unlab, unlab_r, 0)
    
    addn_data_unlab = torch.from_numpy(addn_data_unlab_1)
    
    print('Filtered addn unlabelled data:', addn_data_unlab.shape)

    # Display corpus statistics
    print("Vocab: {}".format(len(vocab)))

    id2w = {i: w for w, i in w2id.items()}
    id2label = {i: l for l, i in label2id.items()}

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Save the dataset as pytorch serialized files
    torch.save(unlabel_data,
               os.path.join(args.output, args.save_data + '.unlabel.pth'))
    torch.save(train_data,
               os.path.join(args.output, args.save_data + '.train.pth'))
    torch.save(valid_data,
               os.path.join(args.output, args.save_data + '.valid.pth'))
    torch.save(test_data,
               os.path.join(args.output, args.save_data + '.test.pth'))
    
    torch.save(addn_data_unlab,
               os.path.join(args.output, args.save_data + '.unlabel_addn.pth'))
    torch.save(addn_data,
               os.path.join(args.output, args.save_data + '.train_addn.pth'))
    torch.save(addn_data_t,
               os.path.join(args.output, args.save_data + '.valid_addn.pth'))
    torch.save(addn_data_t,
               os.path.join(args.output, args.save_data + '.test_addn.pth'))
    if args.behaviour_test:
        torch.save(addn_data_fr,
                   os.path.join(args.output, args.save_data + '.test_addn_fr.pth'))
        torch.save(addn_data_f,
                   os.path.join(args.output, args.save_data + '.test_addn_f.pth'))
        torch.save(addn_data_r,
                   os.path.join(args.output, args.save_data + '.test_addn_r.pth'))
    
    torch.save(ek,
               os.path.join(args.output, args.save_data + '.ek.pth'))
    torch.save(ek_t,
               os.path.join(args.output, args.save_data + '.ek_t.pth'))
    torch.save(ek_u,
               os.path.join(args.output, args.save_data + '.ek_u.pth'))
    
    torch.save(torch.from_numpy(np.delete(np.load(args.input+args.data_path+'train_g_embs.npy'), train_r, 0)),
               os.path.join(args.output, args.save_data + '.train_graph_embs.pth'))
    torch.save(torch.from_numpy(np.delete(np.load(args.input+args.data_path+'test_g_embs.npy'), test_r, 0)),
               os.path.join(args.output, args.save_data + '.test_graph_embs.pth'))
    torch.save(torch.from_numpy(np.delete(np.load(args.input+args.data_path+'unlab_g_embs.npy'), unlab_r, 0)),
               os.path.join(args.output, args.save_data + '.unlab_graph_embs.pth'))

    # Save the word vocab
    with open(os.path.join(args.output, args.save_data + '.vocab.pickle'), 'wb') as f:
        pickle.dump(id2w, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save the label vocab
    with open(os.path.join(args.output, args.save_data + '.label.pickle'),'wb') as f:
        pickle.dump(id2label, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print('Done')