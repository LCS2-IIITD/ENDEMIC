import json
import pickle
import os
import torch

from config import get_train_args
from training import Training
from general_utils import get_logger

import pandas as pd

import numpy as np

args = get_train_args()

with open('config.pickle', 'wb') as f:
    pickle.dump(args, f)

print('---CONFIG SAVED----')

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)
logger = get_logger(args.log_path)
logger.info(json.dumps(args.__dict__, indent=4))

# Reading the int indexed text dataset
train_data = torch.load(os.path.join(args.input, args.save_data + ".train.pth"))
dev_data = torch.load(os.path.join(args.input, args.save_data + ".valid.pth"))
test_data = torch.load(os.path.join(args.input, args.save_data + ".test.pth"))
unlabel_data = torch.load(os.path.join(args.input, args.save_data + ".unlabel.pth"))

addn_data = torch.load(os.path.join(args.input, args.save_data + ".train_addn.pth"))
addn_data_t = torch.load(os.path.join(args.input, args.save_data + ".valid_addn.pth"))
addn_data_t = torch.load(os.path.join(args.input, args.save_data + ".test_addn.pth"))
addn_data_unlab = torch.load(os.path.join(args.input, args.save_data + ".unlabel_addn.pth"))

if args.behaviour_test:
    addn_data_fr = torch.load(os.path.join(args.input, args.save_data + ".test_addn_fr.pth"))
    addn_data_f = torch.load(os.path.join(args.input, args.save_data + ".test_addn_f.pth"))
    addn_data_r = torch.load(os.path.join(args.input, args.save_data + ".test_addn_r.pth"))

ek = torch.load(os.path.join(args.input, args.save_data + ".ek.pth"))
ek_t = torch.load(os.path.join(args.input, args.save_data + ".ek_t.pth"))
ek_u = torch.load(os.path.join(args.input, args.save_data + ".ek_u.pth"))

graph_embs = torch.load(os.path.join(args.input, args.save_data + ".train_graph_embs.pth"))
graph_embs_t = torch.load(os.path.join(args.input, args.save_data + ".test_graph_embs.pth"))
graph_embs_u = torch.load(os.path.join(args.input, args.save_data + ".unlab_graph_embs.pth"))

# Reading the word vocab file
with open(os.path.join(args.input, args.save_data + '.vocab.pickle'),
          'rb') as f:
    id2w = pickle.load(f)

# Reading the label vocab file
with open(os.path.join(args.input, args.save_data + '.label.pickle'),
          'rb') as f:
    id2label = pickle.load(f)

args.id2w = id2w
args.n_vocab = len(id2w)
args.id2label = id2label
args.num_classes = len(id2label)

object = Training(args, logger)

logger.info('Corpus: {}'.format(args.corpus))
logger.info('Pytorch Model')
logger.info(repr(object.embedder))
logger.info(repr(object.encoder))
logger.info(repr(object.clf))
logger.info(repr(object.clf_loss))
if args.lambda_ae:
    logger.info(repr(object.ae))

# Train the model
if args.behaviour_test:
    object(train_data, dev_data, test_data, unlabel_data,
           addn_data, addn_data_unlab, addn_data_t,
           ek, ek_t, ek_u,
           graph_embs, graph_embs_t, graph_embs_u,
           addn_data_fr, addn_data_f, addn_data_r,
           mode=args.mode, checkPth=args.checkPth)
else:
    object(train_data, dev_data, test_data, unlabel_data,
           addn_data, addn_data_unlab, addn_data_t,
           ek, ek_t, ek_u,
           graph_embs, graph_embs_t, graph_embs_u)