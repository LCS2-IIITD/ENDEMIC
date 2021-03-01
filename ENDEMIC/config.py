import os
from argparse import ArgumentParser
from datetime import datetime
import random


def get_train_args():
    parser = ArgumentParser(description='Text Classification')

    # Model
    parser.add_argument('--use_addn', type=bool, default=True,
                        help='Whether to use additional features or not')
    
    parser.add_argument('--attention', type=bool, default=True,
                        help='To use attention or not')
    
    parser.add_argument('--behaviour_test', type=bool, 
                        default=True,
                        help='To perform behaviour evaluation')
    
    parser.add_argument('--model', default='WordLstm', type=str)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--checkPth', type=str, default=None)
    parser.add_argument('--d_units', type=int, default=300,
                        help='Number of units in the embedding layer')
    parser.add_argument('--d_proj', type=int, default=768,
                        help='Number of units in the projection layer')
    parser.add_argument('--d_hidden', type=int, default=256,
                        help='Number of units in the hidden layer')
    parser.add_argument('--addn_dim', type=int, default=19,
                        help='Number of units in the addn data')
    parser.add_argument('--projection', dest='projection', action='store_true',
                        help="projection layer after word embeddings")
    parser.set_defaults(projection=False)
    parser.add_argument('--d_down_proj', type=int, default=300,
                        help='Number of units in the down projection layer')
    parser.add_argument('--down_projection', dest='down_projection',
                        action='store_true',
                        help="down projection layer after encoder")
    parser.set_defaults(down_projection=False)

    parser.add_argument('--num_discriminator_layers', type=int, default=3,
                        help='Number of discriminative layers')

    # Option for forward LSTM or BiLSTM
    parser.add_argument('--frnn', dest='brnn', action='store_false')
    parser.set_defaults(brnn=True)

    parser.add_argument('--timedistributed', dest='timedistributed',
                        action='store_true',
                        help='option to get all the hidden states of LSTM')
    parser.set_defaults(timedistributed=False)

    parser.add_argument('--init_scalar', default=0.05, type=float)

    # parser.add_argument('--kernel_size_list', default=[1, 2, 3, 4, 5, 6, 7], type=list)
    # parser.add_argument('--num_cnn_features_list', default=[50, 50, 50, 50, 50, 50, 50], type=list)
    # parser.add_argument('--word_kernel_size_list', default=[3, 4, 5], type=list)
    # parser.add_argument('--num_word_cnn_features_list', default=[400, 300, 300], type=list)

    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--unif', help='Initializer bounds for embeddings',
                        default=0.25)

    # Multi-GPU
    parser.add_argument('--multi_gpu', dest='multi_gpu', action='store_true')
    parser.set_defaults(multi_gpu=False)
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0, 1, 2, 3])

    # Gradient Clipping
    parser.add_argument('--gradient_clipping', dest='gradient_clipping', action='store_true')
    parser.add_argument('--no_gradient_clipping', dest='gradient_clipping', action='store_false')
    parser.set_defaults(gradient_clipping=True)
    parser.add_argument('--max_norm', default=1.0, type=float)
    # parser.add_argument('--max_len', default=200, type=int)

    # Weight Decay
    parser.add_argument('--weight_decay', default=0.0, type=float)

    # Evaluation and Checkpoint
    parser.add_argument('--load_checkpoint', default=True, type=bool)

    # Pre-trained word embeddings path
    parser.add_argument('--use_pretrained_embeddings',
                        dest='use_pretrained_embeddings', action='store_true')
    parser.set_defaults(use_pretrained_embeddings=False)
    parser.add_argument('--train_embeddings', default=True, type=bool)
    parser.add_argument('--adaptive_dropout', dest='adaptive_dropout', action='store_true')
    parser.add_argument('--finetune', dest='finetune', action='store_true')
    parser.set_defaults(finetune=False)

    home = os.environ['HOME']

    parser.add_argument('--max_iter', default=None)
    parser.add_argument('--nepochs', default=20, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--word_dropout', default=0.5, type=float)
    parser.add_argument('--lstm_dropout', default=0.5, type=float)
    parser.add_argument('--locked_dropout', default=0.5, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--nepoch_no_imprv', default=3, type=int)
    parser.add_argument('--nchkp_no_imprv', default=30, type=int)
    parser.add_argument('--hidden_size', default=1024, type=int)
    parser.add_argument('--subsampling', default=1e-4, type=float)
    parser.add_argument('--class_weight', default='uniform', type=str)
    parser.add_argument('--optim', default='adam', type=str)
    parser.add_argument('--eval_steps', default=1000, type=int)

    # Optimizer Params
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--lr_decay', default=0.5, type=float)
    # Paper: https://arxiv.org/pdf/1707.05589.pdf (Beta1 is set to 0)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.999, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--patience', default=20, type=int)
    parser.add_argument('--scheduler', default='ReduceLROnPlateau',
                        type=str, help='ReduceLROnPlateau|ExponentialLR')
    parser.add_argument('--gamma', default=0.99995, type=float)

    parser.add_argument('--pool_type', default='max_pool', type=str)
    parser.add_argument('--dynamic_pool_size', default=20, type=int)
    parser.add_argument('--wbatchsize', default=3000, type=int)
    parser.add_argument('--wbatchsize_unlabel', default=12000, type=int)

    parser.add_argument('--lambda_clf', default=1.0, type=float)
    parser.add_argument('--lambda_ae', default=0.0, type=float)
    parser.add_argument('--lambda_at', default=0.0, type=float)
    parser.add_argument('--lambda_vat', default=0.0, type=float)
    parser.add_argument('--lambda_entropy', default=0.0, type=float)
    parser.add_argument('--inc_unlabeled_loss', dest='inc_unlabeled_loss', action='store_true')
    parser.set_defaults(inc_unlabeled_loss=False)
    parser.add_argument('--unlabeled_loss_type', default='AvgTrainUnlabel', type=str)

    parser.add_argument('--perturb_norm_length', default=5.0, type=float)
    parser.add_argument('--max_embedding_norm', default=None, type=float)

    parser.add_argument('--normalize_embedding', dest='normalize_embedding', action='store_true')
    parser.set_defaults(normalize_embedding=False)

    parser.add_argument('--add_noise', dest='add_noise', action='store_true')
    parser.set_defaults(add_noise=False)
    parser.add_argument('--noise_dropout', default=0.1, type=float)
    parser.add_argument('--random_permutation', default=3, type=int)

    # Debug Mode
    # In debug mode, print progress bar, otherwise not
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    parser.add_argument('--report_every', default=100, type=int)

    parser.add_argument('--input', '-i', type=str,
                        default='temp',
                        help='Input directory')
    parser.add_argument('--save_data', type=str,
                        default='demo',
                        help='Prefix for the prepared data')
    
    parser.add_argument('--addndata', type=str,
                    default='data/covid19/',
                    help='For Tweet and User Features')

    parser.add_argument('--output_path', default="results/clf/", type=str)
    parser.add_argument('--exp_name', required=True, type=str)
    # parser.add_argument('--model_file', default='my_model.pt', type=str)

    # Corpus Name
    parser.add_argument('--corpus', default='sst', type=str)
    args = parser.parse_args()

    # args.model_output = os.path.join(args.output_path, "model.weights")
    # args.checkpoint_dir = args.model_output

    args.model_file = args.exp_name + ".pt"
    now = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
    random_num = random.randint(1, 1000)
    log_path = os.path.join(args.output_path, "log_{}_time-{}_rand_{}.txt".format(now,
                                                                                  args.exp_name,
                                                                                  random_num))
    args.log_path = log_path
    return args


def get_preprocess_args():
    """Data Preprocessing Options"""

    parser = ArgumentParser(description='Pre-process Options')
    parser.add_argument('--vocab_size', type=int, default=80000,
                        help='Vocabulary size of source language')
    parser.add_argument('--no_tok', dest='tok', action='store_false')
    parser.set_defaults(tok=True)
    parser.add_argument('--max_seq_length', type=int, default=1000,
                        help='maximum sequence length')
    parser.add_argument('--input', '-i', type=str,
                        default='data',
                        help='Input directory')
    parser.add_argument('--output', '-o', type=str,
                        default='temp/ECTF/data',
                        help='Output directory')
    parser.add_argument('--save_data', type=str,
                        default='demo',
                        help='Output file for the prepared data')
    parser.add_argument('--addndata', type=str,
                        default='data/ECTF/actual/',
                        help='For Tweet and User Features')
    
    parser.add_argument('--ek_dim', type=int, 
                        default=1,
                        help='Maximum dimension for EK Embeddings')
    
    parser.add_argument('--behaviour_test', type=bool, 
                        default=True,
                        help='To perform behaviour evaluation')
    
    # PolitiFact
    parser.add_argument('--politifact_dev',
                        default='politifact/dev.txt',
                        type=str)
    parser.add_argument('--politifact_test',
                        default='politifact/test.txt',
                        type=str)
    parser.add_argument('--politifact_train',
                        default='politifact/train.txt',
                        type=str)
    parser.add_argument('--politifact_unlabel',
                        default='politifact/train.txt',
                        type=str)

    # GossipCon
    parser.add_argument('--gossipcon_dev',
                        default='gossipcon/dev.txt',
                        type=str)
    parser.add_argument('--gossipcon_test',
                        default='gossipcon/test.txt',
                        type=str)
    parser.add_argument('--gossipcon_train',
                        default='gossipcon/train.txt',
                        type=str)
    parser.add_argument('--gossipcon_unlabel',
                        default='gossipcon/unlabel.txt',
                        type=str)
    
    # ECTF
    parser.add_argument('--ECTF_dev',
                        default='ECTF/actual/test.txt',
                        type=str)  # No dev set, test and dev set are the same
    parser.add_argument('--ECTF_test',
                        default='ECTF/actual/test.txt',
                        type=str)
    parser.add_argument('--ECTF_train',
                        default='ECTF/actual/train.txt',
                        type=str)
    parser.add_argument('--ECTF_unlabel',
                        default='ECTF/actual/unlabel.txt',
                        type=str)
    
    parser.add_argument('--ECTF_debug_dev',
                        default='ECTF/debug/test.txt',
                        type=str)  # No dev set, test and dev set are the same
    parser.add_argument('--ECTF_debug_test',
                        default='ECTF/debug/test.txt',
                        type=str)
    parser.add_argument('--ECTF_debug_train',
                        default='ECTF/debug/train.txt',
                        type=str)
    parser.add_argument('--ECTF_debug_unlabel',
                        default='ECTF/debug/unlabel.txt',
                        type=str)
    
    parser.add_argument('--ekdata',
                        default='/ECTF/actual/ek',
                        type=str)
    parser.add_argument('--data_path',
                        default='/ECTF/actual/', type=str)

    # Corpus Name
    parser.add_argument('--corpus', default='ECTF', type=str)

    args = parser.parse_args()
    args.train_filename = eval("args." + args.corpus + "_train")
    args.dev_filename = eval("args." + args.corpus + "_dev")
    args.test_filename = eval("args." + args.corpus + "_test")
    args.unlabel_filename = eval("args." + args.corpus + "_unlabel")

    return args