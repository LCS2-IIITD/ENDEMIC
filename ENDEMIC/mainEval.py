from configEval import get_test_args
import pickle

args = get_test_args()

with open('config.pickle', 'wb') as f:
    pickle.dump(args, f)