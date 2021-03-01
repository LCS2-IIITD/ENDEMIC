from config import get_preprocess_args
import pickle

args = get_preprocess_args()

pickle.dump(args, open('preprocess_args.pkl', 'wb'))