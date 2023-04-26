import os
from os.path import join
import torch
from parse import parse_args
import multiprocessing


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()
ROOT_PATH = os.path.dirname(os.path.abspath(__file__ + "/../"))
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')

config = {}

# Default hparams
dataset = args.dataset
model_name = args.model
seed = args.seed
GPU_NUM = args.gpu
parallel = args.parallel
topks = eval(args.topks)
config['test_u_batch_size'] = args.testbatch
# EASE hparams
config['reg_p'] = args.reg_p
config['diag_const'] = args.diag_const
# ELDAE hparams
config['drop_p'] = args.drop_p
# RLAE hparams
config['xi'] = args.xi
# GF-CF hparams
config['alpha'] = args.alpha

CORES = multiprocessing.cpu_count() // 2

if GPU_NUM == -1:
    device = torch.device('cpu')
else:
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU


from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)


def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")


print ('Current cuda device ', torch.cuda.current_device()) # check