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
config['test_u_batch_size'] = args.testbatch
# EASE hparams
config['reg_p'] = args.reg_p
config['diag_const'] = args.diag_const
# EDLAE hparams
config['drop_p'] = args.drop_p
# RLAE hparams
config['xi'] = args.xi

GPU_NUM = args.gpu
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

if GPU_NUM == -1:
    device = torch.device('cpu')
else:
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU

dataset = args.dataset
model_name = args.model

topks = eval(args.topks)

# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)


def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")
    
    
print ('Current cuda device ', torch.cuda.current_device()) # check