cd code &&
python main.py --dataset="ml-20m" --model="GFCF" --alpha=1.0 &&
python main.py --dataset="ml-20m" --model="EASE" --diag_const=False --reg_p=10000 &&
python main.py --dataset="ml-20m" --model="EASE" --reg_p=800 &&
python main.py --dataset="ml-20m" --model="RLAE" --reg_p=1000 --xi=0.6 &&
python main.py --dataset="ml-20m" --model="EDLAE" --diag_const=False --reg_p=1000 --drop_p=0.3 &&
python main.py --dataset="ml-20m" --model="EDLAE" --reg_p=400 --drop_p=0.3 &&
python main.py --dataset="ml-20m" --model="RDLAE" --reg_p=1000 --drop_p=0.2 --xi=0.6
