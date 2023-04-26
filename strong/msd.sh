cd code &&
python main.py --dataset="msd" --model="GFCF" --alpha=0.0 &&
python main.py --dataset="msd" --model="EASE" --diag_const=False --reg_p=4000 &&
python main.py --dataset="msd" --model="EASE" --reg_p=100 &&
python main.py --dataset="msd" --model="RLAE" --reg_p=100 --xi=0.5 &&
python main.py --dataset="msd" --model="EDLAE" --diag_const=False --reg_p=30 --drop_p=0.4 &&
python main.py --dataset="msd" --model="EDLAE" --reg_p=70 --drop_p=0.3 &&
python main.py --dataset="msd" --model="RDLAE" --reg_p=80 --drop_p=0.2 --xi=0.4