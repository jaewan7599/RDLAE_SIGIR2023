cd code &&
python main.py --dataset="gowalla" --model="GFCF" --alpha=0.1 &&
python main.py --dataset="gowalla" --model="EASE" --diag_const=False --reg_p=300 &&
python main.py --dataset="gowalla" --model="EASE" --reg_p=80 &&
python main.py --dataset="gowalla" --model="RLAE" --reg_p=80 --xi=0.4 &&
python main.py --dataset="gowalla" --model="EDLAE" --diag_const=False --reg_p=100 --drop_p=0.5 &&
python main.py --dataset="gowalla" --model="EDLAE" --reg_p=90 --drop_p=0.6 &&
python main.py --dataset="gowalla" --model="RDLAE" --reg_p=70 --drop_p=0.5 --xi=0.4