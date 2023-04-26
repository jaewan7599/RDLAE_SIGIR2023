python main.py --dataset="abook" --model="EASE" --diag_const=False --reg_p=90 &&
python main.py --dataset="abook" --model="EASE" --reg_p=70 &&
python main.py --dataset="abook" --model="RLAE" --reg_p=60 --xi=0.6 &&
python main.py --dataset="abook" --model="EDLAE" --diag_const=False --reg_p=60 --drop_p=0.2 &&
python main.py --dataset="abook" --model="EDLAE" --reg_p=50 --drop_p=0.2 &&
python main.py --dataset="abook" --model="RDLAE" --reg_p=60 --drop_p=0.1 --xi=0.6