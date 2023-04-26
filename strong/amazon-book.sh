cd code &&
python main.py --dataset="abook" --model="GFCF" --alpha=0.0 &&
python main.py --dataset="abook" --model="EASE" --diag_const=False --reg_p=100 &&
python main.py --dataset="abook" --model="EASE" --reg_p=80 &&
python main.py --dataset="abook" --model="RLAE" --reg_p=70 --xi=0.5 &&
python main.py --dataset="abook" --model="EDLAE" --diag_const=False --reg_p=70 --drop_p=0.4 &&
python main.py --dataset="abook" --model="EDLAE" --reg_p=60 --drop_p=0.6 &&
python main.py --dataset="abook" --model="RDLAE" --reg_p=50 --drop_p=0.4 --xi=0.4