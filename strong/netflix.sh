cd code &&
python main.py --dataset="netflix" --model="GFCF" --alpha=1.0 &&
python main.py --dataset="netflix" --model="EASE" --diag_const=False --reg_p=40000 &&
python main.py --dataset="netflix" --model="EASE" --reg_p=1000 &&
python main.py --dataset="netflix" --model="RLAE" --reg_p=2000 --xi=0.5 &&
python main.py --dataset="netflix" --model="EDLAE" --diag_const=False --reg_p=900 --drop_p=0.4 &&
python main.py --dataset="netflix" --model="EDLAE" --reg_p=600 --drop_p=0.3 &&
python main.py --dataset="netflix" --model="RDLAE" --reg_p=700 --drop_p=0.3 --xi=0.4