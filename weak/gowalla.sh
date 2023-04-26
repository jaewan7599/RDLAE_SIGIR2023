cd code &&
# python main.py --dataset="gowalla" --model="EASE" --diag_const=False --reg_p=500 &&
python main.py --dataset="gowalla" --model="EASE" --reg_p=80 &&
# python main.py --dataset="gowalla" --model="EDLAE" --diag_const=False --reg_p=60 --drop_p=0.9 &&
python main.py --dataset="gowalla" --model="EDLAE" --reg_p=60 --drop_p=0.8 &&
# python main.py --dataset="gowalla" --model="RLAE" --reg_p=100 --xi=0.2 &&
# python main.py --dataset="gowalla" --model="RDLAE" --reg_p=80 --drop_p=0.8 --xi=0.09
wait