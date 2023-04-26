cd code &&
python main.py --dataset="yelp2018" --model="EASE" --diag_const=False --reg_p=500 &&
python main.py --dataset="yelp2018" --model="EASE" --reg_p=300 &&
python main.py --dataset="yelp2018" --model="RLAE" --reg_p=300 --xi=0.4 &&
python main.py --dataset="yelp2018" --model="EDLAE" --diag_const=False --reg_p=200 --drop_p=0.7 &&
python main.py --dataset="yelp2018" --model="EDLAE" --reg_p=200 --drop_p=0.6 &&
python main.py --dataset="yelp2018" --model="RDLAE" --reg_p=200 --drop_p=0.7 --xi=0.2