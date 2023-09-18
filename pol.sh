#!/bin/bash
d="polblogs"
for i in {1..5}; do
  python witcompnn_mainL.py --dataset $d --seed $i --method transformer --topo local --ptb_rate 0 --device cuda:0 & 
	python witcompnn_mainL.py --dataset $d --seed $i --method transformer --topo local --ptb_rate 0.05 --device cuda:0 &
	python witcompnn_mainL.py --dataset $d --seed $i --method transformer --topo local --ptb_rate 0.1 --device cuda:0 
  python witcompnn_mainL.py --dataset $d --seed $i --method transformer --topo local --ptb_rate 0.15 --device cuda:1 & 
	python witcompnn_mainL.py --dataset $d --seed $i --method transformer --topo local --ptb_rate 0.2 --device cuda:1 &
	python witcompnn_mainL.py --dataset $d --seed $i --method transformer --topo local --ptb_rate 0.25 --device cuda:2 
  #python test_gcn.py --ptb_rate 0 --dataset $d --seed $i & # run GCN
  #python test_gcn.py --ptb_rate 0.05 --dataset $d --seed $i & # run GCN
  #python test_gcn.py --ptb_rate 0.1 --dataset $d --seed $i & # run GCN
  #python test_gcn.py --ptb_rate 0.15 --dataset $d --seed $i & # run GCN
  #python test_gcn.py --ptb_rate 0.2 --dataset $d --seed $i & # run GCN
  #python test_gcn.py --ptb_rate 0.25 --dataset $d --seed $i # run GCN
  
  #python witcompnn_main.py --seed $i --method transformer --topo both --ptb_rate 0 --dataset $d --device cuda:1 &
	#python witcompnn_main.py --seed $i --method transformer --topo both --ptb_rate 0.05 --dataset $d --device cuda:1 & # runs GCN + LWTL + GWTL + Topoloss
	#python witcompnn_main.py --seed $i --method transformer --topo both --ptb_rate 0.1 --dataset $d --device cuda:1 & # runs GCN + LWTL + GWTL + Topoloss
	#python witcompnn_main.py --seed $i --method transformer --topo both --ptb_rate 0.15 --dataset $d --device cuda:2 & # runs GCN + LWTL + GWTL + Topoloss
	#python witcompnn_main.py --seed $i --method transformer --topo both --ptb_rate 0.2 --dataset $d --device cuda:2 &
	#python witcompnn_main.py --seed $i --method transformer --topo both --ptb_rate 0.25 --dataset $d --device cuda:2
done
