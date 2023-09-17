#!/bin/bash
for i in {1..5}; do
#	python witcompnn_main.py --dataset citeseer --seed $i --method transformer --topo local --ptb_rate 0.15 --device cuda:0 & 
	#python witcompnn_main.py --dataset citeseer --seed $i --method transformer --topo local --ptb_rate 0.2 --device cuda:1 &
	python witcompnn_main.py --dataset citeseer --seed $i --method transformer --topo local --ptb_rate 0 --device cuda:2 
 	#python test_gcn.py --ptb_rate 0.25 --dataset citeseer --seed $i # run GCN
	#python witcompnn_main.py --seed $i --method transformer --topo both --ptb_rate 0 --dataset citeseer # runs GCN + LWTL + GWTL + Topoloss
done