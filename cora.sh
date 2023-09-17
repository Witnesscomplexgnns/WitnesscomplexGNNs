#!/bin/bash
for i in {1..5}; do
	python witcompnn_main.py --seed $i --method transformer --topo local --ptb_rate 0 --dataset cora # runs GCN + LWTL
 	#python test_gcn.py --ptb_rate 0.2 --dataset citeseer --seed $i # runs GCN
	# python witcompnn_main.py --seed $i --method transformer --topo both --ptb_rate 0 --dataset cora # runs GCN + LWTL + GWTL + Topoloss
done
