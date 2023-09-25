#!/bin/bash
d="cora"
for i in {1..5}; do
	# python test_gcn.py --ptb_rate 0.05 --dataset $d --seed $i &
	# python test_gcn.py --ptb_rate 0.1 --dataset $d --seed $i &
	# python test_gcn.py --ptb_rate 0.15 --dataset $d --seed $i &
	# python test_gcn.py --ptb_rate 0.2 --dataset $d --seed $i &
	# python test_gcn.py --ptb_rate 0.25 --dataset $d --seed $i 
 	#python test_gcn.py --ptb_rate 0 --dataset $d --seed $i & 
	#python test_gcn.py --ptb_rate 1 --dataset $d --seed $i &
	#python test_gcn.py --ptb_rate 2 --dataset $d --seed $i &
	#python test_gcn.py --ptb_rate 3 --dataset $d --seed $i &
	#python test_gcn.py --ptb_rate 4 --dataset $d --seed $i &
	#python test_gcn.py --ptb_rate 5 --dataset $d --seed $i
  
  	#python witcompnn_main.py --seed $i --method transformer --topo both --ptb_rate 0 --dataset $d --device cuda:0 & 
	#python witcompnn_main.py --seed $i --method transformer --topo both --ptb_rate 1 --dataset $d --device cuda:0 & # runs GCN + LWTL + GWTL + Topoloss
	#python witcompnn_main.py --seed $i --method transformer --topo both --ptb_rate 2 --dataset $d --device cuda:0 & # runs GCN + LWTL + GWTL + Topoloss
	#python witcompnn_main.py --seed $i --method transformer --topo both --ptb_rate 3 --dataset $d --device cuda:1  # runs GCN + LWTL + GWTL + Topoloss
	#python witcompnn_main.py --seed $i --method transformer --topo both --ptb_rate 4 --dataset $d --device cuda:0 &
	#python witcompnn_main.py --seed $i --method transformer --topo both --ptb_rate 5 --dataset $d --device cuda:0
done
