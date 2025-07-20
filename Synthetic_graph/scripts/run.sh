python3 SBM_graph.py --graphnum TEST1 --feat_dim 5 --num_node 3000
python3 SBM_graph.py --graphnum TEST15 --feat_dim 5 --num_node 3000
python3 SBM_graph.py --graphnum TEST2 --feat_dim 5 --num_node 3000
python3 SBM_graph.py --graphnum TEST25 --feat_dim 5 --num_node 3000
python3 SBM_graph.py --graphnum TEST3 --feat_dim 5 --num_node 3000
python3 SBM_graph.py --graphnum TEST35 --feat_dim 5 --num_node 3000
python3 SBM_graph.py --graphnum TEST4 --feat_dim 5 --num_node 3000
python3 SBM_graph.py --graphnum TEST45 --feat_dim 5 --num_node 3000
python3 SBM_graph.py --graphnum TEST5 --feat_dim 5 --num_node 3000

python3 SBM_graph.py --graphnum TEST1 --sym --feat_dim 5 --num_node 3000
python3 SBM_graph.py --graphnum TEST15 --sym --feat_dim 5 --num_node 3000
python3 SBM_graph.py --graphnum TEST2 --sym --feat_dim 5 --num_node 3000
python3 SBM_graph.py --graphnum TEST25 --sym --feat_dim 5 --num_node 3000
python3 SBM_graph.py --graphnum TEST3 --sym --feat_dim 5 --num_node 3000
python3 SBM_graph.py --graphnum TEST35 --sym --feat_dim 5 --num_node 3000
python3 SBM_graph.py --graphnum TEST4 --sym --feat_dim 5 --num_node 3000
python3 SBM_graph.py --graphnum TEST45 --sym --feat_dim 5 --num_node 3000
python3 SBM_graph.py --graphnum TEST5 --sym --feat_dim 5 --num_node 3000

python3 SBM_train.py --dataset TEST1_feat5_node3000 --epoch 2000
python3 SBM_train.py --dataset TEST15_feat5_node3000 --epoch 2000
python3 SBM_train.py --dataset TEST2_feat5_node3000 --epoch 2000
python3 SBM_train.py --dataset TEST25_feat5_node3000 --epoch 2000
python3 SBM_train.py --dataset TEST3_feat5_node3000 --epoch 2000
python3 SBM_train.py --dataset TEST35_feat5_node3000 --epoch 2000
python3 SBM_train.py --dataset TEST4_feat5_node3000 --epoch 2000
python3 SBM_train.py --dataset TEST45_feat5_node3000 --epoch 2000
python3 SBM_train.py --dataset TEST5_feat5_node3000 --epoch 2000

python3 SBM_train.py --dataset TEST1_sym_feat5_node3000 --epoch 2000
python3 SBM_train.py --dataset TEST15_sym_feat5_node3000 --epoch 2000
python3 SBM_train.py --dataset TEST2_sym_feat5_node3000 --epoch 2000
python3 SBM_train.py --dataset TEST25_sym_feat5_node3000 --epoch 2000
python3 SBM_train.py --dataset TEST3_sym_feat5_node3000 --epoch 2000
python3 SBM_train.py --dataset TEST35_sym_feat5_node3000 --epoch 2000
python3 SBM_train.py --dataset TEST4_sym_feat5_node3000 --epoch 2000
python3 SBM_train.py --dataset TEST45_sym_feat5_node3000 --epoch 2000
python3 SBM_train.py --dataset TEST5_sym_feat5_node3000 --epoch 2000

# Approx. 16min