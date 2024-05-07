# run downstream_plus
# python downstream_plus.py --dataset Coauthor_CS --seed 0 --pretrained_model pretrained_gnn/Citation0.15.pth --load_components_path downstream_model/prompt_pool_coauthor_cs_task0.pth downstream_model/prompt_pool_coauthor_physics_task0.pth --input_dim 500 --hidden_dim 200 --output_dim 200 --prompt_dim 200
# python downstream_plus.py --dataset Coauthor_CS --seed 40 --pretrained_model pretrained_gnn/Citation0.15.pth --load_components_path downstream_model/prompt_pool_coauthor_cs_task0.pth downstream_model/prompt_pool_coauthor_physics_task0.pth --input_dim 500 --hidden_dim 200 --output_dim 200 --prompt_dim 200
# python downstream_plus.py --dataset Coauthor_CS --seed 42 --pretrained_model pretrained_gnn/Citation0.15.pth --load_components_path downstream_model/prompt_pool_coauthor_cs_task0.pth downstream_model/prompt_pool_coauthor_physics_task0.pth --input_dim 500 --hidden_dim 200 --output_dim 200 --prompt_dim 200

# python downstream_plus.py --dataset Coauthor_Physics --seed 0 --pretrained_model pretrained_gnn/Citation0.15.pth --load_components_path downstream_model/prompt_pool_coauthor_physics_task0.pth downstream_model/prompt_pool_coauthor_cs_task0.pth --input_dim 500 --hidden_dim 200 --output_dim 200 --prompt_dim 200
# python downstream_plus.py --dataset Coauthor_Physics --seed 40 --pretrained_model pretrained_gnn/Citation0.15.pth --load_components_path downstream_model/prompt_pool_coauthor_physics_task0.pth downstream_model/prompt_pool_coauthor_cs_task0.pth --input_dim 500 --hidden_dim 200 --output_dim 200 --prompt_dim 200
# python downstream_plus.py --dataset Coauthor_Physics --seed 42 --pretrained_model pretrained_gnn/Citation0.15.pth --load_components_path downstream_model/prompt_pool_coauthor_physics_task0.pth downstream_model/prompt_pool_coauthor_cs_task0.pth --input_dim 500 --hidden_dim 200 --output_dim 200 --prompt_dim 200

# python downstream_plus.py --dataset Amazon_Computer --seed 0 --pretrained_model 'pretrained_gnn/Amazon(114514).pth' --load_components_path downstream_model/prompt_pool_amazon_computer_task0.pth downstream_model/prompt_pool_amazon_photo_task0.pth downstream_model/prompt_pool_amazon_fraud_task0.pth --input_dim 100 --hidden_dim 100 --output_dim 100 --prompt_dim 100
# python downstream_plus.py --dataset Amazon_Computer --seed 40 --pretrained_model 'pretrained_gnn/Amazon(114514).pth' --load_components_path downstream_model/prompt_pool_amazon_computer_task0.pth downstream_model/prompt_pool_amazon_photo_task0.pth downstream_model/prompt_pool_amazon_fraud_task0.pth --input_dim 100 --hidden_dim 100 --output_dim 100 --prompt_dim 100
# python downstream_plus.py --dataset Amazon_Computer --seed 42 --pretrained_model 'pretrained_gnn/Amazon(114514).pth' --load_components_path downstream_model/prompt_pool_amazon_computer_task0.pth downstream_model/prompt_pool_amazon_photo_task0.pth downstream_model/prompt_pool_amazon_fraud_task0.pth --input_dim 100 --hidden_dim 100 --output_dim 100 --prompt_dim 100

# python downstream_plus.py --dataset Amazon_Photo --seed 0 --pretrained_model 'pretrained_gnn/Amazon(114514).pth' --load_components_path downstream_model/prompt_pool_amazon_photo_task0.pth downstream_model/prompt_pool_amazon_computer_task0.pth downstream_model/prompt_pool_amazon_fraud_task0.pth --input_dim 100 --hidden_dim 100 --output_dim 100 --prompt_dim 100
# python downstream_plus.py --dataset Amazon_Photo --seed 40 --pretrained_model 'pretrained_gnn/Amazon(114514).pth' --load_components_path downstream_model/prompt_pool_amazon_photo_task0.pth downstream_model/prompt_pool_amazon_computer_task0.pth downstream_model/prompt_pool_amazon_fraud_task0.pth --input_dim 100 --hidden_dim 100 --output_dim 100 --prompt_dim 100
# python downstream_plus.py --dataset Amazon_Photo --seed 42 --pretrained_model 'pretrained_gnn/Amazon(114514).pth' --load_components_path downstream_model/prompt_pool_amazon_photo_task0.pth downstream_model/prompt_pool_amazon_computer_task0.pth downstream_model/prompt_pool_amazon_fraud_task0.pth --input_dim 100 --hidden_dim 100 --output_dim 100 --prompt_dim 100

# python downstream_plus.py --dataset Amazon_Fraud --seed 0 --pretrained_model 'pretrained_gnn/Amazon(114514).pth' --load_components_path downstream_model/prompt_pool_amazon_fraud_task0.pth downstream_model/prompt_pool_amazon_computer_task0.pth downstream_model/prompt_pool_amazon_photo_task0.pth --input_dim 100 --hidden_dim 100 --output_dim 100 --prompt_dim 100 --k_hop 1 --k_hop_nodes 50 --lr 0.001
# python downstream_plus.py --dataset Amazon_Fraud --seed 40 --pretrained_model 'pretrained_gnn/Amazon(114514).pth' --load_components_path downstream_model/prompt_pool_amazon_fraud_task0.pth downstream_model/prompt_pool_amazon_computer_task0.pth downstream_model/prompt_pool_amazon_photo_task0.pth --input_dim 100 --hidden_dim 100 --output_dim 100 --prompt_dim 100 --k_hop 1 --k_hop_nodes 50 --lr 0.001
# python downstream_plus.py --dataset Amazon_Fraud --seed 42 --pretrained_model 'pretrained_gnn/Amazon(114514).pth' --load_components_path downstream_model/prompt_pool_amazon_fraud_task0.pth downstream_model/prompt_pool_amazon_computer_task0.pth downstream_model/prompt_pool_amazon_photo_task0.pth --input_dim 100 --hidden_dim 100 --output_dim 100 --prompt_dim 100 --k_hop 1 --k_hop_nodes 50 --lr 0.001

#python downstream_plus.py --dataset Cora --seed 42 --pretrained_model pretrained_gnn/Citation0.15.pth --load_components_path downstream_model/prompt_pool_coauthor_physics_task0.pth downstream_model/prompt_pool_coauthor_cs_task0.pth --input_dim 500 --hidden_dim 200 --output_dim 200 --prompt_dim 200
# python downstream_plus.py --dataset Cora --seed 0 --pretrained_model pretrained_gnn/Citation0.15.pth --load_components_path downstream_model/prompt_pool_coauthor_physics_task0.pth downstream_model/prompt_pool_coauthor_cs_task0.pth --input_dim 500 --hidden_dim 200 --output_dim 200 --prompt_dim 200
# python downstream_plus.py --dataset Cora --seed 40 --pretrained_model pretrained_gnn/Citation0.15.pth --load_components_path downstream_model/prompt_pool_coauthor_physics_task0.pth downstream_model/prompt_pool_coauthor_cs_task0.pth --input_dim 500 --hidden_dim 200 --output_dim 200 --prompt_dim 200

# python downstream_plus.py --dataset Yelp_Fraud --seed 0 --pretrained_model 'pretrained_gnn/Amazon(114514).pth' --load_components_path downstream_model/prompt_pool_amazon_fraud_task0.pth downstream_model/prompt_pool_amazon_computer_task0.pth downstream_model/prompt_pool_amazon_photo_task0.pth --input_dim 100 --hidden_dim 100 --output_dim 100 --prompt_dim 100 --k_hop 1 --k_hop_nodes 50 --lr 0.01
# python downstream_plus.py --dataset Yelp_Fraud --seed 40 --pretrained_model 'pretrained_gnn/Amazon(114514).pth' --load_components_path downstream_model/prompt_pool_amazon_fraud_task0.pth downstream_model/prompt_pool_amazon_computer_task0.pth downstream_model/prompt_pool_amazon_photo_task0.pth --input_dim 100 --hidden_dim 100 --output_dim 100 --prompt_dim 100 --k_hop 1 --k_hop_nodes 50 --lr 0.01
# python downstream_plus.py --dataset Yelp_Fraud --seed 42 --pretrained_model 'pretrained_gnn/Amazon(114514).pth' --load_components_path downstream_model/prompt_pool_amazon_fraud_task0.pth downstream_model/prompt_pool_amazon_computer_task0.pth downstream_model/prompt_pool_amazon_photo_task0.pth --input_dim 100 --hidden_dim 100 --output_dim 100 --prompt_dim 100 --k_hop 1 --k_hop_nodes 50 --lr 0.01


# python downstream.py --dataset Coauthor_CS --seed 0 --pretrained_model pretrained_gnn/Citation0.15.pth --input_dim 500 --hidden_dim 200 --output_dim 200 --prompt_dim 200
# python downstream.py --dataset Coauthor_CS --seed 40 --pretrained_model pretrained_gnn/Citation0.15.pth --input_dim 500 --hidden_dim 200 --output_dim 200 --prompt_dim 200
# python downstream.py --dataset Coauthor_CS --seed 42 --pretrained_model pretrained_gnn/Citation0.15.pth --input_dim 500 --hidden_dim 200 --output_dim 200 --prompt_dim 200

# python downstream.py --dataset Coauthor_Physics --seed 0 --pretrained_model pretrained_gnn/Citation0.15.pth --input_dim 500 --hidden_dim 200 --output_dim 200 --prompt_dim 200
# python downstream.py --dataset Coauthor_Physics --seed 40 --pretrained_model pretrained_gnn/Citation0.15.pth --input_dim 500 --hidden_dim 200 --output_dim 200 --prompt_dim 200
# python downstream.py --dataset Coauthor_Physics --seed 42 --pretrained_model pretrained_gnn/Citation0.15.pth --input_dim 500 --hidden_dim 200 --output_dim 200 --prompt_dim 200

# python downstream.py --dataset Amazon_Computer --seed 0 --pretrained_model 'pretrained_gnn/Amazon(114514).pth' --input_dim 100 --hidden_dim 100 --output_dim 100 --prompt_dim 100
# python downstream.py --dataset Amazon_Computer --seed 40 --pretrained_model 'pretrained_gnn/Amazon(114514).pth' --input_dim 100 --hidden_dim 100 --output_dim 100 --prompt_dim 100
# python downstream.py --dataset Amazon_Computer --seed 42 --pretrained_model 'pretrained_gnn/Amazon(114514).pth' --input_dim 100 --hidden_dim 100 --output_dim 100 --prompt_dim 100

# python downstream.py --dataset Amazon_Photo --seed 0 --pretrained_model 'pretrained_gnn/Amazon(114514).pth' --input_dim 100 --hidden_dim 100 --output_dim 100 --prompt_dim 100
# python downstream.py --dataset Amazon_Photo --seed 40 --pretrained_model 'pretrained_gnn/Amazon(114514).pth' --input_dim 100 --hidden_dim 100 --output_dim 100 --prompt_dim 100
# python downstream.py --dataset Amazon_Photo --seed 42 --pretrained_model 'pretrained_gnn/Amazon(114514).pth' --input_dim 100 --hidden_dim 100 --output_dim 100 --prompt_dim 100



# python downstream.py --dataset Yelp_Fraud --seed 0 --pretrained_model 'pretrained_gnn/Amazon(114514).pth' --input_dim 100 --hidden_dim 100 --output_dim 100 --prompt_dim 100 --k_hop 1 --k_hop_nodes 50 --lr 0.005
# python downstream.py --dataset Yelp_Fraud --seed 40 --pretrained_model 'pretrained_gnn/Amazon(114514).pth' --input_dim 100 --hidden_dim 100 --output_dim 100 --prompt_dim 100 --k_hop 1 --k_hop_nodes 50 --lr 0.005
python downstream.py --dataset Yelp_Fraud --seed 1145 --pretrained_model 'pretrained_gnn/Amazon(114514).pth' --input_dim 100 --hidden_dim 100 --output_dim 100 --prompt_dim 100 --k_hop 1 --k_hop_nodes 50 --lr 0.005