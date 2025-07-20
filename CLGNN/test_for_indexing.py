import numpy as np
train_index=np.array([0,1,2,3,4])
valid_index=np.array([5,6,7])
test_index=np.array([8,9,10,11])

train_node_nums=len(train_index)
valid_node_nums=len(valid_index)
test_node_nums=len(test_index)
total_num_nodes=train_node_nums+valid_node_nums+test_node_nums
total=np.concatenate([train_index, valid_index, test_index])
shuffle=np.random.permutation(total_num_nodes)
train_index=total[shuffle[:train_node_nums]]
valid_index=total[shuffle[train_node_nums:train_node_nums+valid_node_nums]]
test_index=total[shuffle[train_node_nums+valid_node_nums:total_num_nodes]]

print(type(train_index))
print(train_index)
print(valid_index)
print(test_index)

