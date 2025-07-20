import numpy as np
import torch
import multiprocessing

num_processes = multiprocessing.cpu_count()


def process_data(start, end, clone_x, paper_year, labels, result_queue):
    train_mean_part = torch.zeros((218, 172, clone_x.shape[1]))
    train_cnt_part = torch.zeros((218, 172))
    train_time_mean_part = torch.zeros((218, clone_x.shape[1]))
    train_time_cnt_part = torch.zeros(218)
    test_cnt_part = 0
    test_mean_part = torch.zeros((clone_x.shape[1]))

    for u in range(start, end):
        if paper_year[u] == 2019:
            test_cnt_part += 1
            test_mean_part += clone_x[u]
        elif paper_year[u] <= 2017:
            train_cnt_part[paper_year[u] - 1800][labels[u]] += 1
            train_time_cnt_part[paper_year[u] - 1800] += 1
            train_mean_part[paper_year[u] - 1800][labels[u]] += clone_x[u]
    result_queue.put((train_mean_part, train_cnt_part, train_time_mean_part, train_time_cnt_part, test_cnt_part, test_mean_part))


def add_norm_data(start, end):
    local_test_var = 0
    local_rsq = torch.zeros(218)
    local_msq = torch.zeros(218)

    for u in range(start, end):
        if paper_year[u] == 2019:
            local_test_var += torch.norm(clone_x[u] - test_mean) ** 2
        elif paper_year[u] <= 2017:
            t = paper_year[u] - 1800
            local_msq[t] += torch.norm(train_mean[t][labels[u]] - train_time_mean[t]) ** 2
            local_rsq[t] += torch.norm(clone_x[u] - train_mean[t][labels[u]]) ** 2
    return local_test_var, local_rsq, local_msq


def update_data(start, end):
    for u in range(start, end):
        if paper_year[u] <= 2017:
            t = paper_year[u] - 1800
            clone_x[u] = alpha[t] * clone_x[u] + (1 - alpha[t]) * train_mean[t][labels[u]]



clone_x=np.copy(x)
shapes = list(clone_x.shape)
chunk_size = shapes[0] //num_processes


# Calculte means for each (y,t)
processes = []
result_queue = multiprocessing.Queue()
for i in range(num_processes):
    start = i * chunk_size
    end = start + chunk_size if i < num_processes - 1 else shapes[0]
    p = multiprocessing.Process(target=process_data, args=(start, end, clone_x, paper_year, labels, result_queue))
    processes.append(p)
    p.start()
train_mean = torch.zeros((218, 172, clone_x.shape[1]))
train_cnt = torch.zeros((218, 172))
train_time_mean = torch.zeros((218, clone_x.shape[1]))
train_time_cnt = torch.zeros(218)
test_cnt = 0
test_mean = torch.zeros((clone_x.shape[1]))

for _ in range(num_processes):
    train_mean_part, train_cnt_part, train_time_mean_part, train_time_cnt_part, test_cnt_part, test_mean_part = result_queue.get()
    train_mean += train_mean_part
    train_cnt += train_cnt_part
    train_time_mean += train_time_mean_part
    train_time_cnt += train_time_cnt_part
    test_cnt += test_cnt_part
    test_mean += test_mean_part


# Add norm values
processes = []
result_queue = multiprocessing.Queue()
for i in range(num_processes):
    start = i * chunk_size
    end = start + chunk_size if i < num_processes - 1 else shapes[0]
    p = multiprocessing.Process(target=add_norm_data, args=(start, end))
    processes.append(p)
    p.start()
test_var = 0
rsq = torch.zeros(218)
msq = torch.zeros(218)
for p in processes:
    p.join()
    local_test_var, local_rsq, local_msq = result_queue.get()
    test_var += local_test_var
    rsq += local_rsq
    msq += local_msq


# Calculate Statistics
test_var/=max(1,test_cnt-1)
for t in range(218):
    msq[t]/=max(1,train_time_cnt[t]-1)
    rsq[t]/=max(1,train_time_cnt[t]-1)
alpha=torch.zeros(218)
for t in range(218):
    alpha_sq=(test_var-msq[t])/max(0.000001,rsq[t])
    if(alpha_sq>0):
        alpha[t]=torch.sqrt(alpha_sq)
    else:
        alpha[t]=0


processes = []
for i in range(num_processes):
    start = i * chunk_size
    end = start + chunk_size if i < num_processes - 1 else shapes[0]
    p = multiprocessing.Process(target=update_data, args=(start, end))
    processes.append(p)
    p.start()

# Join processes
for p in processes:
    p.join()