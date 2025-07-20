print(f"Processing {i+1}th features... {time.time()-t0}", flush=True)
f_log.write(f"Processing {i+1}th features... {time.time()-t0}\n")
f_log.flush()

clone_x=np.copy(x)
shapes = list(clone_x.shape)
print(f"shapes : {shapes}\n")
f_log.write(f"shapes : {shapes}\n")
f_log.flush()

# Calculte means
train_mean = torch.zeros((218, 172, clone_x.shape[1]))
train_cnt = torch.zeros((218, 172))
train_time_mean = torch.zeros((218, clone_x.shape[1]))
train_time_cnt = torch.zeros(218)
test_cnt = 0
test_mean = torch.zeros((clone_x.shape[1]))

for u in tqdm(range(shapes[0])):
    if paper_year[u] == 2019:
        test_cnt += 1
        test_mean += clone_x[u]
    elif paper_year[u] <= 2017:
        t=max(0,paper_year[u] - 1800)
        if labels[u]<0 or labels[u]>=172:
                continue
        train_cnt[t][labels[u]] += 1
        train_time_cnt[t] += 1
        train_mean[t][labels[u]] += clone_x[u]



# Add norm values
test_var = 0
rsq = torch.zeros(218)
msq = torch.zeros(218)
for u in tqdm(range(shapes[0])):
    if paper_year[u] == 2019:
        test_var += torch.norm(clone_x[u] - test_mean) ** 2
    elif paper_year[u] <= 2017:
        t = max(0,paper_year[u] - 1800)
        if labels[u]<0 or labels[u]>=172:
                continue
        msq[t] += torch.norm(train_mean[t][labels[u]] - train_time_mean[t]) ** 2
        rsq[t] += torch.norm(clone_x[u] - train_mean[t][labels[u]]) ** 2

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


# Update modified vals
for u in tqdm(range(shapes[0])):
    if paper_year[u] <= 2017:
        t = max(0,paper_year[u] - 1800)
        if labels[u]<0 or labels[u]>=172:
                continue
        clone_x[u] = alpha[t] * clone_x[u] + (1 - alpha[t]) * train_mean[t][labels[u]]