clone_x=np.copy(x)
shapes=list(clone_x.shape)

train_mean=torch.zeros((218,172,shapes[1]))
train_cnt=torch.zeros((218,172))
train_time_mean=torch.zeros((218,shapes[1]))
train_time_cnt=torch.zeros(218)
test_cnt=0
test_mean=torch.zeros((shapes[1]))

for u in range(shapes[0]):
    if(paper_year[u]==2019):
        test_cnt+=1
        test_mean+=clone_x[u]
    elif(paper_year[u]<=2017):
        train_cnt[paper_year[u]-1800][labels[u]]+=1
        train_time_cnt[paper_year[u]-1800]+=1
        train_mean[paper_year[u]-1800][labels[u]]+=clone_x[u]

for t in range(0,218):
    for l in range(172):
        train_time_mean[t]+=train_mean[t][l]
        train_mean[t][l]/=max(1,train_cnt[t][l])
    train_time_mean[t]/=max(1,train_time_cnt[t])
test_mean/=max(1,test_cnt)

test_var=0
rsq=torch.zeros(218)
msq=torch.zeros(218)
for u in range(shapes[0]):
    if(paper_year[u]==2019):
        test_var+=torch.norm(clone_x[u]-test_mean)**2
    elif(paper_year[u]<=2017):
        t=paper_year[u]-1800
        msq[t]+=torch.norm(train_mean[t][labels[u]]-train_time_mean[t])**2
        rsq[t]+=torch.norm(clone_x[u]-train_mean[t][labels[u]])**2
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

for u in range(shapes[0]):
    if(paper_year[u]<=2017):
        t=paper_year[u]-1800
        clone_x[u]=alpha[t]*clone_x[u]+(1-alpha[t])*train_mean[t][labels[u]]

save = np.concatenate((clone_x[train_idx], clone_x[valid_idx], clone_x[test_idx]), axis=0)
    