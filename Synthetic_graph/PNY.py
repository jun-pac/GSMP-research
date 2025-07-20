import torch

# class PNY(torch.nn.Module):
#     def __init__(self, labels, times, num_time, num_label, split, relative_connectivity):
#         super().__init__()
#         self.labels = labels
#         self.times = times
#         self.num_time = num_time
#         self.num_label = num_label
#         self.split = split
#         self.P=relative_connectivity
#         self.train_idx = (self.times<self.split).nonzero().squeeze()
#         self.test_idx = (self.times>=self.split).nonzero().squeeze()

#     def forward(self, prev_x, x):
#         clone_x=torch.clone(x)
#         previous_cov_matrix=[torch.cov(prev_x[torch.squeeze((self.labels==y).nonzero())].T) for y in range(self.num_label)]

#         # Calculate covariances
#         current_cov_matrix=torch.zeros((self.num_label, self.num_time, x.shape[1], x.shape[1]))
#         for y1 in range(self.num_label):
#             for t1 in range(self.num_time):
#                 denom=0
#                 for y2 in range(self.num_label):
#                     for t2 in range(self.num_time):
#                         denom+=self.P[y1][t1][y2][t2]*(2 if abs(t2-t1)>min(self.num_time-1-t1,t1) else 1)
#                 denom=denom*denom
#                 for y2 in range(self.num_label):
#                     temp=0
#                     for t2 in range(self.num_time):
#                         temp+=self.P[y1][t1][y2][t2]*(4 if abs(t2-t1)>min(self.num_time-1-t1,t1) else 1)
#                     temp=temp/denom
#                     current_cov_matrix[y1][t1]+=temp*previous_cov_matrix[y2]
#         #print(f"cur cov matrix : {current_cov_matrix[0][0]}")
#         #print(f"some Ps {self.P[0][0][0][0]} {self.P[0][4][1][3]} {self.P[1][4][3][2]}")

#         # Calculate feature transform matrix
#         cov_transform_matrix=torch.zeros((self.num_label, self.num_time, x.shape[1], x.shape[1]))
#         for y1 in range(self.num_label):
#             L_max, Q_max=torch.linalg.eigh(current_cov_matrix[y1][self.num_time - 1])

#             L_max=L_max.real
#             Q_max=Q_max.real
#             L_max_sq=torch.sqrt(L_max)
#             L_max_sq=torch.diag(L_max_sq)
#             for t1 in range(self.split):
#                 L, Q=torch.linalg.eigh(current_cov_matrix[y1][t1])
#                 L=L.real
#                 Q=Q.real
#                 L_sq=torch.sqrt(L)
#                 Inv_L_sq=1/L_sq
#                 Inv_L_sq=torch.diag(Inv_L_sq)
#                 cov_transform_matrix[y1][t1]=Q_max@L_max_sq@Inv_L_sq@Q.T

#         # Calculate feature means
#         train_mean = torch.zeros((self.num_time, self.num_label, x.shape[1]), requires_grad=False)
#         train_cnt = torch.zeros((self.num_time, self.num_label), requires_grad=False)
#         for u in range(x.shape[0]):
#             t=self.times[u]
#             train_cnt[t][self.labels[u]] += 1
#             train_mean[t][self.labels[u]] += x[u]                 
#         for t in range(self.num_time):
#             for l in range(self.num_label):
#                 train_mean[t][l]=train_mean[t][l]/max(1,train_cnt[t][l])

#         # Update modified vals
#         for u in self.train_idx:
#             t = self.times[u]
#             y = self.labels[u]
#             clone_x[u] = cov_transform_matrix[y][t]@(x[u]-train_mean[t][y])+train_mean[t][y]
#         return clone_x


class PNY(torch.nn.Module):
    def __init__(self, labels, times, num_time, num_label, split, relative_connectivity):
        super().__init__()
        self.labels = labels
        self.times = times
        self.num_time = num_time
        self.num_label = num_label
        self.split = split
        self.P=relative_connectivity
        self.train_idx = (self.times<self.split).nonzero().squeeze()
        self.test_idx = (self.times>=self.split).nonzero().squeeze()

    def forward(self, prev_x, x):
        clone_x=torch.clone(x)
        previous_cov_matrix=[torch.cov(prev_x[torch.squeeze((self.labels==y).nonzero())].T) for y in range(self.num_label)]

        # Calculate covariances
        current_cov_matrix=torch.zeros((self.num_label, self.num_time, x.shape[1], x.shape[1]))
        for y1 in range(self.num_label):
            for t1 in range(self.num_time):
                denom=0
                for y2 in range(self.num_label):
                    for t2 in range(self.num_time):
                        denom+=self.P[y1][t1][y2][t2]*(2 if abs(t2-t1)>min(self.num_time-1-t1,t1) else 1)
                denom=denom*denom
                for y2 in range(self.num_label):
                    temp=0
                    for t2 in range(self.num_time):
                        temp+=self.P[y1][t1][y2][t2]*self.P[y1][t1][y2][t2]*(4 if abs(t2-t1)>min(self.num_time-1-t1,t1) else 1)
                    temp=temp/denom
                    current_cov_matrix[y1][t1]+=temp*previous_cov_matrix[y2]
        #print(f"cur cov matrix : {current_cov_matrix[0][0]}")
        #print(f"some Ps {self.P[0][0][0][0]} {self.P[0][4][1][3]} {self.P[1][4][3][2]}")

        # Calculate feature transform matrix
        cov_transform_matrix=torch.zeros((self.num_label, self.num_time, x.shape[1], x.shape[1]))
        for y1 in range(self.num_label):
            L_max, Q_max=torch.linalg.eigh(current_cov_matrix[y1][self.num_time - 1])

            L_max=L_max.real
            Q_max=Q_max.real
            L_max_sq=torch.sqrt(L_max)
            L_max_sq=torch.diag(L_max_sq)
            for t1 in range(self.split):
                L, Q=torch.linalg.eigh(current_cov_matrix[y1][t1])
                L=L.real
                Q=Q.real
                L_sq=torch.sqrt(L)
                Inv_L_sq=1/L_sq
                Inv_L_sq=torch.diag(Inv_L_sq)
                cov_transform_matrix[y1][t1]=Q_max@L_max_sq@Inv_L_sq@Q.T

        # Calculate feature means
        train_mean = torch.zeros((self.num_time, self.num_label, x.shape[1]), requires_grad=False)
        train_cnt = torch.zeros((self.num_time, self.num_label), requires_grad=False)
        for u in range(x.shape[0]):
            t=self.times[u]
            train_cnt[t][self.labels[u]] += 1
            train_mean[t][self.labels[u]] += x[u]                 
        for t in range(self.num_time):
            for l in range(self.num_label):
                train_mean[t][l]=train_mean[t][l]/max(1,train_cnt[t][l])

        # Update modified vals
        for u in self.train_idx:
            t = self.times[u]
            y = self.labels[u]
            clone_x[u] = cov_transform_matrix[y][t]@(x[u]-train_mean[t][y])+train_mean[t][y]
        return clone_x
