import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--name",
    type=str,
    default="ER",
    help="Name of processed file, without .txt",
)
args = parser.parse_args()

f=open(args.name, 'r')
f_eval=open(args.name+'_Acc_eval.txt','w')
f_train=open(args.name+'_Acc_train.txt','w')
f_loss_eval=open(args.name+'_Loss_eval.txt','w')
f_loss_train=open(args.name+'_Loss_train.txt','w')


lines=f.readlines()
n=len(lines)
print(f"N? {n}")
eval_acc=[]
train_acc=[]
eval_loss=[]
train_loss=[]

for line in lines:
    if ("Top1_Acc_Exp" in line):
        eval_acc.append(line[-7:-1])
        f_eval.write(line[-7:-1]+'\n')
        if(len(eval_acc)%6==0):
            f_eval.write('\n')
    elif("Top1_Acc_Epoch" in line):
        train_acc.append(line[-7:-1])
        f_train.write(line[-7:-1]+'\n')
        if(len(train_acc)%50==0):
            f_train.write('\n')
    elif("Loss_Exp" in line):
        eval_loss.append(line[-7:-1])
        f_loss_eval.write(line[-7:-1]+'\n')
        if(len(eval_loss)%6==0):
            f_loss_eval.write('\n')
    elif("Loss_Epoch" in line):
        train_loss.append(line[-7:-1])
        f_loss_train.write(line[-7:-1]+'\n')
        if(len(train_loss)%50==0):
            f_loss_train.write('\n')

sum_acc=0
sum_loss=0
flag=0
for l in eval_acc[-6:]:
    print(l)
    if(flag==0):
        sum_acc+=5*eval(l)
        flag=1
    else:
        sum_acc+=eval(l)
print("Average eval Acc. : ",sum_acc/10)

flag=0
for l in eval_loss[-6:]:
    print(l)
    if(flag==0):
        sum_loss+=5*eval(l)
        flag=1
    else:
        sum_loss+=eval(l)
print("Average eval Loss. : ",sum_loss/10)


f.close()
f_train.close()
f_eval.close()
f_loss_train.close()
f_loss_eval.close()

