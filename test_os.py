import os
if os.path.exists('min_epoch_loss.csv'):
  with open('min_epoch_loss.csv','r') as _rf:
    min_epoch_loss = float(_rf.readline())
else:
  min_epoch_loss = 9999.9999
print(min_epoch_loss)
with open("min_epoch_loss.csv",'w') as f:
  f.write(str(min_epoch_loss-0.0001))