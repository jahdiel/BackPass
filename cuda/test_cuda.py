import torch
import numpy as np
print(torch.cuda.current_device())

#Measuring times
#Define cuda Event and enable timing
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)





#this will only work for cuda torch tensor !
def s(x,b):
    a=torch.pow(x,2)
    c =  a*b
    z = a+c
    y= torch.sum(z)
    print(y)
    # Backpropagation
    y.backward()





print('---- Test  ----')
for n in range(25):

    x = 100*np.random.random(size=np.power(2,n))
    b = 100*np.random.random(size=np.power(2,n)) 

    x_cuda = torch.tensor(x,requires_grad =True).cuda()
    b_cuda = torch.tensor(b,requires_grad=True).cuda()

    s(x_cuda,b_cuda)
