import torch

import torch.nn.functional as F
import torch.nn


class Connectome_Filter_Block(torch.nn.Module):
    
    '''Connectome Filter Block'''

    def __init__(self, n_filt, planes, bias=False):
       
        super(Connectome_Filter_Block, self).__init__() #initialize
       
        self.d = 100 
        self.in_planes = 1
        
        self.cnn1 = torch.nn.Conv2d(n_filt,planes,(1,self.d),bias=bias) #row 
        self.cnn2 = torch.nn.Conv2d(n_filt,planes,(self.d,1),bias=bias) #column

        
    def forward(self, x, l, g_flag):
        
        '''
        Input : 
            x -> rs-fMRI connectome
            l -> DTI Laplacian
            g_flag -> graph filtering on
        '''
        
        if g_flag: #graph pre-filtering if True
            x = torch.matmul(l,x) 
        
        r = self.cnn1(x) #row filtering
        c = self.cnn2(x) #column filterning
        
        return torch.cat([r]*self.d,3)+torch.cat([c]*self.d,2)

class M_GCN(torch.nn.Module):
    
    def __init__(self,  num_classes =2):
        super(M_GCN, self).__init__()
        
        self.in_planes = 1
        self.d = 100
        
        self.cf_1 = Connectome_Filter_Block(1,32,bias=True)      
        self.ef_1 = torch.nn.Conv2d(32,1,(1,self.d))
        self.nf_1 = torch.nn.Conv2d(1,256,(self.d,1))
        
        #ANN for regression
        self.dense1 = torch.nn.Linear(256,128)
        self.dense2 = torch.nn.Linear(128,30)
        self.dense3 = torch.nn.Linear(30,num_classes)
        
    def forward(self, data, g_f = True):
        x = data[...,0]
        l = data[...,1]
        
        x = x.unsqueeze(dim=1)
        l = l.unsqueeze(dim=1)
        out = F.leaky_relu(self.cf_1(x, l, g_f))

        if g_f:  # graph filtering     
            out = torch.matmul(l,out) # graph filtering
            
        out = F.leaky_relu(self.ef_1(out))
       
        if g_f:  # graph filtering     
            out = torch.matmul(l,out)
       
        out = F.leaky_relu(self.nf_1(out))
     
        #regression
        out = out.view(out.size(0), -1)     
        out = F.dropout(F.leaky_relu(self.dense1(out)),p=0.3,training = self.training)
        out = F.dropout(F.leaky_relu(self.dense2(out)),p=0.3,training = self.training)
        out = F.dropout(F.leaky_relu(self.dense3(out)),p=0.3,training = self.training)
        out = F.softmax(out,1)
        return out
    
def init_weights(m):
    
    if type(m) == torch.nn.Linear:
            
            torch.nn.init.xavier_normal(m.weight, gain=torch.nn.init.calculate_gain('relu'))
            m.bias.data.fill_(1e-02)