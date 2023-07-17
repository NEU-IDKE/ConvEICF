from xml.dom import xmlbuilder
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
class InfoNCE(nn.Module):
    def __init__(self, temperature=0.01):
        super().__init__()
        self.temperature = temperature

    def forward(self, values,labels):
        return self.info_nce(values, labels)
    
    def normalize(self, x):
        return F.normalize(x, dim=-1)
    
    def info_nce(self,values, labels):
        logits = self.normalize(values)
        logits = logits/self.temperature
        logits = torch.softmax(logits, dim = -1)
        return F.binary_cross_entropy(logits, labels)

def get_param(shape):
    param = nn.Parameter(torch.Tensor(*shape))
    nn.init.xavier_normal_(param.data)
    return param


class LTEModel(nn.Module):
    def __init__(self, num_ents, num_rels, params=None):
        super(LTEModel, self).__init__()

        self.p = params
        self.bceloss = InfoNCE(self.p.temperature)

        self.init_embed = get_param((num_ents, self.p.init_dim))
        self.init_rel = get_param((num_rels * 2, self.p.init_dim))
        self.bias = nn.Parameter(torch.zeros(num_ents))

        #Embedding Enhancement Module
        self.h_ops_dict = nn.ModuleDict({
            'p': nn.Linear(self.p.init_dim, self.p.gcn_dim, bias=False),
            'b': nn.BatchNorm1d(self.p.gcn_dim),
            'd': nn.Dropout(self.p.hid_drop),
            'a': nn.Tanh(),
        })

        self.t_ops_dict = nn.ModuleDict({
            'p': nn.Linear(self.p.init_dim, self.p.gcn_dim, bias=False),
            'b': nn.BatchNorm1d(self.p.gcn_dim),
            'd': nn.Dropout(self.p.hid_drop),
            'a': nn.Tanh(),
        })

        self.r_ops_dict = nn.ModuleDict({
            'p': nn.Linear(self.p.init_dim, self.p.gcn_dim, bias=False),
            'b': nn.BatchNorm1d(self.p.gcn_dim),
            'd': nn.Dropout(self.p.hid_drop),
            'a': nn.Tanh(),
        })

        self.x_ops = self.p.x_ops
        self.r_ops = self.p.r_ops
        self.diff_ht = False

    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)

    def exop(self, x, r, x_ops=None, r_ops=None, diff_ht=False):
        x_head = x_tail = x
        if len(x_ops) > 0:
            for x_op in x_ops.split("."):
                if diff_ht:
                    x_head = self.h_ops_dict[x_op](x_head)
                    x_tail = self.t_ops_dict[x_op](x_tail)
                else:
                    x_head = x_tail = self.h_ops_dict[x_op](x_head)

        if len(r_ops) > 0:
            for r_op in r_ops.split("."):
                r = self.r_ops_dict[r_op](r)

        return x_head, x_tail, r


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 8, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 8, in_planes, 1, bias=False))
        self.sigmoid = nn.Tanh()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Tanh()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ConvEICF(LTEModel):
    def __init__(self, num_ents, num_rels, params=None):
        super(self.__class__, self).__init__(num_ents, num_rels, params)

        self.half = self.p.num_filt0

        self.d2 = self.p.embed_dim*2
        self.d1 = self.p.embed_dim

        self.bn0 = torch.nn.BatchNorm2d(2)
        self.bn10 = torch.nn.BatchNorm2d(self.half)
        self.bn11 = torch.nn.BatchNorm2d(self.half)
        self.bn12 = torch.nn.BatchNorm2d(self.p.num_filt)
        self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)


        self.t_hidden_drop = torch.nn.Dropout(self.p.th1)
        self.t_hidden_drop1 = torch.nn.Dropout(self.p.th2)
        self.hidden_drop2 = torch.nn.Dropout(self.p.conve_hid_drop)
        self.feature_drop = torch.nn.Dropout(self.p.feat_drop)
        self.feature_drop1 = torch.nn.Dropout(self.p.feat_drop)



        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (self.d1, self.d2, self.d1)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))
        self.m_conv1 = torch.nn.Conv2d(2, out_channels=self.half, kernel_size=self.p.ker_sz,
                                       stride=1, padding=3,dilation=1, bias=self.p.bias)
        self.m_conv2 = torch.nn.Conv2d(self.half, out_channels=self.p.num_filt, kernel_size=self.p.ker_sz,
                                       stride=1, padding=3,dilation=1, bias=self.p.bias)
        
        flat_sz_h = int(self.p.k_w) 
        flat_sz_w = self.p.k_h
        self.flat_sz = flat_sz_h * flat_sz_w


        self.fc = torch.nn.Linear(self.flat_sz*self.p.num_filt, self.p.embed_dim)
        self.con = torch.nn.Linear(self.p.embed_dim*2, self.p.embed_dim*2, bias=False)
        
        self.channel_att = ChannelAttention(self.half)
        self.spatial = SpatialAttention()

    def Interect3D(self, e1_embed, rel_embed, W):
        x = e1_embed.view(-1, 1, self.p.embed_dim)
        W_mat = torch.mm(rel_embed, W.view(rel_embed.size(1), -1))
        W_mat = W_mat.view(-1, e1_embed.size(1), e1_embed.size(1)*2)
        W_mat = self.t_hidden_drop(W_mat)
        x = torch.bmm(x, W_mat) 
        return x
    
    def InterectLinear(self, e1_embed, rel_embed):
        x = torch.cat([e1_embed, rel_embed], dim=-1)
        x = self.con(x)
        return x

    def forward(self,sub, rel):
        x = self.init_embed
        r = self.init_rel
        x_h, x_t, r = self.exop(x, r, self.x_ops, self.r_ops)
        
        #the head entities and relations obtained from the enhanced embedding
        sub_emb = torch.index_select(x_h, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)
        all_ent = x_t

        hr_list = []
        #3D transformation
        stk_inp = self.Interect3D(sub_emb, rel_emb, self.W)
        stk_inp = stk_inp.view(-1, 1, self.p.k_w, self.p.k_h)
        hr_list.append(stk_inp)

        #linear transformation
        h_r = self.InterectLinear(sub_emb, rel_emb)
        h_r = h_r.view(-1, 1, self.p.k_w, self.p.k_h)
        hr_list.append(h_r)

        #multi-channel
        x = torch.cat(hr_list, dim = 1)

        x = self.bn0(x)
        x = self.t_hidden_drop1(x)
        x = self.m_conv1(x)
        
        x = self.bn10(x)

        #CBAM
        x = self.channel_att(x)*x
        x = self.spatial(x)*x

        x = self.bn11(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        
        x = self.m_conv2(x)

        x = self.bn12(x)
        x = F.relu(x)
        x = self.feature_drop1(x)

        x = x.view(-1, self.flat_sz*self.p.num_filt)
        x = self.fc(x)
  
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)
        return x
