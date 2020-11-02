import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module

class SEModule_FC(Module):
    '''
    原生SE模块，提取通道注意力权重
    '''
    def __init__(self, channels, reduction,L=4):
        super(SEModule_FC, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        d = max(channels // reduction, L)  # 计算向量Z 的长度d   下限为L
        self.fc = nn.Sequential(
            nn.Linear(channels, d, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d, channels, bias=False),
            nn.Sigmoid()
        )
        # 未初始化权重

    def forward(self, x):
        batch, channle,  _, _ = x.size()
        x = self.avg_pool(x).view(batch, channle)
        x = self.fc(x).view(batch, channle, 1, 1)
        return x


class SEModule_Conv(Module):
    '''
    SE模块提取通道注意力权重（Conv替代FC层）
    '''
    def __init__(self, channels, reduction):
        super(SEModule_Conv, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)

        torch.nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        # 未初始化fc2
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return  x

class Select_TopK(Module):
    '''
    根据 注意力权重，选择 TopK 对应的特征通道
    '''
    def __init__(self,top_k):
        super(Select_TopK,self).__init__()
        self.top_k=top_k # 屏蔽的通道数

    def forward(self, x):
        # x∈[batch,channel,1,1]
        batch, channel, h, w = x.shape
        x = x.squeeze()
        # 通道维度 按权重降序时的下标值(索引)
        topk_index = torch.argsort(x, dim=1, descending=True)[:,:self.top_k].long()
        #  初始化全零
        T = torch.zeros_like(x)

        value = torch.ones_like(x)  # 填充值
        # dim=1 行不变（依次递增）列变化   index:列索引值，用于变化列的值  src:填充值
        T.scatter_(dim=1,index=topk_index,src=value)
        return T.reshape(batch, channel, h, w)

class Local_Mask(Module):
    '''
    针对标记通道，生成 位置mask
    '''
    def __init__(self,drop_block):
        super(Local_Mask, self).__init__()
        self.drop_block=torch.FloatTensor([drop_block]) # 屏蔽矩阵大小
    def forward(self,x,T):
        '''
        :param x: 输入特征  [batch,channel,H,W]
        :param T: 标记通道T 0-1  [batch,channel,1,1]
        :return:
        '''
        batch, channel, h, w = x.shape
        assert h >= self.drop_block.item() and w >= self.drop_block.item()
        # ===========提取目标特征图x_mask===========
        x=x.reshape(-1,h,w)
        T=T.reshape(-1,1)
        index = torch.nonzero(T, as_tuple=True)[0] #非0索引
        x_mask=x[index]
        # ===========每张特征图最大值索引max_index==========
        x_mask_squeeze=x_mask.reshape(x_mask.shape[0],-1)
        index=x_mask_squeeze.argmax(dim=1)
        max_index = torch.cat(((index//h).unsqueeze(-1), (index % h).unsqueeze(-1)), 1)
        # ===========生成kxk屏蔽矩阵S===========
        S=torch.ones_like(x_mask)

        max_index_h=max_index[:,0]
        max_index_w=max_index[:, 1]

        h1=max_index_h-(self.drop_block / 2.).floor().to(max_index_h.device)
        h2=max_index_h+(self.drop_block / 2.).floor().to(max_index_h.device)
        w1=max_index_w-(self.drop_block / 2.).floor().to(max_index_w.device)
        w2 = max_index_w + (self.drop_block / 2.).floor().to(max_index_w.device)
        h1, h2 = torch.clamp(torch.cat((h1.unsqueeze(0), h2.unsqueeze(0))), min=0, max=h - 1).int().cpu().numpy().tolist()  # 防止越界
        w1, w2 = torch.clamp(torch.cat((w1.unsqueeze(0), w2.unsqueeze(0))), min=0, max=w - 1).int().cpu().numpy().tolist()
        for i in range(len(S)):
            S[i, h1[i]:h2[i], w1[i]:w2[i]] = 0

        # 加权并规范化
        S_reshape=S.reshape(S.shape[0],-1)
        λ =S_reshape.shape[-1]/S_reshape.sum(dim=-1) # 规范化系数 公式（6）
        x_mask=x_mask * S * λ.reshape(λ.shape[0],1,1)

        # 根据序号放回原特征图中
        x[index]=x_mask
        return x.reshape(batch, channel, h, w)
class TargetDrop(Module):
    def __init__(self, channels,reduction=16,drop_prob=0.15,drop_block=5):
        '''
        论文默认参数  reduction=16,drop_prob=0.15,drop_block=5
        :param channels:   输入特征的通道数
        :param reduction:  SE通道注意力
        :param drop_prob: 屏蔽通道数占总通道数的百分比
        :param drop_block: 屏蔽矩阵大小
        '''
        super(TargetDrop, self).__init__()

        # 提取通道注意力权重
        self.SEModule_Conv = SEModule_Conv(channels,reduction)
        # self.SEModule_FC = SEModule_FC(channels, reduction)

        # 标记通道T 0-1
        self.Select_TopK = Select_TopK(int(channels * drop_prob))

        # 针对标记通道，生成 位置mask
        self.Local_Mask = Local_Mask(drop_block)
    def forward(self, x):
        '''
        :param x: 输入特征
        :return:
        '''
        if self.training:
            # 通道注意力权重  M∈[batch,channel,1,1]
            M = self.SEModule_Conv(x)
            # 标记通道 0-1    T∈[batch,channel,1,1]
            T = self.Select_TopK(M)
            # 生成 位置mask   Mask_X∈[batch,channel,H,W]
            return self.Local_Mask(x,T)
        else:
            return x  # eval模式无需drop

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    feature = torch.rand(8,64, 24,24).cuda()
    target_drop = TargetDrop(channels=feature.shape[1]).cuda()
    target_drop =target_drop.train() # 训练模式

    result = target_drop(feature)
    print(result.size())