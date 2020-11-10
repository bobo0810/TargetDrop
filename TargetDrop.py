import os
import torch
import torch.nn as nn
from torch.nn import Conv2d, ReLU, Sigmoid, AdaptiveAvgPool2d, Module


class SEModule(Module):
    '''
    SE模块 提取通道注意力权重（Conv替代FC层）
    '''
    def __init__(self, channels, reduction,L=4):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        d = max(channels // reduction, L)  # 下限
        self.fc1 = Conv2d(channels, d , kernel_size=1, padding=0, bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(d , channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class Select_TopK(Module):
    '''
    选择 TopK 对应的特征通道
    '''
    def __init__(self, top_k):
        super(Select_TopK, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        batch, channel, h, w = x.shape
        x = x.squeeze()
        # 通道维度 按权重降序时的下标值(索引)
        topk_index = torch.argsort(x, dim=1, descending=True)[:, :self.top_k].long()
        T = torch.zeros_like(x)
        T.scatter_(dim=1, index=topk_index, src=torch.ones_like(x))
        return T.reshape(batch, channel, h, w)


class Apply_Mask(Module):
    '''
    针对标记通道，生成 位置mask
    '''
    def __init__(self, drop_block):
        super(Apply_Mask, self).__init__()
        self.drop_block = torch.FloatTensor([drop_block])  # 屏蔽矩阵大小

    def forward(self, x, T):
        '''
        :param x: 输入特征  [batch,channel,H,W]
        :param T: 标记通道T 0-1  [batch,channel,1,1]
        :return:
        '''
        batch, channel, h, w = x.shape
        self.drop_block = self.drop_block.to(x.device)
        assert h >= self.drop_block.item() and w >= self.drop_block.item()
        # 提取目标特征图x_mask
        x = x.reshape(-1, h, w)
        T = T.reshape(-1, 1)
        index = torch.nonzero(T, as_tuple=True)[0]  # 非0索引
        x_mask = x[index]
        # 每张特征图最大值索引max_index
        x_mask_squeeze = x_mask.reshape(x_mask.shape[0], -1)
        max_index = x_mask_squeeze.argmax(dim=1)
        max_index_h, max_index_w = max_index // w, max_index % w
        # 生成kxk屏蔽矩阵S  公式4
        h1 = max_index_h - (self.drop_block / 2.).floor()
        h2 = max_index_h + (self.drop_block / 2.).floor()
        w1 = max_index_w - (self.drop_block / 2.).floor()
        w2 = max_index_w + (self.drop_block / 2.).floor()
        h1, h2 = torch.clamp(torch.cat((h1.unsqueeze(0), h2.unsqueeze(0))), min=0, max=h - 1).int()
        w1, w2 = torch.clamp(torch.cat((w1.unsqueeze(0), w2.unsqueeze(0))), min=0, max=w - 1).int()
        S = torch.ones_like(x_mask)
        for i in range(len(S)):
            S[i, h1[i]:h2[i] + 1, w1[i]:w2[i] + 1] = 0  # 公式5
        # mask并规范化  公式6
        S_flatten = S.reshape(S.shape[0], -1)
        λ = S_flatten.shape[-1] / S_flatten.sum(dim=-1)
        x_mask = x_mask * S * λ.reshape(λ.shape[0], 1, 1)
        # 根据序号放回原特征图中
        x[index] = x_mask
        return x.reshape(batch, channel, h, w)


class TargetDrop(Module):
    def __init__(self, channels, reduction=16, drop_prob=0.15, drop_block=5):
        '''
        论文默认参数
        :param channels:   输入特征的通道数
        :param reduction:  SE通道注意力
        :param drop_prob:  屏蔽通道数占总通道数的百分比
        :param drop_block: 屏蔽矩阵大小
        '''
        super(TargetDrop, self).__init__()
        self.SEModule = SEModule(channels, reduction)
        self.Select_TopK = Select_TopK(int(channels * drop_prob))
        self.Apply_Mask = Apply_Mask(drop_block)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        '''
        :param x: 输入特征
        :return:
        '''
        if self.training:
            # 通道注意力权重 [batch,channel,1,1]
            M = self.SEModule(x)
            # 标记通道 0-1  [batch,channel,1,1]
            T = self.Select_TopK(M)
            # [batch,channel,H,W]
            return self.Apply_Mask(x, T)
        else:
            return x  # eval模式无需drop


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    device = torch.device("cuda:0")
    # [batch,channel,H,W]
    feature = torch.rand(8, 64, 24, 22).to(device)
    channels = feature.shape[1]
    target_drop = TargetDrop(channels=channels).to(device)
    target_drop = target_drop.train()

    result = target_drop(feature)
    print(result.size())
