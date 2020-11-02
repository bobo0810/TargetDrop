# [TargetDrop](https://arxiv.org/abs/2010.10716)
#### 收录到[PytorchNetHub](https://github.com/bobo0810/PytorchNetHub)

## 说明
- 非官方
- 基于注意力机制的Drop正则化方法

## 环境

| python版本 | pytorch版本 | 系统   |
|------------|-------------|--------|
| 3.6        | 1.6.0       | Ubuntu |

## 使用

```python
import TargetDrop
class MyNet():
    def __init__(self,channels):
        ...
        self.target_drop=TargetDrop(channels)
    def forward(self, x):
        ...
        x=self.target_drop(x)
```

## 算法框架
![](https://github.com/bobo0810/TargetDrop/blob/main/imgs/TargetDrop.png)