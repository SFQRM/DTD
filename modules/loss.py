class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        # 获取x对应的概率值
        x = self.generator(x)
        # 计算loss
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        # 反向bp计算，获取梯度值
        loss.backward()
        if self.opt is not None:
            self.opt.step()  # 梯度更新到模型
            self.opt.optimizer.zero_grad()  # 清零梯度值，准备用于下一次梯度计算
        return loss.data[0] * norm
