import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import os
# Writer will output to ./runs/ directory by default

class TensorboardVisualizer():#将数据可视化到TensorBoard的类TensorboardVisualizer
    #初始化方法，指定了日志存储的目录路径log_dir，如果该目录不存在则创建。
    # 创建了一个SummaryWriter对象self.writer，用于写入日志到指定的目录。
    def __init__(self,log_dir='./logs/'):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)

    #接受一个图像批次image_batch、迭代次数n_iter和图像名称image_name作为参数。
    #使用torchvision.utils.make_grid将图像批次转换为网格形式。
    # 使用self.writer.add_image将网格图像添加到TensorBoard日志中，可以在TensorBoard中查看这些图像。

    def visualize_image_batch(self,image_batch,n_iter,image_name='Image_batch'):
        grid = torchvision.utils.make_grid(image_batch)
        self.writer.add_image(image_name,grid,n_iter)

    #接受损失值loss_val、迭代次数n_iter和损失名称loss_name作为参数。
    # 使用self.writer.add_scalar将损失值添加到TensorBoard日志中，可以在TensorBoard中查看损失值随着迭代次数的变化情况。

    def plot_loss(self, loss_val, n_iter, loss_name='loss'):
        self.writer.add_scalar(loss_name, loss_val, n_iter)

