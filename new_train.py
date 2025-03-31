import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from data_loader import MVTecDRAEMTrainDataset
from torch.utils.data import DataLoader
from torch import optim, nn
from tensorboard_visualizer import TensorboardVisualizer
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork, Model_D_net, D_net
from loss import FocalLoss, SSIM
import os

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train_on_device(obj_names, args):

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    for obj_name in obj_names:
        print("All training obj_names:", obj_names)

        print("The training obj_name: " + str(obj_name))
        run_name = 'DRAEM_test_'+str(args.lr)+'_'+str(args.epochs)+'_bs'+str(args.bs)+"_"+obj_name+'_'
        #run_name 的最终值将是一个由固定前缀、学习率、轮数、批量大小和目标名称组成的字符串，用于唯一标识或描述某个运行的特征。例如，可能的
        # run_name 可能是类似于 'DRAEM_test_0.001_50_bs32_object1_' 的字符串，表示学习率为0.001，训练时期为50，批量大小为32，并且目标名称为 'object1'
        #创建Tensorboard日志的对象，它的log_dir参数指定了日志文件的路径，这里使用了os.path.join将日志路径和run_name结合起来。
        visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, run_name+"/"))

        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model.cuda()
        model.apply(weights_init)#应用了一个叫做weights_init的函数来初始化模型的权重。

        conv = nn.Conv2d(in_channels = 2,out_channels=1,  kernel_size=1,stride = 1)
        conv.cuda()

        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        model_seg.cuda()
        model_seg.apply(weights_init)

        model_d_net = Model_D_net()
        model_d_net.cuda()
        model_d_net.apply(weights_init)  # 应用了一个叫做weights_init的函数来初始化模型的权重。


        model_seg_d_net = D_net()
        model_seg_d_net.cuda()
        model_seg_d_net.apply(weights_init)

        optimizer = torch.optim.Adam([
            {"params": model.parameters(), "lr": args.lr},
            {"params": model_seg.parameters(), "lr": args.lr}])
        optimizer_d_net = torch.optim.Adam(model_d_net.parameters(), args.lr)
        optimizer_seg_d_net = torch.optim.Adam(model_seg_d_net.parameters(),args.lr)

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)
        scheduler_d_net = optim.lr_scheduler.MultiStepLR(optimizer_d_net, [args.epochs * 0.8, args.epochs * 0.9],gamma=0.2, last_epoch=-1)

        scheduler_seg_d_net = optim.lr_scheduler.MultiStepLR(optimizer_seg_d_net, [args.epochs * 0.8, args.epochs * 0.9], gamma=0.2,last_epoch=-1)

        loss_l2 = torch.nn.modules.loss.MSELoss()
        loss_ssim = SSIM()
        loss_focal = FocalLoss()

        data = MVTecDRAEMTrainDataset(args.data_path + obj_name + "/train/good/", args.anomaly_source_path, resize_shape=[256, 256])
        data_path = args.data_path + obj_name + "/train/good/"
        dataloader = DataLoader(data, batch_size=args.bs,
                                shuffle=True, num_workers=0)

        n_iter = 0
        for epoch in range(args.epochs):
            #print("Epoch: "+str(epoch))
            for i_batch, sample_batched in enumerate(dataloader):#这是一个内部循环，用于遍历数据加载器 dataloader 中的每个批次数据
                #从数据批次中提取出灰度图像 gray_batch、增强后的灰度图像 aug_gray_batch 和异常掩码 anomaly_mask，
                # 并将它们转移到 GPU 上进行加速计算（.cuda()）。
                gray_batch = sample_batched["image"].cuda()
                aug_gray_batch = sample_batched["augmented_image"].cuda()
                anomaly_mask = sample_batched["anomaly_mask"].cuda()

                gray_rec = model(aug_gray_batch)#使用模型 model 对增强后的灰度图像 aug_gray_batch 进行重建，生成 gray_rec。

                d_fake_output1 = model_d_net(gray_rec)  # 将重构的图像给判别器
                d_real_output1 = model_d_net(gray_batch)  # 将原始图像给判别器

                model_d_net_fake = torch.ones([d_fake_output1.shape[0], 1], dtype=torch.float32).cuda()
                model_d_net_valid = torch.zeros([d_real_output1.shape[0], 1], dtype=torch.float32).cuda()

                model_d_fake_loss = F.binary_cross_entropy(torch.squeeze(d_fake_output1),torch.squeeze(model_d_net_fake))
                model_d_real_loss = F.binary_cross_entropy(torch.squeeze(d_real_output1),torch.squeeze(model_d_net_valid))
                model_d_net_sum_loss = 0.5 * (model_d_fake_loss + model_d_real_loss)

                optimizer_d_net.zero_grad()
                model_d_net_sum_loss.backward(retain_graph=True)
                optimizer_d_net.step()

                joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)
                #将重建的灰度图像 gray_rec 和增强后的灰度图像 aug_gray_batch 沿着通道维度（dim=1）拼接在一起，形成一个新的输入张量 joined_in。

                out_mask = model_seg(joined_in)#使用分割模型 model_seg 对拼接后的输入 joined_in 进行分割，得到输出的掩码 out_mask。
                #对分割输出 out_mask 进行 softmax 操作，将其转换为概率分布 out_mask_sm，在通道维度（dim=1）上进行操作。
                #print(out_mask.shape)   torch.Size([4, 2, 256, 256])
                out_mask_conv = conv(out_mask)

                out_mask_sm = torch.softmax(out_mask, dim=1)
                #print(out_mask_sm.shape)   torch.Size([4, 2, 256, 256])

                l2_loss = loss_l2(gray_rec,gray_batch)
                ssim_loss = loss_ssim(gray_rec, gray_batch)
                segment_loss = loss_focal(out_mask_sm, anomaly_mask)

                loss = l2_loss + ssim_loss + segment_loss


                d_fake_output = model_seg_d_net(out_mask_conv)
                d_real_output = model_seg_d_net(anomaly_mask)

                fake = torch.ones([d_fake_output.shape[0], 1], dtype=torch.float32).cuda()
                valid = torch.zeros([d_real_output.shape[0], 1], dtype=torch.float32).cuda()


                # d_fake_loss = F.binary_cross_entropy(torch.squeeze(d_fake_output), fake)
                d_fake_loss = F.binary_cross_entropy(torch.squeeze(d_fake_output), torch.squeeze(fake))
                # print("fake:",fake.size())
                # print("torch.squeeze(d_fake_output):",torch.squeeze(d_fake_output).size())
                d_real_loss = F.binary_cross_entropy(torch.squeeze(d_real_output), torch.squeeze(valid))
                d_sum_loss = 0.5 * (d_fake_loss + d_real_loss)

                optimizer_seg_d_net.zero_grad()
                d_sum_loss.backward(retain_graph=True)
                optimizer_seg_d_net.step()

                #执行优化器的操作：清除之前的梯度信息、计算当前损失的梯度并执行优化步骤。
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #如果设置了可视化标志 args.visualize 并且当前迭代次数 n_iter 是 200 的倍数，则可视化各个损失函数的变化。
                if args.visualize and n_iter % 200 == 0:
                    visualizer.plot_loss(l2_loss, n_iter, loss_name='l2_loss')
                    visualizer.plot_loss(ssim_loss, n_iter, loss_name='ssim_loss')
                    visualizer.plot_loss(segment_loss, n_iter, loss_name='segment_loss')

                #如果设置了可视化标志 args.visualize 并且当前迭代次数 n_iter 是 400 的倍数，
                # 则可视化增强后的灰度图像、重建目标图像、重建输出图像、目标掩码图像和输出掩码图像。
                if args.visualize and n_iter % 400 == 0:
                    t_mask = out_mask_sm[:, 1:, :, :]
                    visualizer.visualize_image_batch(aug_gray_batch, n_iter, image_name='batch_augmented')
                    visualizer.visualize_image_batch(gray_batch, n_iter, image_name='batch_recon_target')
                    visualizer.visualize_image_batch(gray_rec, n_iter, image_name='batch_recon_out')
                    visualizer.visualize_image_batch(anomaly_mask, n_iter, image_name='mask_target')
                    visualizer.visualize_image_batch(t_mask, n_iter, image_name='mask_out')


                n_iter +=1#更新迭代计数器 n_iter。
            scheduler_d_net.step()
            scheduler_seg_d_net.step()
            scheduler.step()#调度器执行步长调整，可能用于调整学习率等
            #在每个 epoch 结束时，保存重构模型 model 和分割模型 model_seg 的状态字典（即它们的权重参数）到指定路径下的文件中，
            # 文件名包含当前运行名称 run_name。
            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name+".pckl"))
            torch.save(model_seg.state_dict(), os.path.join(args.checkpoint_path, run_name+"_seg.pckl"))
            torch.save(model_d_net.state_dict(), os.path.join(args.checkpoint_path, run_name + "_model_d_net.pckl"))
            torch.save(model_seg_d_net.state_dict(),os.path.join(args.checkpoint_path, run_name + "_model_seg__d_net.pckl"))
            print(f'Epoch [{epoch + 1}/{args.epochs}],  Loss: {loss:.4f},  model_d_net_sum_loss:{model_d_net_sum_loss:.4f}, d_sum_Loss: {d_sum_loss:.4f}')
        print("==============当前类别训练结束=============")

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, default=1, required=True)
    parser.add_argument('--bs', action='store', type=int, default=4, required=True)
    parser.add_argument('--lr', action='store', type=float, default=0.0001, required=True)
    parser.add_argument('--epochs', action='store', type=int, default=100, required=True)
    parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
    parser.add_argument('--data_path', action='store', type=str, default='./datasets/mvtec/mvtec_anomaly_detection/', required=True)
    parser.add_argument('--anomaly_source_path', action='store', type=str, default=' ./datasets/dtd/images/', required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, default=' ./checkpoints/', required=True)
    parser.add_argument('--log_path', action='store', type=str, default='./logs/', required=True)
    parser.add_argument('--visualize', action='store_true')

    args = parser.parse_args()

    obj_batch = [['capsule'],
                 ['bottle'],
                 ['carpet'],
                 ['leather'],
                 ['pill'],
                 ['transistor'],
                 ['tile'],
                 ['cable'],
                 ['zipper'],
                 ['toothbrush'],
                 ['metal_nut'],
                 ['hazelnut'],
                 ['screw'],
                 ['grid'],
                 ['wood']
                 ]

    if int(args.obj_id) == -1:
        #这是一个包含字符串元素的列表，每个字符串代表一种物体的类别，
        # 例如'capsule'（胶囊）、'bottle'（瓶子）、'carpet'（地毯）等等。这些是默认的物体类别。
        obj_list = ['capsule',
                     'bottle',
                     'carpet',
                     'leather',
                     'pill',
                     'transistor',
                     'tile',
                     'cable',
                     'zipper',
                     'toothbrush',
                     'metal_nut',
                     'hazelnut',
                     'screw',
                     'grid',
                     'wood'
                     ]
        picked_classes = obj_list#如果args.obj_id为-1，则picked_classes将被设置为obj_list，即所有预定义的对象类别。
    else:
        picked_classes = obj_batch[int(args.obj_id)]#如果args.obj_id不等于-1，从obj_batch列表中根据args.obj_id选择一个特定的对象类别列表。

    with torch.cuda.device(args.gpu_id):#这是一个使用PyTorch进行GPU加速的上下文管理器。args.gpu_id是指定的GPU设备ID。
        train_on_device(picked_classes, args)
        #调用了一个函数train_on_device，传递了两个参数：picked_classes（选择的对象类别列表）和args（参数集合）。
        # 这个函数可能是用来在指定的GPU设备上训练模型的函数。
