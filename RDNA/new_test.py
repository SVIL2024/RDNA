import torch
import torch.nn.functional as F
from torch import nn

from data_loader import MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork, D_net
import os
###  这段代码的作用是将评估的图像检测性能指标（image_auc、pixel_auc、image_ap、pixel_ap）以及它们的平均值写入到results.txt 文件中，同时保证每次运行的结果能够区分开来。
def write_results_to_file(run_name, image_auc, pixel_auc, image_ap, pixel_ap):
    if not os.path.exists('./outputs/'):
        os.makedirs('./outputs/')
    #这段代码首先检查当前工作目录下是否存在名为 outputs 的文件夹。如果不存在，则使用 os.makedirs() 函数创建这个文件夹。这个文件夹将用于存储输出结果文件 results.txt。

    fin_str = "img_auc,"+run_name #构建并初始化结果字符串 fin_str，以字符串 "img_auc," 后跟 run_name 变量的值开头。
    for i in image_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(image_auc), 3))
    fin_str += "\n"
    #代码段依次处理 pixel_auc、image_ap 和 pixel_ap，为每个指标生成相应的数据行，并将它们追加到 fin_str
    fin_str += "pixel_auc,"+run_name
    for i in pixel_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(pixel_auc), 3))
    fin_str += "\n"
    fin_str += "img_ap,"+run_name
    for i in image_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(image_ap), 3))
    fin_str += "\n"
    fin_str += "pixel_ap,"+run_name
    for i in pixel_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(pixel_ap), 3))
    fin_str += "\n"
    fin_str += "--------------------------\n"#在所有结果数据记录完成后，添加一条分隔线，用于区分不同批次运行的结果。

    # 打开 ./outputs/results.txt 文件以追加模式（'a+'），将整理好的结果字符串 fin_str 写入文件中
    with open("./outputs/results.txt",'a+') as file:
        file.write(fin_str)


def test(obj_names, mvtec_path, checkpoint_path, base_model_name):
    # 这些列表用来存储每个对象的评估指标，

    obj_ap_pixel_list = []#像素级平均精度
    obj_auroc_pixel_list = []#像素级AUROC
    obj_ap_image_list = []#图像级平均精度
    obj_auroc_image_list = []#图像级AUROC

    for obj_name in obj_names:#循环遍历 obj_names 列表中的每个对象名称，这些名称通常是指不同的测试数据集中的对象或类别。
        img_dim = 256   #设置图像的维度为 256x256 像素。
        run_name = base_model_name+"_"+obj_name+'_' #run_name 构建每个对象的运行名称，最终格式为"base_model_name_obj_name_"

        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)     #重建输入图像。
        model.load_state_dict(torch.load(os.path.join(checkpoint_path,run_name+".pckl"), map_location='cuda:0'))
        #使用 torch.load() 加载预训练的模型参数文件。
        model.cuda()
        model.eval()#将模型移动到 GPU (cuda()) 并设置为评估模式 (eval()).

        '''conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1)
        conv.cuda()

        d_net = D_net()
        d_net.load_state_dict(torch.load(os.path.join(checkpoint_path, run_name + "_d_net.pckl"), map_location='cuda:0'))
        d_net.cuda()
        d_net.eval()'''


        with torch.no_grad():
            model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)      #实例化一个判别子网络模型
            model_seg.load_state_dict(torch.load(os.path.join(checkpoint_path, run_name+"_seg.pckl"), map_location='cuda:0'))
            model_seg.cuda()
            model_seg.eval()

        dataset = MVTecDRAEMTestDataset(mvtec_path + obj_name + "/test/", resize_shape=[img_dim, img_dim])
        dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=0)


        total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))#初始化一个全0数组，存储所有图像的像素得分
        total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))#再初始化一个全0数组，存储所有真实标签的像素得分
        mask_cnt = 0

        anomaly_score_gt = []  # 存储真实标签
        anomaly_score_prediction = []  #   存储模型预测的异常分数

        #用于显示图像和掩码的张量。
        display_images = torch.zeros((16 ,3 ,256 ,256)).cuda()
        display_gt_images = torch.zeros((16 ,3 ,256 ,256)).cuda()
        display_out_masks = torch.zeros((16 ,1 ,256 ,256)).cuda()
        display_in_masks = torch.zeros((16 ,1 ,256 ,256)).cuda()
        cnt_display = 0  #计数显示的图像数量
        display_indices = np.random.randint(len(dataloader), size=(16,))  #用于随机选择显示图像的索引


        for i_batch, sample_batched in enumerate(dataloader):

            gray_batch = sample_batched["image"].cuda()

            is_normal = sample_batched["has_anomaly"].detach().numpy()[0 ,0]
            anomaly_score_gt.append(is_normal)
            true_mask = sample_batched["mask"]
            true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))

            with torch.no_grad():
                gray_rec = model(gray_batch)
                joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

                out_mask = model_seg(joined_in)
               # out_mask_conv = conv(out_mask)
                out_mask_sm = torch.softmax(out_mask, dim=1)

            if i_batch in display_indices:
                t_mask = out_mask_sm[:, 1:, :, :]
                display_images[cnt_display] = gray_rec[0]
                display_gt_images[cnt_display] = gray_batch[0]
                display_out_masks[cnt_display] = t_mask[0]
                display_in_masks[cnt_display] = true_mask[0]
                cnt_display += 1


            out_mask_cv = out_mask_sm[0 ,1 ,: ,:].detach().cpu().numpy()

            out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:], 21, stride=1,
                                                               padding=21 // 2).cpu().detach().numpy()
            image_score = np.max(out_mask_averaged)

            anomaly_score_prediction.append(image_score)

            flat_true_mask = true_mask_cv.flatten()
            flat_out_mask = out_mask_cv.flatten()
            total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
            total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
            mask_cnt += 1



        anomaly_score_prediction = np.array(anomaly_score_prediction)#被转换为NumPy数组，代表预测的异常分数
        anomaly_score_gt = np.array(anomaly_score_gt)  #被转换为NumPy数组，分别代表实际的异常标签
        #roc_auc_score 和 average_precision_score 分别计算了这两个数组之间的ROC AUC和平均精度。
        auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
        ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)

        #total_gt_pixel_scores 和 total_pixel_scores 被截取并转换为np.uint8类型。它们分别代表总体像素级别的真实异常分数和预测异常分数。
        # 再次计算了它们之间的ROC AUC和平均精度，并将结果分别添加到 obj_auroc_pixel_list 和 obj_ap_pixel_list 中。
        total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
        total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
        total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]

        auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
        ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)

        obj_ap_pixel_list.append(ap_pixel)
        obj_auroc_pixel_list.append(auroc_pixel)
        obj_auroc_image_list.append(auroc)
        obj_ap_image_list.append(ap)



        #打印每个对象（或图像）的名称以及计算出的AUC和AP值（分为图像级别和像素级别）。
        print(obj_name)
        print("AUC Image:  " +str(auroc))
        print("AP Image:  " +str(ap))
        print("AUC Pixel:  " +str(auroc_pixel))
        print("AP Pixel:  " +str(ap_pixel))
        print("==============================")

    #打印整个运行过程中所有对象（或图像）的平均AUC和AP值（分为图像级别和像素级别的平均值）。
    print(run_name)
    print("AUC Image mean:  " + str(np.mean(obj_auroc_image_list)))
    print("AP Image mean:  " + str(np.mean(obj_ap_image_list)))
    print("AUC Pixel mean:  " + str(np.mean(obj_auroc_pixel_list)))
    print("AP Pixel mean:  " + str(np.mean(obj_ap_pixel_list)))

    write_results_to_file(run_name, obj_auroc_image_list, obj_auroc_pixel_list, obj_ap_image_list, obj_ap_pixel_list)
    #调用一个函数 write_results_to_file，将所有的AUC和AP值写入文件，以及可能的其他结果。



if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', action='store', type=int, required=True)
    parser.add_argument('--base_model_name', action='store', type=str, required=True)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)

    args = parser.parse_args()

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
    #使用 torch.cuda.device 上下文管理器，将GPU设备切换到 args.gpu_id 指定的GPU上。这确保了接下来的计算都在指定的GPU上执行
    #调用名为 test 的函数，传入 obj_list、args.data_path、args.checkpoint_path 和 args.base_model_name 作为参数。
    # 这些参数包含了从命令行传入的用户指定的信息，用于在特定的GPU上运行模型测试或评估。
    with torch.cuda.device(args.gpu_id):
        test(obj_list,args.data_path, args.checkpoint_path, args.base_model_name)
