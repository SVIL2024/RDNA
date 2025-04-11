# Anomaly Detection in Industrial Images via Reconstruction Discriminant Networks with Adversarial Learning

## Authors

Xiaoru Liu
, Shifeng Li 
, Yan Cheng
, Xi Luo
Corresponding author: Shifeng Li limax_2008@outlook.com
## Description
In the field of industrial image anomaly detection, traditional manual inspection methods are inefficient and ill-suited for large-scale applications due to the scarcity and irregularity of anomalous samples. Consequently, unsupervised learning approaches based on reconstruction have gained prominence. However, these methods often struggle to capture all types of anomalies, particularly small ones. To address this, we propose Reconstruction Discriminant Networks with Adversarial Training (RDNA) for anomaly detection in industrial images. RDNA comprises reconstruction and discriminant modules, each equipped with a reconstruction network and a discriminator. Through adversarial training, the reconstruction module generates images that closely align with the normal data distribution, while the discriminant module produces accurate anomaly segmentation maps, improving anomaly localization. Experiments on the MVTec AD dataset demonstrate the effectiveness of our approach, achieving image-level AUROC, pixel-level AUROC, and AP scores of 98.5%, 97.6%, and 72.8%, respectively. 

## Datasets
To train on the MVtec Anomaly Detection dataset [download](https://www.mvtec.com/company/research/datasets/mvtec-ad)
the data and extract it. The [Describable Textures dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/) was used as the anomaly source 
image set in most of the experiments in the paper. You can run the **download_dataset.sh** script from the project directory
to download the MVTec and the DTD datasets to the **datasets** folder in the project directory:
```
./scripts/download_dataset.sh
```


## Training
Pass the folder containing the training dataset to the **new_train.py** script as the --data_path argument and the
folder locating the anomaly source images as the --anomaly_source_path argument. 
The training script also requires the batch size (--bs), learning rate (--lr), epochs (--epochs), path to store checkpoints
(--checkpoint_path) and path to store logs (--log_path).
Example:

```
python new_train.py --gpu_id 0 --obj_id -1 --lr 0.0001 --bs 4 --epochs 8 --data_path ./datasets/mvtec/mvtec_anomaly_detection/ --anomaly_source_path ./datasets/dtd/images/ --checkpoint_path ./checkpoints/ --log_path ./logs/
```

The conda environement used in the project is decsribed in **requirements.txt**.

## Pretrained models

The pretrained models achieve a 98.5 image-level ROC AUC, 97.6 pixel-wise ROC AUC and a 72.8 pixel-wise AP.


## Evaluating
The test script requires the --gpu_id arguments, the name of the checkpoint files (--base_model_name) for trained models, the 
location of the MVTec anomaly detection dataset (--data_path) and the folder where the checkpoint files are located (--checkpoint_path)
with pretrained models can be run with:

```
python new_test.py --gpu_id 0 --base_model_name "DRAEM_test_0.0001_800_bs4" --data_path ./datasets/mvtec/mvtec_anomaly_detection/ --checkpoint_path ./checkpoints_new_train_2/
```
