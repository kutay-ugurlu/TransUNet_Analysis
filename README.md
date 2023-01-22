
# Analysis of TransUNet
This repo holds code for the replicated results and visualizations for the study _TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation_ for the term project of MIN711. 

For the Python environment setup, training and testing, the procedure described in [original repository](https://github.com/Beckschen/TransUNet) is followed. 
The preprocessed data has been used with R50-ViT-B_16 pretrained model in training for 150 epoch gfor batch size of 8. Testing is on the test volume provided. 

For the visualization of the data, [visualization script](https://github.com/kutay-ugurlu/TransUNet_Analysis/blob/main/project_TransUNet/predictions/TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs8_224/visualize_predictions.py) is used. 

![a)Ground truth data, b)Labels, c)Predictions](https://github.com/kutay-ugurlu/TransUNet_Analysis/blob/main/project_TransUNet/predictions/gif_predict/example01.gif)

## Leaving the explanation in the original repo as is below:
# TransUNet
This repo holds code for [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/pdf/2102.04306.pdf)

## Usage

### 1. Download Google pre-trained ViT models
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16, ViT-B_16, ViT-L_16...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz &&
mkdir ../model/vit_checkpoint/imagenet21k &&
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

### 2. Prepare data

Please go to ["./datasets/README.md"](datasets/README.md) for details, or use the [preprocessed data](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd?usp=sharing) for research purposes.

### 3. Environment

Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

### 4. Train/Test

- Run the train script on synapse dataset. The batch size can be reduced to 12 or 6 to save memory (please also decrease the base_lr linearly), and both can reach similar performance.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16
```

- Run the test script on synapse dataset. It supports testing for both 2D images and 3D volumes.

```bash
python test.py --dataset Synapse --vit_name R50-ViT-B_16
```

## Reference
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)

## Citations

```bibtex
@article{chen2021transunet,
  title={TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation},
  author={Chen, Jieneng and Lu, Yongyi and Yu, Qihang and Luo, Xiangde and Adeli, Ehsan and Wang, Yan and Lu, Le and Yuille, Alan L., and Zhou, Yuyin},
  journal={arXiv preprint arXiv:2102.04306},
  year={2021}
}
```
