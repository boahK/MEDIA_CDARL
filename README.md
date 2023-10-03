# MEDIA_CDARL
Official repository for "C-DARL: Contrastive diffusion adversarial representation learning for label-free blood vessel segmentation"
[[arXiv](https://arxiv.org/abs/2308.00193)]

![Image of The Proposed method](figs/method.png)

## Requirements
  * OS : Ubuntu
  * Python >= 3.9
  * PyTorch >= 1.12.1

## Data
In our experiments, we used the publicly available XCAD dataset. Please refer to our main paper.

## Training

To train our model, run this command:

```train
python3 main.py -p train -c config/train.json
```

## Test

To test the trained our model, run:

```eval
python3 main.py -p test -c config/test.json
```

## Pre-trained Models

You can download our pre-trained model of the XCAD dataset [here](https://drive.google.com/).
Then, you can test the model by saving the pre-trained weights in the directory ./experiments/pretrained_model.
To briefly test our method given the pre-trained model, we provided the toy example in the directory './data/'.

## Citations

```
@article{kim2023c,
  title={C-DARL: Contrastive diffusion adversarial representation learning for label-free blood vessel segmentation},
  author={Kim, Boah and Oh, Yujin and Wood, Bradford J and Summers, Ronald M and Ye, Jong Chul},
  journal={arXiv preprint arXiv:2308.00193},
  year={2023}
}
```

