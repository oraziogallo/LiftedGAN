fting 2D StyleGAN for 3D-Aware Face Generation
## Yichun Shi, Divyansh Aggarwal, Anil K Jain

<a href="https://arxiv.org/abs/2011.13126"><img src="https://img.shields.io/badge/arXiv-2008.00951-b31b1b.svg"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>

Oficial implementation of paper "Lifting 2D StyleGAN for 3D-Aware Face Generation".

## Requirements
You can create the conda environment by using `conda env create -f environment.yml`

## Training
### Training from pre-trained StyleGAN2
Download our pre-trained StyleGAN and face embedding network from [here](https://drive.google.com/file/d/1qVoWu_fps17iTzYptwuN3ptgYeCpIl2e/view?usp=sharing). Unzip them into the `pretrained/` folder. Then you can start training by:
```sh
python tools/train.py config/ffhq_256.py
```
Note that you do **not** need an image dataset here becuase we simply lift the StyleGAN2 using its own generated images.

### Training from custom data
We use a re-cropped version of FFHQ to fit the style of our face embedding network. You can find this dataset [here](https://drive.google.com/file/d/1pLHzbZS52XGyejubv5tT0CqhpsocaYuD/view?usp=sharing). The cats dataset can be found [here](https://drive.google.com/file/d/1soEXvvLV0uhasg9GlVhH5YW_9FsAmb3d/view?usp=sharing).
To train a StyleGAN2 from you own dataset, check the content under `stylegan2-pytorch` folder. After training a StyleGAN2, you can lift it by using our training code. Note that our method might not apply to other kind of images, if they are very different from human faces.

## Testing
### Sampling random faces
You can generate random samples from a lifted gan by running:
```sh
python tools/generate_images.py /path/to/the/checkpoint --output_dir results/
```
Make sure the checkpoint file and its `config.py` file are under the same folder.

### Generating controlled faces
You can generate GIF animations of rotated faces by running:
```sh
python tools/generate_poses.py /path/to/the/checkpoint --output_dir results/ --type yaw
```
Similarly, you can generate faces with different light directions:
```sh
python tools/generate_lighting.py /path/to/the/checkpoint --output_dir results/
```

### Testing FID
We use the code from rosinality's stylegan2-pytorch to compute FID. To compute the FID, you first need to compute the statistics of real images:
```sh
python utils/calc_inception.py /path/to/the/dataset/lmdb
```
You might skip this step if you are using our pre-calculated statistics file ([link](https://drive.google.com/file/d/1qVoWu_fps17iTzYptwuN3ptgYeCpIl2e/view?usp=sharing)). Then, to test the FID, you can run:
```sh
python tools/test_fid.py /path/to/the/checkpoint --inception /path/to/the/inception/file
```

## Ackowledgment
Part of this code is based on Wu's [Unsup3D](https://github.com/elliottwu/unsup3d) and Rosinality's [StyleGAN2-Pytorch](https://github.com/rosinality/stylegan2-pytorch).


