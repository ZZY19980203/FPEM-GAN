# FPEM-GAN

![Image text](https://github.com/ZZY19980203/FPEM-GAN/blob/main/img_folder/framework.jpg)


## Prerequisites

- Python 3.8
- PyTorch 1.12.0 + cu113 
- NVIDIA GPU + CUDA cuDNN

## Installation

- Clone this repo:

  ```
  git clone https://github.com/ZZY19980203/FPEM-GAN.git
  ```

- Install PyTorch and dependencies from [http://pytorch.org](http://pytorch.org/)

- Install python requirements:

  ```
  pip install -r requirements.txt
  ```

## Started



###  generate reliable pseudo-labels

1、First, we need to generate reliable pseudo-labels by teacher model, the generation model is based on DDPM, and we need to download the corresponding pre-trained model.

```
cd teacher_model
pip install --upgrade gdown && bash ./download.sh
```

2、We need to crop the original logging image. The logging image is first cropped into multiple images of equal length and width, and then resized to 256 x 256.

3、running code.  In the .yam file inside the confs, you need to set the location of the original logging image and mask, and you also need to set the location where the image will be saved.

```
python test.py --conf_path confs/test_inet256_thin.yml
```



### Training and testing student models

#### Trainting

1、Training the student model also requires a configuration .yml file, which we save in the checkpoint file.

2、Our FPEM-GAN is trained in the same way as EdgeConnect in three stages: 1) training the edge model; 2) training the inpainting model; 3) training the joint model.

```
python train.py --model [stage] --checkpoints [path to checkpoints]
```

#### Testing

You can test the model on all three stages: 1) edge model, 2) inpaint model and 3) joint model. In each case, you need to provide an input image (image with a mask) and a grayscale mask file. Please make sure that the mask file covers the entire mask region in the input image. To test the model:

```
python test.py --model [stage] --checkpoints [path to checkpoints] --input [path to input directory or file] --mask [path to masks directory or mask file] --output [path to the output directory]
```

## Example
Below is a display of some of our results, the original images and results can be viewed in the file ``` /img_folder/example/ ```.

![Image text](https://github.com/ZZY19980203/FPEM-GAN/blob/main/img_folder/exmple.jpg)

## Acknowledgements

We would like to thank [edge-connect](https://github.com/knazeri/edge-connect), [RePaint](https://github.com/andreas128/RePaint) and [guided-diffuion](https://github.com/openai/guided-diffusion.git).

If we missed a contribution, please contact us.
