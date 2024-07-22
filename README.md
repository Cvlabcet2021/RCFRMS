
## Training
- Download the [Datasets](Datasets/README.md)

- Train the model with default arguments by running

```
python train.py
```


## Evaluation

1. Download the [model](https://drive.google.com/file/d/1O3WEJbcat7eTY6doXWeorAbQ1l_WmMnM/view?usp=sharing) and place it in `./pretrained_models/`

2. Download test datasets (test) from [here](https://drive.google.com/drive/folders/1300oiuof6OIM4K6_XCqBYzp4hVNVSplO?usp=drive_link) and place them in `./Datasets/Haze_Datasets/test/`

3. Run
```
python test.py
```

#### To reproduce PSNR/SSIM scores of the paper, run
```
evaluate_PSNR_SSIM.m 
```
