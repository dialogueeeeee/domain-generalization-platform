# domain-generalization-platform
- 基于 **DASSL** 的跨域分类任务模板。
- 本项目基于 **DASSL(v0.5.0)** [官方项目](https://github.com/KaiyangZhou/Dassl.pytorch)实现，采用继承设计和注册机制，使得开发人员和研究人员能够快速完成本地跨域分类任务的项目搭建和部署，同时也能够更方便的集成到自己的项目中。

## Installation & Prerequisite
- The installation of DASSL based on pytorch, See the [official tutorial](https://github.com/KaiyangZhou/Dassl.pytorch#installation) for details;
- It requires the following packages:
```
Python 3.7
PyTorch >= 1.6
torchvision 0.7.0
CUDA 10.1
DASSL >= v0.5.0
flake8==3.7.9
yapf==0.29.0
isort==4.3.21
yacs
gdown
tb-nightly
future
scipy
scikit-learn
tqdm
```

## Usage
### Data preparation
Download and prepare data following https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/DATASETS.md. The PACS dataset structure as example looks like below:
```
pacs/
|–– images/
|–– splits/
```
### Training and testing
- You can run training and testing directly by the linux command (PACS dataset as example) like below:
```
python train.py --root ${DATASET_PATH} --trainer ${TRAINER} --source-domains art_painting --target-domains cartoon photo sketch --dataset-config-file ${DASSL_PATH}/configs/datasets/dg/pacs.yaml --config-file ${DASSL_PATH}/configs/trainers/dg/vanilla/pacs.yaml --output-dir ${OUTPUT_DIR} MODEL.BACKBONE.NAME resnet18
```
- Or you can ``cd script`` and run ``bash res18.sh`` to train and test like below: 
``` shell
#!/bin/bash

cd ..

DATA=~/dataset/
DASSL=~/Dassl.pytorch

D1=art_painting
D2=cartoon
D3=photo
D4=sketch

################### leave one domain out setting
DATASET=pacs
TRAINER=Vanilla
NET=resnet18

for SEED in $(seq 1 2)
do
    for SETUP in $(seq 1 4)
    do
        if [ ${SETUP} == 1 ]; then
            S1=${D2}
            S2=${D3}
            S3=${D4}
            T=${D1}
        elif [ ${SETUP} == 2 ]; then
            S1=${D1}
            S2=${D3}
            S3=${D4}
            T=${D2}
        elif [ ${SETUP} == 3 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D4}
            T=${D3}
        elif [ ${SETUP} == 4 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D3}
            T=${D4}
        fi
        
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --source-domains ${S1} ${S2} ${S3}  \
        --target-domains ${T} \
        --dataset-config-file ${DASSL}/configs/datasets/dg/${DATASET}.yaml \
        --config-file ${DASSL}/configs/trainers/dg/vanilla/${DATASET}.yaml \
        --output-dir baseline/${DATASET}/${TRAINER}/${NET}_nodetach/${T}/seed${SEED} \
        MODEL.BACKBONE.NAME ${NET}
    done
done

```

## VSCode debugger launch config
``` json
{
    "name": "Python: train.py",
    "type": "python",
    "request": "launch",
    "program": "${workspaceRoot}/train.py",
    "console": "integratedTerminal",
    "justMyCode": false,
    "args": [
        "--root","~/dataset/",
        "--seed", "1",
        "--trainer", "Vanilla_freezen",
        "--source-domains", "art_painting",
        "--target-domains", "cartoon", "photo", "sketch",
        "--dataset-config-file", "configs/datasets/shape_task1_pacs.yaml",
        "--config-file", "~/Dassl.pytorch/configs/trainers/dg/vanilla/pacs.yaml",       
        "--output-dir", "shapetask1/pacs/Vanilla_singles/resnet18_nodetach/random/art_painting/seed1",
        "MODEL.BACKBONE.NAME", "resnet18" 
    ]
}
```



