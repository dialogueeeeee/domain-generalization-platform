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
        --output-dir baseline/${DATASET}/${TRAINER}/${NET}_nodetach/${MIX}/${T}/seed${SEED} \
        MODEL.BACKBONE.NAME ${NET}
    done
done

############## single source generalization setting.
DATASET=pacs
TRAINER=Vanilla
NET=resnet18

for SEED in $(seq 3 4)
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
        --source-domains ${T}  \
        --target-domains ${S1} ${S2} ${S3} \
        --dataset-config-file ${DASSL}/configs/datasets/dg/${DATASET}.yaml \
        --config-file ${DASSL}/configs/trainers/dg/vanilla/${DATASET}.yaml \
        --output-dir baseline/${DATASET}/${TRAINER}_singles/${NET}_nodetach/${MIX}/${T}/seed${SEED} \
        MODEL.BACKBONE.NAME ${NET}

    done
done
