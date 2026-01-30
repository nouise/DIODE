#!/usr/bin/env bash

# --------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Official PyTorch implementation of WACV2021 paper:
# Data-Free Knowledge Distillation for Object Detection
# A Chawla, H Yin, P Molchanov, J Alvarez
# --------------------------------------------------------

##############################################################################
# This script runs main_yolo.py multiple times to generate multiple batches
# of inverted images at resolution 512.
#
# Characteristics:
# - Single GPU
# - Sequential execution (one python finishes before next starts)
# - Each batch writes its own run.log
#
# Usage:
#   bash LINE_looped_runner_yolo_res512_bs16.sh
##############################################################################

set -e  # exit immediately if a command fails

############################
# User configuration
############################
GPU_ID=7
TOTAL_IMAGES=168
BATCHSIZE=8
RESOLUTION=512

MANIFEST="/data1/home/ypliu/merged_coco/manifest.txt"
WEIGHTS="/data1/home/ypliu/DIODE/knowledge_distillation/yolov3-master/runs/exp/weights/best.pt"

ROOT_OUTDIR="/data1/home/ypliu/merged_coco_syn"
mkdir -p "${ROOT_OUTDIR}"

export CUDA_VISIBLE_DEVICES=${GPU_ID}
echo "[INFO] Running on GPU ${CUDA_VISIBLE_DEVICES}"

############################
# Loop control
############################
STARTLINE=161
ENDLINE=$((STARTLINE + TOTAL_IMAGES))
CURLINE=${STARTLINE}

while [ ${CURLINE} -lt ${ENDLINE} ]
do
    CURENDLINE=$((CURLINE + BATCHSIZE))
    if [ ${CURENDLINE} -gt ${ENDLINE} ]; then
        CURENDLINE=${ENDLINE}
        CUR_BS=$((CURENDLINE - CURLINE))
    else
        CUR_BS=${BATCHSIZE}
    fi

    echo "--------------------------------------------------"
    echo "[INFO] Lines: [${CURLINE}, ${CURENDLINE}) | bs=${CUR_BS} | res=${RESOLUTION}"
    echo "--------------------------------------------------"

    SUBSETFILE="subset_${CURLINE}_${CURENDLINE}_bs${CUR_BS}_res${RESOLUTION}.txt"
    OUTDIR="${ROOT_OUTDIR}/subset_${CURLINE}_${CURENDLINE}_bs${CUR_BS}_res${RESOLUTION}"

    ############################
    # Extract subset manifest
    ############################
    cat "${MANIFEST}" \
        | head -n $((CURENDLINE - 1)) \
        | tail -n ${CUR_BS} \
        > "${SUBSETFILE}"

    NLINES=$(wc -l < "${SUBSETFILE}")
    if [ "${NLINES}" -ne "${CUR_BS}" ]; then
        echo "[WARN] bs=${CUR_BS} but nlines=${NLINES}"
    fi

    ############################
    # Prepare output directory
    ############################
    mkdir -p "${OUTDIR}"

    ############################
    # Run Deep Inversion (res=512)
    # NOTE:
    # - NO nohup here
    # - NO background (&)
    # - bash WILL WAIT until python finishes
    ############################
    python -u main_yolo.py \
        --resolution=${RESOLUTION} \
        --bs=${CUR_BS} \
        --jitter=40 \
        --do_flip \
        --rand_brightness \
        --rand_contrast \
        --random_erase \
        --path="${OUTDIR}" \
        --train_txt_path="${SUBSETFILE}" \
        --iterations=1500 \
        --r_feature=0.1 \
        --p_norm=2 \
        --alpha-mean=1.0 \
        --alpha-var=1.0 \
        --num_layers=51 \
        --first_bn_coef=0.0 \
        --main_loss_multiplier=1.0 \
        --alpha_img_stats=0.0 \
        --tv_l1=75.0 \
        --tv_l2=0.0 \
        --lr=0.002 \
        --min_lr=0.0005 \
        --wd=0.0 \
        --save_every=100 \
        --seeds="0,0,23456" \
        --display_every=100 \
        --init_scale=1.0 \
        --init_bias=0.0 \
        --nms_conf_thres=0.1 \
        --alpha-ssim=0.0 \
        --save-coco \
        --real_mixin_alpha=1.0 \
        > "${OUTDIR}/running.log" 2>&1

    ############################
    # Cleanup (match official style)
    ############################
    rm -f "${OUTDIR}/chkpt.pt"
    rm -f "${OUTDIR}/iteration_targets"*
    rm -f "${OUTDIR}/tracker.data"

    mv "${SUBSETFILE}" "${OUTDIR}/"

    cat "${OUTDIR}/losses.log" | grep "Initialization" || true
    cat "${OUTDIR}/losses.log" | grep "Verifier RealImage" || true
    cat "${OUTDIR}/losses.log" | grep "Verifier GeneratedImage" | tail -n 1 || true

    ############################
    # Advance loop
    ############################
    CURLINE=${CURENDLINE}
done

echo "[INFO] Finished generating ${TOTAL_IMAGES} images at resolution ${RESOLUTION}"
