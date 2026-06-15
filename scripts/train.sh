#!/bin/bash

echo "Start SBERT model training..."
python model.py \
    --data_dir ./datasets \
    --epochs 12 \
    --batch_size 64
echo "Training completed!"