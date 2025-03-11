#!/bin/bash

# Meniscus Axial Training
echo "Starting Meniscus Axial Training..."
python 'src/train_baseline.py' \
    --prefix_name 'base' \
    -t 'meniscus' \
    -p 'axial' \
    --epochs 200 \
    --augment_prob 0.40

echo "Completed Meniscus Axial Training"
echo "---------------------------------"

# Meniscus Coronal Training
echo "Starting Meniscus Coronal Training..."
python 'src/train_baseline.py' \
    --prefix_name 'base' \
    -t 'meniscus' \
    -p 'coronal' \
    --epochs 200 \
    --augment_prob 0.40

echo "Completed Meniscus Coronal Training"
echo "---------------------------------"

# Meniscus Sagittal Training
echo "Starting Meniscus Sagittal Training..."
python 'src/train_baseline.py' \
    --prefix_name 'base' \
    -t 'meniscus' \
    -p 'sagittal' \
    --epochs 100 \
    --augment_prob 0.90

echo "Completed Meniscus Sagittal Training"
echo "---------------------------------"

python src/combine.py -t 'meniscus'
# python src/predict.py

echo "All training runs completed!"