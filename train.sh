#!/bin/bash

data_path=makeup/red/red
dataset=RED

python train.py --data_path=$data_path --dataset=$dataset
