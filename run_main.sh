#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main.py \
  --content_names cat backpack berry clock \
  --style_names oilpainting watercolorpainting flat flat \