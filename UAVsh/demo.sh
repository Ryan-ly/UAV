#!/bin/bash
python demo/image_demo.py \
/datasets/UAV/detect/val/images \
/workspace/mmyolo/configs/yolov5/yolov5_x-UAV.py \
/workspace/work_dirs/UAV/epoch_10.pth \
--out-dir /workspace/work_dirs/UAV/output