#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "/kaggle/input/gtadataset/HW2_ObjectDetection_2023"
        # self.data_dir = "datasets/HW2_ObjectDetection_2023"
        self.train_ann = "train_labels.json"
        self.val_ann = "val_labels.json"
        self.output_dir = "/kaggle/working/"
        self.num_classes = 1    
        self.input_size = (704, 704)
        self.test_size = (704, 704) 
        self.mosaic_scale = (0.5, 2)
        self.max_epoch = 17
        self.data_num_workers = 4
        self.eval_interval = 1
        self.print_interval = 10
        self.save_history_ckpt = False