import os
import sys
import argparse
import numpy as np

import paddle
from paddle.jit import TracedLayer

import ppgan
from ppgan.utils.config import get_config
from ppgan.utils.setup import setup
from ppgan.engine.trainer import Trainer
from ppgan.utils.animate import normalize_kp

import numpy as np
from scipy.spatial import ConvexHull

import paddle
from paddle.static import InputSpec

    
cfg = get_config('configs/basicvsr_reds.yaml')
# cfg = get_config(args.config_file)
model = ppgan.models.builder.build_model(cfg.model)
model.setup_train_mode(is_train=False)

weight = paddle.load('basicvsr_reds.pdparams')

# model.nets['Gen_Full'].kp_extractor.set_state_dict(weight['kp_detector'])
# model.nets['Gen_Full'].generator.set_state_dict(weight['generator'])
model.nets['generator'].set_state_dict(weight['generator'])

# source = np.random.random([1, 10, 3, 64, 64]).astype('float32')
source = np.ones([1, 10, 3, 64, 64]).astype('float32')
# driving = np.random.random([1, 3, 256, 256]).astype('float32')

source = paddle.to_tensor(source)
# driving = paddle.to_tensor(driving)

# vsr_dygraph, vsr_static = TracedLayer.trace(model.nets['generator'], inputs=[source])

# save_dirname = 'basicvsr_dy2st/saved_infer_model'
# # 将转换后的模型保存
# vsr_static.save_inference_model(save_dirname, feed=[0], fetch=[0])
y = model.nets['generator'](source)
print(y)

# paddle.jit.save(model.nets['generator'], "./basicvsr_dy2st/generator", input_spec=[source])