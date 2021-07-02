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


def normalize_kp(kp_source,
                 kp_driving,
                 kp_driving_initial,
                 adapt_movement_scale=False,
                 use_relative_movement=False,
                 use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = paddle.matmul(
                kp_driving['jacobian'],
                paddle.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = paddle.matmul(jacobian_diff,
                                               kp_source['jacobian'])

    return kp_new
    
cfg = get_config('configs/firstorder_vox_256.yaml')
# cfg = get_config(args.config_file)
# model = ppgan.models.builder.build_model(cfg.model)
# model.setup_train_mode(is_train=False)

# weight = paddle.load('vox-cpk.pdparams')

# model.nets['Gen_Full'].kp_extractor.set_state_dict(weight['kp_detector'])
# model.nets['Gen_Full'].generator.set_state_dict(weight['generator'])

# source = np.random.random([1, 3, 256, 256]).astype('float32')
# driving = np.random.random([1, 3, 256, 256]).astype('float32')

# source = paddle.to_tensor(source)
# driving = paddle.to_tensor(driving)

# # kp_dygraph, kp_static = TracedLayer.trace(model.nets['Gen_Full'].kp_extractor, inputs=[source])

# save_dirname = 'fom_dy2st/saved_infer_model'
# # 将转换后的模型保存
# # static_layer.save_inference_model(save_dirname, feed=[0], fetch=[0])

# paddle.jit.save(model.nets['Gen_Full'].kp_extractor, "./fom_dy2st/kp_detector", input_spec=[source])

class Full(paddle.nn.Layer):
    def __init__(self):
        super(Full, self).__init__()
        model = ppgan.models.builder.build_model(cfg.model)
        model.setup_train_mode(is_train=False)
        weight = paddle.load('vox-cpk.pdparams')

        model.nets['Gen_Full'].kp_extractor.set_state_dict(weight['kp_detector'])
        model.nets['Gen_Full'].generator.set_state_dict(weight['generator'])

        self.kp_detector = model.nets['Gen_Full'].kp_extractor
        self.generator = model.nets['Gen_Full'].generator

    # @paddle.jit.to_static
    # @paddle.jit.to_static(input_spec=[InputSpec(shape=[None, 3, 256, 256], name='source'), {'value': InputSpec(shape=[None, 10, 2], name='value1')}, {'jacobian': InputSpec(shape=[None, 10, 2, 2], name='j1')}, InputSpec(shape=[None, 3, 256, 256], name='driving_frame')])
    def forward(self, source, kp_source, driving_frame, kp_driving_initial):
        relative = True
        adapt_movement_scale = False

        kp_driving = self.kp_detector(driving_frame)
        print('debug export', kp_driving['value'].shape, kp_driving['jacobian'].shape)
        kp_norm = normalize_kp(
            kp_source=kp_source,
            kp_driving=kp_driving,
            kp_driving_initial=kp_driving_initial,
            use_relative_movement=relative,
            use_relative_jacobian=relative,
            adapt_movement_scale=adapt_movement_scale)
        # print('generator shape:', source.shape, kp_source.shape, kp_norm.shape)
        out = self.generator(source, kp_source=kp_source, kp_driving=kp_norm)
        return out['prediction']



full_model = Full()

source = np.random.rand(1, 3, 256, 256).astype('float32')
driving = np.random.rand(1, 3, 256, 256).astype('float32')
value = np.random.rand(1, 10, 2).astype('float32')
j = np.random.rand(1, 10, 2, 2).astype('float32')
# driving1 = np.random.random([1, 3, 256, 256]).astype('float32')
# driving2 = np.random.random([1, 3, 256, 256]).astype('float32')

source = paddle.to_tensor(source)
driving1 = {'value': paddle.to_tensor(value), 'jacobian': paddle.to_tensor(j)}
driving2 = {'value': paddle.to_tensor(value), 'jacobian': paddle.to_tensor(j)}
driving = paddle.to_tensor(driving)

paddle.jit.save(full_model, "./fom_dy2st/gen_full", input_spec=[source, driving1, driving, driving2])
