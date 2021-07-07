import paddle
import numpy as np

paddle.enable_static()

exe = paddle.static.Executor(paddle.CUDAPlace(0))

# 保存预测模型
path_prefix = "./basicvsr_dy2st/generator"
[inference_program, feed_target_names, fetch_targets] = (
    paddle.static.load_inference_model(path_prefix, exe))
print(feed_target_names, fetch_targets[0].name)

results = exe.run(inference_program,
              feed={feed_target_names[0]: np.ones([1, 10, 3, 180, 320]).astype('float32')},
              fetch_list=fetch_targets)

for r in results:
    print(r.shape)