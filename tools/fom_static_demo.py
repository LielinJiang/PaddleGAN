import paddle
import numpy as np

paddle.enable_static()

# 构建模型
# startup_prog = paddle.static.default_startup_program()
# main_prog = paddle.static.default_main_program()
# with paddle.static.program_guard(main_prog, startup_prog):
#     image = paddle.static.data(name="img", shape=[64, 784])
#     w = paddle.create_parameter(shape=[784, 200], dtype='float32')
#     b = paddle.create_parameter(shape=[200], dtype='float32')
#     hidden_w = paddle.matmul(x=image, y=w)
#     hidden_b = paddle.add(hidden_w, b)
exe = paddle.static.Executor(paddle.CUDAPlace(0))
# exe.run(startup_prog)

# 保存预测模型
# path_prefix = "./infer_model"
# paddle.static.save_inference_model(path_prefix, [image], [hidden_b], exe)
path_prefix = "./fom_dy2st/gen_full"
[inference_program, feed_target_names, fetch_targets] = (
    paddle.static.load_inference_model(path_prefix, exe))
print(feed_target_names)
source = np.random.rand(1, 3, 256, 256).astype('float32')
driving = np.random.rand(1, 3, 256, 256).astype('float32')
value1 = np.random.rand(1, 10, 2).astype('float32')
j1 = np.random.rand(1, 10, 2, 2).astype('float32')
value2 = np.random.rand(1, 10, 2).astype('float32')
j2 = np.random.rand(1, 10, 2, 2).astype('float32')
# tensor_img = np.array(np.random.random((64, 784)), dtype=np.float32)
results = exe.run(inference_program,
              feed={feed_target_names[0]: source, feed_target_names[1]: j1, feed_target_names[2]: value1, feed_target_names[3]: driving, feed_target_names[4]: j2, feed_target_names[5]: value2},
              fetch_list=fetch_targets)

for r in results:
    print(r.shape)
# 在上述示例中，inference program 被保存在 "./infer_model.pdmodel" 文件中，
# 参数被保存在 "./infer_model.pdiparams" 文件中。
# 加载 inference program 后， executor可使用 fetch_targets 和 feed_target_names,
# 执行Program，并得到预测结果。