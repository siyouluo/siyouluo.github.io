# python creat & train model
import torch
import torch.nn as nn
import torch.nn.functional as F

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
    #  np.random.seed(seed)
    #  random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b, 1元线性回归
        self.fc = nn.Linear(1,2)

    def forward(self, x):
        x = self.fc(x)
        return x


model = Net()
print(model)

input_x = torch.randn(1,1)
output_y = model(input_x)
print(f"output_y: {output_y}")
W = torch.Tensor([[12,4]])
b = 9
target_y = torch.matmul(input_x,W)
# target_y = input_x*w+9  # y = kx+b
# target_y = target_y.view(1,-1)  # 使目标值与数据值尺寸一致
print(f"target_y: {target_y}")
criterion = nn.MSELoss()

loss = criterion(output_y, target_y)
print(f"loss: {loss}")

import torch.optim as optim

# 创建优化器(optimizer）
optimizer = optim.SGD(model.parameters(), lr=0.01)

for i in range(2000):
    # 在训练的迭代中：
    optimizer.zero_grad()   # 清零梯度缓存
    input_x = torch.randn(1,1)
    target_y = torch.matmul(input_x,W)+b
    # target_y = target_y.view(1,-1)
    output_y = model(input_x)
    loss = criterion(output_y, target_y)
    loss.backward()
    optimizer.step()    # 更新参数
print("train model...")
print(f"params: {list(model.parameters())}")
print(f"input_x: {input_x}")
print(f"target_y:{target_y} = input_x*{W}+{b} = {torch.matmul(input_x,W)+b}")
print(f"output_y: {output_y}")

device = torch.device('cpu')


torch.onnx.export(
    model.to(device),
    input_x,
    'model.onnx',
    opset_version=11,
    input_names=["input_x"],		# 输入名
    output_names=["output_y"],	# 输出名
    dynamic_axes={  "input_x":{0:"batch_size"},	# 批处理变量
                    "output_y":{0:"batch_size"}
                    },
    verbose=False  # set True to print model graph
    )

import onnx
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
print(f"\nmodel graph:{onnx.helper.printable_graph(onnx_model.graph)}\n")

import onnxruntime
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

ort_session = onnxruntime.InferenceSession("model.onnx")
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_x)}
# print(ort_inputs)
ort_output = ort_session.run(None,input_feed=ort_inputs)

print(f"ort_output: {ort_output[0]}")

