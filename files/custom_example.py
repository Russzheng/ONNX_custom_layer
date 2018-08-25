import torch.onnx
import torchvision
from torch import nn
from torch.autograd import Function

class MyReLUFunction(Function):

    @staticmethod
    def symbolic(g, input):
        return g.op('MyReLU', input)

    @staticmethod
    def forward(ctx, input):
        ctx.input = ctx
        return input.clamp(0)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input.masked_fill_(ctx.input < 0, 0)
        return grad_input

class MyReLU(nn.Module):

    def forward(self, input):
        return MyReLUFunction.apply(input)

## model = torchvision.models.resnet18()
## dummy_input = torch.randn(10, 3, 224, 224)
model = nn.Sequential(
    nn.Conv2d(1, 1, 3),
    MyReLU(),
)
dummy_input = torch.randn(10, 1, 3, 3)
torch.onnx.export(model, dummy_input, "model.onnx", verbose = True)
import os
os.remove("model.onnx")