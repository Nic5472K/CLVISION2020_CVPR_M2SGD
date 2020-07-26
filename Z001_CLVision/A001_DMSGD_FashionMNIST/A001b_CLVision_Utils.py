import  torch
import  torch.nn            as      nn
import  torch.nn.functional as      F
import  torch.optim         as      optim
from    torch.autograd      import  Variable

###===###
# This is the base learner for the Fashion MNIST experiments
class MyMLP(nn.Module):

    def __init__(self):
        super(MyMLP, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 32)
        self.linear2 = nn.Linear(32, 10)

    def forward(self, inputs):
        x = inputs.view(-1, 28 * 28)
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return F.log_softmax(x, dim = 1)

###===###
# misc
def get_batch(batch_size):
    x = torch.randn(batch_size, 10)
    x = x - 2 * x.pow(2)
    y = x.sum(1)
    return x, y
