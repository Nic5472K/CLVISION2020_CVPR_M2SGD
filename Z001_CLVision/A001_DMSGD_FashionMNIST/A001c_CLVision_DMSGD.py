###===###
# This is the script code for the DMSGD neural optimiser

#---
import  torch
import  torch.nn            as      nn
import  torch.nn.functional as      Fnc
import  torch.optim         as      optim
from    torch.autograd      import  Variable
import  numpy               as      np
import  math
#---
from    functools           import reduce
from    operator            import mul

from scipy.stats import ortho_group

###===###
# The following code is used for pre-processing the raw gradient
# refer to descriptions in Appendix A
def preprocess_gradients(x):
    p = 10
    eps = 1e-6
    indicator = (x.abs() > math.exp(-p)).float()
    x1 = (x.abs() + eps).log() / p * indicator - (1 - indicator)
    x2 = x.sign() * indicator + math.exp(p) * x * (1 - indicator)

    x1 = x1.unsqueeze(1)
    x2 = x2.unsqueeze(1)

    return torch.cat((x1, x2), 1)

###===###
# The following is our DMSGD neural optimiser
class RNNOptimiser(nn.Module):

    def __init__(self, model, PC, HD, OM):
        super(RNNOptimiser, self).__init__()

        ###===###
        self.RefM = model
        self.HD = HD
        self.PC = PC

        ###===###
        # introduces an LSTM to introduce
        # a dynamic set of learning rates to update the base learner
        # see descriptions in Section 3
        self.LSTM1  = nn.LSTMCell(HD, HD)
        self.PT1    = nn.Linear(PC, HD)
        self.PT2    = nn.Linear(5, 1)

        ###===###
        # This is the inherited static neural optimisers
        # from the original meta-SGD
        self.alpha_base = nn.Parameter(torch.ones(PC) * 1e-3)

        self.W2 = nn.Linear(HD, PC)        

    ###===###
    # ROH refers to Reset Optimiser Hidden states
    # removes variables from the computational graph
    # and achieves this mainly via the ".data" command
    def ROH(self, keep_states=False, model=None):

        num_layers = 2
        #---
        self.RefM.reset()
        self.RefM.copy_params_from(model)

        if keep_states:
            for i in range(num_layers):
                self.h1x[i] = Variable(self.h1x[i].data)
                self.c1x[i] = Variable(self.c1x[i].data)
                self.h2x[i] = Variable(self.h2x[i].data)
                self.c2x[i] = Variable(self.c2x[i].data)
            
        else:
            self.h1x = []
            self.c1x = []
            self.h2x = []
            self.c2x = []
            for i in range(num_layers):
                self.h1x.append(Variable(torch.zeros(1, self.HD)))
                self.c1x.append(Variable(torch.zeros(1, self.HD)))
                self.h1x[i], self.c1x[i] = \
                             self.h1x[i].cuda(), self.c1x[i].cuda()
                self.h2x.append(Variable(torch.zeros(1, self.HD)))
                self.c2x.append(Variable(torch.zeros(1, self.HD)))
                self.h2x[i], self.c2x[i] = \
                             self.h2x[i].cuda(), self.c2x[i].cuda()

    def forward(self, pgrads, grad):

        ###===###
        # takes in all information
        pre_XI0 = torch.cat([pgrads,
                             grad.unsqueeze(1),
                             torch.relu(self.alpha_base).unsqueeze(1)],
                            dim = 1).transpose(0, 1)
        pre_XI1 = self.PT1(pre_XI0).transpose(0, 1)        
        pre_XI2 = self.PT2(pre_XI1).squeeze(1)
        B_k = pre_XI2

        ###===###
        # loads the appropriate hidden states
        S_k = self.c1x[0]
        Q_k = self.h1x[0]

        B_k = B_k.unsqueeze(0)

        ###===###
        # applies the LSTM
        # see Equation (8) of Table 1
        Q_k, S_k = self.LSTM1(B_k, (Q_k, S_k))
        self.h1x[0] = Q_k
        self.c1x[0] = S_k

        Z_k     = Q_k.squeeze(0)

        # and then piece together
        # the static learning rate and the dynamic learning rates
        # to get the final learning rates
        # see Equation (7) of Table 1
        # gamma is default to a vector of ones,
        # so it is not explicitly stated here
        self.i = torch.relu(self.alpha_base + torch.tanh(self.W2(Z_k)))
        
        return self.i
    # see further comments in UpdateTransfer

    def UpdateTransfer(self, CurOptimisee):

        grads = []
        
        for module in CurOptimisee.children():
            if isinstance(module, nn.Linear):
                grads.append(module._parameters['weight'].grad.data.view(-1))
                grads.append(module._parameters['bias'].grad.data.view(-1))

            if isinstance(module, nn.Conv2d):
                grads.append(module._parameters['weight'].grad.data.view(-1))                

        flat_params = self.RefM.get_flat_params()
        flat_params = flat_params.unsqueeze(1)
        flat_grads = preprocess_gradients(torch.cat(grads))

        inputs = Variable(torch.cat((flat_grads, flat_params.data), 1))
        flat_params = flat_params.squeeze(1)

        #---
        task_LR     = self(inputs, torch.cat(grads))

        ###===###
        # This is the part which we update
        # the parameters of the base learner
        flat_params = flat_params - task_LR * torch.cat(grads)

        self.RefM.set_flat_params(flat_params)

        self.RefM.copy_params_to(CurOptimisee)
        return self.RefM.model      

class RefMode:

    def __init__(self, model):
        self.model = model
        
    def reset(self):
        
        for module in self.model.children():
            if isinstance(module, nn.Linear):
                module._parameters['weight'] = Variable(
                    module._parameters['weight'].data)
                module._parameters['bias'] = Variable(
                    module._parameters['bias'].data)
            if isinstance(module, nn.Conv2d):
                module._parameters['weight'] = Variable(
                    module._parameters['weight'].data)                

    def get_flat_params(self):
        params = []

        for module in self.model.children():
            if isinstance(module, nn.Linear):
                params.append(module._parameters['weight'].view(-1))
                params.append(module._parameters['bias'].view(-1))
            if isinstance(module, nn.Conv2d):
                params.append(module._parameters['weight'].view(-1))                

        return torch.cat(params)

    def set_flat_params(self, flat_params):

        offset = 0

        for i, module in enumerate(self.model.children()):
            if isinstance(module, nn.Linear):
                weight_shape = module._parameters['weight'].size()
                bias_shape = module._parameters['bias'].size()

                weight_flat_size = reduce(mul, weight_shape, 1)
                bias_flat_size = reduce(mul, bias_shape, 1)

                module._parameters['weight'] = flat_params[
                    offset:offset + weight_flat_size].view(*weight_shape)
                module._parameters['bias'] = flat_params[
                    offset + weight_flat_size:offset + weight_flat_size + bias_flat_size].view(*bias_shape)

                offset += weight_flat_size + bias_flat_size
                
            if isinstance(module, nn.Conv2d):
                
                weight_shape = module._parameters['weight'].size()

                weight_flat_size = reduce(mul, weight_shape, 1)

                module._parameters['weight'] = flat_params[
                    offset:offset + weight_flat_size].view(*weight_shape)

                offset += weight_flat_size                

    def copy_params_from(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelA.data.copy_(modelB.data)

    def copy_params_to(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelB.data.copy_(modelA.data)            
