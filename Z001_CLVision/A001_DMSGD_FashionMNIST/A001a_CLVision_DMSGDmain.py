###===###
# This is the main script for
# applying a
# DMSGD neural optimiser on
# Fashion MNIST

###===###
# specifying pooling to making life easier
# if we were to change from the MNIST dataset
# to STL10
FMNIST_pooling = 8

#---
# RNNstep = 100; RNNstep2 = 10000 means that
# the neural optimiser will be subjected to
#   100 steps of meta-training and that
#   it will be meta-tested to update the base learner
#   for the much longer 10000 steps
RNNstep  = 100
RNNstep2 = 10000

#---
# the network will unroll every 5 steps
# this refers to the unrolling instance Xi
# which is highlighted in Green in Algorithm 1
RoIstep = 5

#---
# the network will undergo 1 learner trial
# this refers to script-Q
# which is highlighted in Pink in Algorithm 1
YOC     = 1

#---
# misc for drawing device
Rsplice     = 1
Rsplice2    = 10
test_every  = 1
test_sim    = 10

###===###
# Auxiliary function for evaluation
def test(model, test_loader):
    test_loss = 0
    correct = 0
    tot_num = 0
    with torch.no_grad():
        for data, target in test_loader:
            
            data, target = data.cuda(), target.cuda()

            output     = model(data)
            test_loss += F.nll_loss(output, target).item()
            pred       = output.argmax(dim=1, keepdim=True)
            correct   += pred.eq(target.view_as(pred)).sum().item()
            tot_num   += len(target)

    test_loss /= tot_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, tot_num,
        100. * correct / tot_num))

    return test_loss, correct / tot_num

###===###
# Auxiliary function for plotting
def testphasetest(Optimiser, optimiseePD, train_loader, test_loader):

    allacc = []
    alllos = []
    C1000 = []
    L1000 = []
    C2000 = []
    L2000 = []
    C3000 = []
    L3000 = []
                
    for itr in range(test_sim):

        if itr+1 > 1:
            plt.subplot(222)
            plt.cla()
            plt.draw()
            plt.pause(1)
            plt.title("Testing Acc")

            #---
            plt.subplot(224)
            plt.cla()
            plt.draw()
            plt.pause(1)

            plt.title("Testing Loss")               
            

        print('this is testing simulation numero {}'.format(itr))
        TTstep200 = []
        TTloss200 = []
        TTcstepF  = []
        
        ccstep = 0
        
        model = MyMLP()
        model.cuda()

        for k in range(RNNstep2 // RoIstep):

            train_iter = iter(train_loader)

            Optimiser.ROH(keep_states=k > 0, model=model)

            loss_sum = 0
            prev_loss = torch.zeros(1)
            prev_loss = prev_loss.cuda()

            for j in range(RoIstep):

                ccstep += 1
                if np.mod(ccstep, int(RNNstep2/Rsplice2)) == 0:
                    print('testing simulation {}, now {}, all {}'.format(itr+1, ccstep, RNNstep2))
                    test_loss, cur_acc = test(RefModel, test_loader)
                    TTstep200.append(cur_acc)
                    TTloss200.append(test_loss)
                    TTcstepF.append(ccstep)

                    plt.subplot(222)
                    plt.title("Testing Acc")                  
                    plt.ylim([-0.125, 1.125])
                    plt.xlim([-RNNstep2/200*10,
                              RNNstep2 + RNNstep2/200+10])
                    plt.plot(TTcstepF, TTstep200, '-o', color = 'blue')
                    plt.draw()

                    plt.pause(1)

                    plt.subplot(224)
                    plt.title("Testing Loss")
                    plt.ylim([0, 0.15])
                    plt.xlim([-RNNstep2/200*10,
                              RNNstep2 + RNNstep2/200+10])
                    plt.plot(TTcstepF, TTloss200, '-o', color = 'pink')
                    plt.draw()

                    plt.pause(1)                    

                if ccstep == RNNstep2:
                    allacc.append(cur_acc)
                    alllos.append(test_loss)

                if ccstep == 1000:
                    C1000.append(cur_acc)
                    L1000.append(test_loss)
                if ccstep == 2000:
                    C2000.append(cur_acc)
                    L2000.append(test_loss)
                if ccstep == 3000:
                    C3000.append(cur_acc)
                    L3000.append(test_loss)
                            
                x, y = next(train_iter)              
                x, y = x.cuda(), y.cuda()
                x, y = Variable(x), Variable(y)
                f_x  = model(x)
                loss = F.nll_loss(f_x, y)
                model.zero_grad()
                loss.backward()
                RefModel = Optimiser.UpdateTransfer(model)
                RefModel.AvgPool = nn.AvgPool2d(optimiseePD, stride = 1)

    ###===###
    allacc = np.array(allacc) * 100
    alllos = np.array(alllos)

    C1000 = np.array(C1000) * 100
    L1000 = np.array(L1000)
    C2000 = np.array(C2000) * 100
    L2000 = np.array(L2000)
    C3000 = np.array(C3000) * 100
    L3000 = np.array(L3000)
    
    return allacc, alllos, C3000, L3000, C2000, L2000, C1000, L1000
    

###===###
# defining dependencies
import  torch
import  torch.nn                as      nn
import  torch.nn.functional     as      F
import  torch.optim             as      optim
from    torch.autograd          import  Variable
from    torchvision             import  datasets, transforms
import  numpy                   as      np
import  torchvision
import  torchvision.transforms  as      transforms
from    torchvision             import  datasets, transforms
import  matplotlib.pyplot       as      plt
#---
# specific dependencies
from    A001b_CLVision_Utils    import  get_batch, MyMLP
from    A001c_CLVision_DMSGD    import  RefMode, RNNOptimiser

###===###
# seeding procesdure
seed = 101           
np.random.seed(             seed)
torch.manual_seed(          seed)
torch.cuda.manual_seed_all( seed)

###===###
# misc
BS_train    = 32
MaxE        = 1
MLPPC       = 25450
RNNHD       = 20
RNN_OM      = 10

###===###
# this is for preparing the Fashion MNIST dataset
Dload = True
dLoc    = './data'
c10_mean = (0.5, 0.5, 0.5)
c10_std  = (0.5, 0.5, 0.5)
c10_bs   = 128

kwargs = {'pin_memory': True}
tr_S_loader = torch.utils.data.DataLoader(
                    datasets.FashionMNIST('./data_F', train = True, download = True,
                                   transform = transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,))
                                   ])),
                    batch_size = BS_train, shuffle = True, **kwargs)

te_S_loader = torch.utils.data.DataLoader(
                    datasets.FashionMNIST('./data_F', train = False, download = True,
                                   transform = transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,))
                                   ])),
                    batch_size = BS_train, shuffle = True, **kwargs)  

###===###
# defining the base learner of the experimental setup
RefModel    = MyMLP()
RefModel    = RefModel.cuda()

###===###
# defines the neural optimiser
Optimiser   = RNNOptimiser(RefMode(RefModel),   
                           PC = MLPPC,          
                           HD = RNNHD,          
                           OM = RNN_OM)         
Optimiser.cuda()
# OOO refers to the Optimiser Of the neural Optimiser
OOO = optim.Adam(Optimiser.parameters(), lr=1e-3)

###===###
# misc for drawing device
ccstep = 0
last200step = None
last200loss = None

###===###
# for the Q amount of learning trials
# see line 4 of Algorithm 1
for i in range(YOC):

    print('this is YOC numero {}'.format(i))

    # we will plot the update once every few steps
    if np.mod(i + 1, test_every) == 0:
        if last200step is not None:
            plt.subplot(221)
            plt.title("FMNIST-Training Acc")
            plt.cla()
            plt.draw()
            plt.pause(1)
            
            plt.subplot(221)
            plt.ylim([-0.125, 1.125])
            plt.xlim([-RNNstep/200*10,
                      RNNstep + RNNstep/200+10])
            plt.plot(cstepF, last200step, '-o', color = 'green')
            plt.draw()

            plt.pause(1)

            #---
            plt.subplot(223)
            plt.title("FMNIST-Training Loss")
            plt.cla()
            plt.draw()
            plt.pause(1)
            
            plt.subplot(223)
            plt.ylim([0, 0.15])
            plt.xlim([-RNNstep/200*10,
                      RNNstep + RNNstep/200+10])
            plt.plot(cstepF, last200loss, '-o', color = 'red')
            plt.draw()

            plt.pause(1)            

        step200 = []
        loss200 = []
        cstepF  = []
            
        
    ccstep = 0

    # every learning trial starts with a randomly initialised base learner
    # see line 5 of Algorithm 1
    model = MyMLP()
    model.cuda()

    # for every steps before an unroll
    for k in range(RNNstep // RoIstep):

        train_iter = iter(tr_S_loader)

        # and the neural optimiser unrolls if accumulated enough instances
        # see line 11 of Algorithm 1
        Optimiser.ROH(keep_states=k > 0, model=model)

        # upon every unroll,
        # the loss used to update the neural optimiser is reset
        # as you will see soon,
        # loss_sum refers to script-L of Equation (4)
        loss_sum = 0
        prev_loss = torch.zeros(1)
        prev_loss = prev_loss.cuda()

        # for all steps before unrolling
        for j in range(RoIstep):

            #---
            # misc for the draing device
            ccstep += 1
            if np.mod(ccstep, int(RNNstep/Rsplice)) == 0:
                print('YOC {}, now {}, all {}'.format(i+1, ccstep, RNNstep))
                if np.mod(i+1, test_every) == 0:
                    test_loss, cur_acc = test(RefModel, te_S_loader)
                    step200.append(cur_acc)
                    loss200.append(test_loss)
                    cstepF.append(ccstep)

                    plt.subplot(221)
                    plt.title("FMNIST-Training Acc")
                    plt.ylim([-0.125, 1.125])
                    plt.xlim([-RNNstep/200*10,
                              RNNstep + RNNstep/200+10])
                    plt.plot(cstepF, step200, '-o', color = 'blue')
                    plt.draw()

                    plt.pause(1)

                    #---
                    plt.subplot(223)
                    plt.title("FMNIST-Training Loss")
                    plt.ylim([0, 0.15])
                    plt.xlim([-RNNstep/200*10,
                              RNNstep + RNNstep/200+10])
                    plt.plot(cstepF, loss200, '-o', color = 'pink')
                    plt.draw()

                    plt.pause(1)
            #---
            # we acquire new pairs of labelled data
            # see line 7 of Algorithm 1
            x, y = next(train_iter)              
            x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)
            f_x  = model(x)

            #---
            # acquire the loss of the current model
            # see line 8 of Algorithm 1
            loss = F.nll_loss(f_x, y)
            model.zero_grad()

            #---
            # and acquire the gradient of the said loss
            loss.backward()

            #---
            # this gradient will be within the defined model
            # and send this model to the neural optimiser
            # to be updated
            # see lines 16 ~ 18 of Algorithm 1
            # also see the neural optimiser script for more comments
            RefModel = Optimiser.UpdateTransfer(model)

            #---
            # now we see how much better the base model has become
            f_x = RefModel(x)
            loss = F.nll_loss(f_x, y)

            #---
            # and this change is used as an indicator
            # for the potentiality to improve the neural optimiser
            # see line 12 of Algorithm 1
            loss_sum += (loss - Variable(prev_loss))
            prev_loss = loss.data

        Optimiser.zero_grad()

        #---
        # update the neural optimiser after acquiring Xi amount of unroll
        # see line 13 of Algorithm 1
        loss_sum.backward()

        #---
        for param in Optimiser.parameters():
            try:
                param.grad.data.clamp_(-1, 1)
            except:
                continue
        OOO.step()

    ###===###       

    ###===###
    if np.mod(i+1, test_every) == 0:
            last200step = step200
            last200loss = loss200
        
CFin,   LFin, \
        C3000,  L3000,\
        C2000,  L2000,\
        C1000,  L1000 =\
            testphasetest(Optimiser, FMNIST_pooling, tr_S_loader, te_S_loader)

if 1:
    C1000.sort()
    C1000 = C1000[-5:]
    C2000.sort()
    C2000 = C2000[-5:]
    C3000.sort()
    C3000 = C3000[-5:]
    CFin.sort()
    CFin = CFin[-5:]

    ###===###
    print('+=+'*20)
    print('SUMMARY')

    print('This is E{} + S{}'.format(YOC, RNNstep))

    print('-'*10)
    print('for 1000 steps')
    print('{} +/- {} %'.format(\
            round(C1000.mean(), 4),
            round(C1000.std() * 1.96 , 4)))
    print('for 2000 steps')
    print('{} +/- {} %'.format(\
            round(C2000.mean(), 4),
            round(C2000.std() * 1.96 , 4)))
    print('for 3000 steps')
    print('{} +/- {} %'.format(\
            round(C3000.mean(), 4),
            round(C3000.std() * 1.96 , 4)))
    print('for all steps')
    print('{} +/- {} %'.format(\
            round(CFin.mean(), 4),
            round(CFin.std() * 1.96 , 4)))

    print('and has a loss performance of')
    print('-'*10)
    print('{} +/- {}'.format(\
            round(L1000.mean(), 4),
            round(L1000.std() *1.96, 4)))
    print('{} +/- {}'.format(\
            round(L2000.mean(), 4),
            round(L2000.std() *1.96, 4)))
    print('{} +/- {}'.format(\
            round(L3000.mean(), 4),
            round(L3000.std() *1.96, 4)))
    print('{} +/- {}'.format(\
            round(LFin.mean(), 4),
            round(LFin.std() *1.96, 4)))

