import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn.parameter import Parameter

# From Stephane Guillitte
# https://discuss.pytorch.org/t/problems-on-implementation-of-deep-neural-decision-forest/837

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)

class DeepNeuralDecisionForest(nn.Module):
    def __init__(self, p_keep_conv, p_keep_hidden, n_leaf, n_label, n_tree, n_depth):
        super(DeepNeuralDecisionForest, self).__init__()

        self.conv = nn.Sequential()
        self.conv.add_module('conv1', nn.Conv2d(1, 10, kernel_size=5))
        self.conv.add_module('relu1', nn.ReLU())
        self.conv.add_module('pool1', nn.MaxPool2d(kernel_size=2))
        self.conv.add_module('drop1', nn.Dropout(1-p_keep_conv))
        self.conv.add_module('conv2', nn.Conv2d(10, 20, kernel_size=5))
        self.conv.add_module('relu2', nn.ReLU())
        self.conv.add_module('pool2', nn.MaxPool2d(kernel_size=2))
        self.conv.add_module('drop2', nn.Dropout(1-p_keep_conv))

        self._nleaf = n_leaf
        self._nlabel = n_label
        self._ntree = n_tree
        self._ndepth = n_depth
        self._batchsize = args.batch_size

        self.treelayers = nn.ModuleList()
        self.pi_e = nn.ParameterList()
        for i in range(self._ntree):
            treelayer = nn.Sequential()
            treelayer.add_module('sub_linear1', nn.Linear(320, 50))
            treelayer.add_module('sub_relu', nn.ReLU())
            treelayer.add_module('sub_drop1', nn.Dropout(1-p_keep_hidden))
            treelayer.add_module('sub_linear2', nn.Linear(50, self._nleaf))
            treelayer.add_module('sub_sigmoid', nn.Sigmoid())

            self.treelayers.append(treelayer)
            self.pi_e.append(Parameter(self.init_prob_weights([self._nleaf, self._nlabel], -2, 2)))

    def init_pi(self):
        return torch.ones(self._nleaf, self._nlabel)/float(self._nlabel)

    def init_weights(self, shape):
        return torch.randn(shape).uniform(-0.01,0.01)

    def init_prob_weights(self, shape, minval=-5, maxval=5):
        return torch.Tensor(shape[0], shape[1]).uniform_(minval, maxval)

    def compute_mu(self, flat_decision_p_e):
        n_batch = self._batchsize
        batch_0_indices = torch.range(0, n_batch * self._nleaf - 1, self._nleaf).unsqueeze(1).repeat(1, self._nleaf).long()

        in_repeat = self._nleaf // 2
        out_repeat = n_batch

        batch_complement_indices = torch.LongTensor(
            np.array([[0] * in_repeat, [n_batch * self._nleaf] * in_repeat] * out_repeat).reshape(n_batch, self._nleaf))

        # First define the routing probabilistics d for root nodes
        mu_e = []
        indices_var = Variable((batch_0_indices + batch_complement_indices).view(-1))
        indices_var = indices_var.cuda()
        # iterate over each tree
        for i, flat_decision_p in enumerate(flat_decision_p_e):
            mu = torch.gather(flat_decision_p, 0, indices_var).view(n_batch, self._nleaf)
            mu_e.append(mu)

        # from the scond layer to the last layer, we make the decison nodes
        for d in range(1, self._ndepth + 1):
            indices = torch.range(2 ** d, 2 ** (d + 1) - 1) - 1
            tile_indices = indices.unsqueeze(1).repeat(1, 2 ** (self._ndepth - d + 1)).view(1, -1)
            batch_indices = batch_0_indices + tile_indices.repeat(n_batch, 1).long()

            in_repeat = in_repeat // 2
            out_repeat = out_repeat * 2
            # Again define the indices that picks d and 1-d for the nodes
            batch_complement_indices = torch.LongTensor(
                np.array([[0] * in_repeat, [n_batch * self._nleaf] * in_repeat] * out_repeat).reshape(n_batch, self._nleaf))

            mu_e_update = []
            indices_var = Variable((batch_indices + batch_complement_indices).view(-1))
            indices_var = indices_var.cuda()
            for mu, flat_decision_p in zip(mu_e, flat_decision_p_e):
                mu = torch.mul(mu, torch.gather(flat_decision_p, 0, indices_var).view(
                    n_batch, self._nleaf))
                mu_e_update.append(mu)
            mu_e = mu_e_update
        return mu_e

    def compute_py_x(self, mu_e, leaf_p_e):
        py_x_e = []
        n_batch = self._batchsize

        for i in range(len(mu_e)):
            py_x_tree = mu_e[i].unsqueeze(2).repeat(1, 1, self._nlabel).mul(leaf_p_e[i].unsqueeze(0).repeat(n_batch, 1, 1)).mean(1)
            py_x_e.append(py_x_tree.squeeze().unsqueeze(0))

        py_x_e = torch.cat(py_x_e, 0)
        py_x = py_x_e.mean(0).squeeze()

        return py_x

    def forward(self, x):
        feat = self.conv.forward(x)
        feat = feat.view(-1, 320)
        self._batchsize = x.size(0)

        flat_decision_p_e = []
        leaf_p_e = []

        for i in range(len(self.treelayers)):
            decision_p = self.treelayers[i].forward(feat)
            decision_p_comp = 1 - decision_p
            decision_p_pack = torch.cat((decision_p, decision_p_comp))
            flat_decision_p = decision_p_pack.view(-1)
            flat_decision_p_e.append(flat_decision_p)
            leaf_p = F.softmax(self.pi_e[i])
            leaf_p_e.append(leaf_p)

        mu_e = self.compute_mu(flat_decision_p_e)

        py_x = self.compute_py_x(mu_e, leaf_p_e)
        return torch.log(py_x)


################ Definition #########################
DEPTH = 3  # Depth of a tree
N_LEAF = 2 ** (DEPTH + 1)  # Number of leaf node
N_LABEL = 10  # Number of classes
N_TREE = 10 # Number of trees (ensemble)
N_BATCH = args.batch_size#args.batch-size  # Number of data points per mini-batch
# network hyperparameters
p_conv_keep = 0.8
p_full_keep = 0.5


model = DeepNeuralDecisionForest(p_keep_conv = p_conv_keep, p_keep_hidden = p_full_keep, n_leaf= N_LEAF, n_label= N_LABEL, n_tree= N_TREE, n_depth= DEPTH)

#model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss((output), target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)