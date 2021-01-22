'''
Reference:
[1] Towards Deep Learning Models Resistant to Adversarial Attacks
Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu
arXiv:1706.06083v3
'''
import torch
import numpy as np
import os
import sys
father_dir = os.path.join('/', *os.path.realpath(__file__).split(os.path.sep)[:-2])
if not father_dir in sys.path:
    sys.path.append(father_dir)
from attack.attack_base import AttackBase, clip_eta

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

class IPGD(AttackBase):
    # ImageNet pre-trained mean and std
    # _mean = torch.tensor(np.array([0.485, 0.456, 0.406]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    # _std = torch.tensor(np.array([0.229, 0.224, 0.225]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])

    # _mean = torch.tensor(np.array([0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    # _std = torch.tensor(np.array([1.0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    def __init__(self, eps = 6 / 255.0, sigma = 3 / 255.0, nb_iter = 20,
                 norm = np.inf, DEVICE = torch.device('cpu'),
                 mean = torch.tensor(np.array([0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
                 std = torch.tensor(np.array([1.0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]), random_start = True):
        '''
        :param eps: maximum distortion of adversarial examples
        :param sigma: single step size
        :param nb_iter: number of attack iterations
        :param norm: which norm to bound the perturbations
        '''
        self.eps = eps
        self.sigma = sigma
        self.nb_iter = nb_iter
        self.norm = norm
        self.criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
        self.DEVICE = DEVICE
        self._mean = mean.to(DEVICE)
        self._std = std.to(DEVICE)
        self.random_start = random_start

    def single_attack(self, net, inp, label, eta, target = None):
        '''
        Given the original image and the perturbation computed so far, computes
        a new perturbation.
        :param net:
        :param inp: original image
        :param label:
        :param eta: perturbation computed so far
        :return: a new perturbation
        '''

        adv_inp = inp + eta

        #net.zero_grad()

        pred = net(adv_inp)
        if target is not None:
            targets = torch.sum(pred[:, target])
            grad_sign = torch.autograd.grad(targets, adv_in, only_inputs=True, retain_graph = False)[0].sign()

        else:
            loss = self.criterion(pred, label)
            grad_sign = torch.autograd.grad(loss, adv_inp,
                                            only_inputs=True, retain_graph = False)[0].sign()

        # adv_inp = adv_inp + grad_sign * (self.sigma / self._std)
        # tmp_adv_inp = adv_inp * self._std +  self._mean
        # tmp_inp = inp * self._std + self._mean
        # tmp_adv_inp = torch.clamp(tmp_adv_inp, 0, 1) ## clip into 0-1
        # #tmp_adv_inp = (tmp_adv_inp - self._mean) / self._std
        # tmp_eta = tmp_adv_inp - tmp_inp
        # tmp_eta = clip_eta(tmp_eta, norm=self.norm, eps=self.eps, DEVICE=self.DEVICE)
        # eta = tmp_eta/ self._std

        eta = eta + self.sigma * grad_sign
        eta.data = clamp(eta, -self.eps, self.eps)
        eta.data[:inp.size(0)] = clamp(eta[:inp.size(0)], lower_limit - inp, upper_limit - inp)

        return eta

    def attack(self, net, inp, label, target = None):
        
        eta = torch.zeros_like(inp)
        if self.random_start:
            for ii in range(len(self.eps)):
                eta[:, ii, :, :].uniform_(-self.eps[ii][0][0].item(), self.eps[ii][0][0].item())
        eta.requires_grad = True
        # else:
        #     eta = torch.zeros_like(inp)
        eta = eta.to(self.DEVICE)
        # eta = (eta - self._mean) / self._std
        net.eval()

        inp.requires_grad = True
        eta.requires_grad = True
        for i in range(self.nb_iter):
            eta = self.single_attack(net, inp, label, eta, target)
            #print(i)

        #print(eta.max())
        adv_inp = inp + eta
        # tmp_adv_inp = adv_inp * self._std +  self._mean
        # tmp_adv_inp = torch.clamp(tmp_adv_inp, 0, 1)
        # adv_inp = (tmp_adv_inp - self._mean) / self._std

        return adv_inp

    def to(self, device):
        self.DEVICE = device
        self._mean = self._mean.to(device)
        self._std = self._std.to(device)
        self.criterion = self.criterion.to(device)

def test_IPGD():
    pass
if __name__ == '__main__':
    test_IPGD()
