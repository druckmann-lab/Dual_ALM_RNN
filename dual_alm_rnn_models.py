import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math
import itertools
from copy import deepcopy

cat = np.concatenate


class TwoHemiRNNLinear(nn.Module):
    '''
    Same as TwoHemiRNNLinear, but use TwoHemiRNNCellLinear2
    '''
    def __init__(self, configs, a, pert_begin, pert_end, zero_init_cross_hemi=False):
        super().__init__()

        self.configs = configs

        self.a = a
        self.pert_begin = pert_begin
        self.pert_end = pert_end
        self.zero_init_cross_hemi = zero_init_cross_hemi
        self.init_cross_hemi_rel_factor = configs['init_cross_hemi_rel_factor']

        self.uni_pert_trials_prob = configs['uni_pert_trials_prob']
        self.left_alm_pert_prob = configs['left_alm_pert_prob']

        self.n_neurons = configs['n_neurons']
        self.n_left_neurons = self.n_neurons//2
        self.n_right_neurons = self.n_neurons - self.n_neurons//2

        self.sigma_input_noise = configs['sigma_input_noise']
        self.sigma_rec_noise = configs['sigma_rec_noise']

        # Define left and right ALM
        self.left_alm_inds = np.arange(self.n_neurons//2)
        self.right_alm_inds = np.arange(self.n_neurons//2, self.n_neurons)


        self.rnn_cell = TwoHemiRNNCellLinear(n_neurons=self.n_neurons, a=self.a, sigma=self.sigma_rec_noise, 
            zero_init_cross_hemi=self.zero_init_cross_hemi, init_cross_hemi_rel_factor=self.init_cross_hemi_rel_factor)
        
        self.w_xh_linear_left_alm = nn.Linear(1, self.n_neurons//2, bias=False)
        self.w_xh_linear_right_alm = nn.Linear(1, self.n_neurons-self.n_neurons//2, bias=False)

        self.readout_linear_left_alm = nn.Linear(self.n_neurons//2, 1)
        self.readout_linear_right_alm = nn.Linear(self.n_neurons-self.n_neurons//2, 1)

        self.init_params()

        self.drop_p_min = configs['drop_p_min']
        self.drop_p_max = configs['drop_p_max']

        self.xs_left_alm_drop_p = configs['xs_left_alm_drop_p']
        self.xs_right_alm_drop_p = configs['xs_right_alm_drop_p']


        self.xs_left_alm_amp = configs['xs_left_alm_amp']
        self.xs_right_alm_amp = configs['xs_right_alm_amp']



    def get_w_hh(self):
        w_hh = torch.zeros((self.n_neurons, self.n_neurons))
        w_hh[:self.n_neurons//2,:self.n_neurons//2] = self.rnn_cell.w_hh_linear_ll.weight
        w_hh[self.n_neurons//2:,self.n_neurons//2:] = self.rnn_cell.w_hh_linear_rr.weight
        w_hh[self.n_neurons//2:,:self.n_neurons//2] = self.rnn_cell.w_hh_linear_lr.weight
        w_hh[:self.n_neurons//2,self.n_neurons//2:] = self.rnn_cell.w_hh_linear_rl.weight

        return w_hh



    def init_params(self):
        init.normal_(self.w_xh_linear_left_alm.weight, 0.0, 1)
        init.normal_(self.w_xh_linear_right_alm.weight, 0.0, 1)


        init.normal_(self.readout_linear_left_alm.weight, 0.0, 1.0/math.sqrt(self.n_neurons//2))
        init.constant_(self.readout_linear_left_alm.bias, 0.0)

        init.normal_(self.readout_linear_right_alm.weight, 0.0, 1.0/math.sqrt(self.n_neurons-self.n_neurons//2))
        init.constant_(self.readout_linear_right_alm.bias, 0.0)


    def apply_pert(self, h, left_pert_trial_inds, right_pert_trial_inds):
        '''
        For each trial, we sample drop_p from [drop_p_min, drop_p_max]. Then, sample drop_p fraction of neurons to silence during the stim period.
        '''
        n_trials, n_neurons = h.size()


        '''
        Construct left_pert_mask
        '''
        n_left_pert_trials = len(left_pert_trial_inds)

        left_pert_drop_ps = np.random.uniform(self.drop_p_min, self.drop_p_max, n_left_pert_trials) # (n_left_per_trials)


        left_pert_mask = np.zeros((n_trials, n_neurons), dtype=bool)

        for i in range(n_left_pert_trials):
            cur_drop_p = left_pert_drop_ps[i]
            left_pert_neuron_inds = np.random.permutation(self.n_left_neurons)[:int(self.n_left_neurons*cur_drop_p)]
            left_pert_mask[left_pert_trial_inds[i],self.left_alm_inds[left_pert_neuron_inds]] = True


        '''
        Construct right_pert_mask
        '''
        n_right_pert_trials = len(right_pert_trial_inds)

        right_pert_drop_ps = np.random.uniform(self.drop_p_min, self.drop_p_max, n_right_pert_trials) # (n_right_per_trials)

        right_pert_mask = np.zeros((n_trials, n_neurons), dtype=bool)

        for i in range(n_right_pert_trials):
            cur_drop_p = right_pert_drop_ps[i]
            right_pert_neuron_inds = np.random.permutation(self.n_right_neurons)[:int(self.n_right_neurons*cur_drop_p)]
            right_pert_mask[right_pert_trial_inds[i],self.right_alm_inds[right_pert_neuron_inds]] = True


        # left pertubation
        h[np.nonzero(left_pert_mask)] = 0

        # right pertubation
        h[np.nonzero(right_pert_mask)] = 0




    def forward(self, xs):
        '''
        Input:
        xs: (n_trials, T, 1)

        Output:
        hs: (n_trials, T, n_neurons)
        zs: (n_trials, T, 2)
        '''
        n_trials = xs.size(0)
        T = xs.size(1)
        h_pre = xs.new_zeros(n_trials, self.n_neurons)
        hs = []

        # input noise
        xs_noise_left_alm = math.sqrt(2/self.a)*self.sigma_input_noise*torch.randn_like(xs)
        xs_noise_right_alm = math.sqrt(2/self.a)*self.sigma_input_noise*torch.randn_like(xs)


        # input trial mask
        n_trials = xs.size(0)
        xs_left_alm_mask = (torch.rand(n_trials,1,1) >= self.xs_left_alm_drop_p).float().to(xs.device)  # (n_trials, 1, 1)
        xs_right_alm_mask = (torch.rand(n_trials,1,1) >= self.xs_right_alm_drop_p).float().to(xs.device)  # (n_trials, 1, 1)


        xs_injected_left_alm = self.w_xh_linear_left_alm(xs*xs_left_alm_mask*self.xs_left_alm_amp + xs_noise_left_alm)
        xs_injected_right_alm = self.w_xh_linear_right_alm(xs*xs_right_alm_mask*self.xs_right_alm_amp + xs_noise_right_alm)

        xs_injected = torch.cat([xs_injected_left_alm, xs_injected_right_alm], 2)

        # Determine trials in which we apply uni pert.
        n_trials = xs.size(0)
        pert_trial_inds = np.random.permutation(n_trials)[:int(self.uni_pert_trials_prob*n_trials)]
        left_pert_trial_inds = pert_trial_inds[:int(self.left_alm_pert_prob*len(pert_trial_inds))]
        right_pert_trial_inds = pert_trial_inds[int(self.left_alm_pert_prob*len(pert_trial_inds)):]


        for t in range(T):
            h = self.rnn_cell(xs_injected[:,t], h_pre) # (n_trials, n_neurons)
            

            # Apply perturbation.
            if t >= self.pert_begin and t <= self.pert_end:
                self.apply_pert(h, left_pert_trial_inds, right_pert_trial_inds)


            hs.append(h)
            h_pre = h

        hs = torch.stack(hs, 1)
        
        zs_left_alm = self.readout_linear_left_alm(hs[...,self.left_alm_inds])
        zs_right_alm = self.readout_linear_right_alm(hs[...,self.right_alm_inds])
        zs = torch.cat([zs_left_alm, zs_right_alm], 2)

        return hs, zs



class TwoHemiRNNCellLinear(nn.Module):
    '''
    Same as TwoHemiRNNCellLinear except that we separately store within-hemi and cross-hemi weights, so that
    they can easily trained separately.
    '''

    def __init__(self, n_neurons=128, a=0.2, sigma=0.05, zero_init_cross_hemi=False,
        init_cross_hemi_rel_factor=1):
        super().__init__()
        self.n_neurons = n_neurons
        self.a = a
        self.sigma = sigma
        self.zero_init_cross_hemi = zero_init_cross_hemi
        self.init_cross_hemi_rel_factor = init_cross_hemi_rel_factor

        self.w_hh_linear_ll = nn.Linear(n_neurons//2, n_neurons//2)
        self.w_hh_linear_rr = nn.Linear(n_neurons-n_neurons//2, n_neurons-n_neurons//2)

        self.w_hh_linear_lr = nn.Linear(n_neurons//2, n_neurons-n_neurons//2, bias=False)
        self.w_hh_linear_rl = nn.Linear(n_neurons-n_neurons//2, n_neurons//2, bias=False)


        self.init_params()

    
    def init_params(self):
        init.normal_(self.w_hh_linear_ll.weight, 0.0, 1.0/math.sqrt(self.n_neurons))
        init.normal_(self.w_hh_linear_rr.weight, 0.0, 1.0/math.sqrt(self.n_neurons))

        if self.zero_init_cross_hemi:
            init.constant_(self.w_hh_linear_lr.weight, 0.0)
            init.constant_(self.w_hh_linear_rl.weight, 0.0)

        else:
            init.normal_(self.w_hh_linear_lr.weight, 0.0, self.init_cross_hemi_rel_factor/math.sqrt(self.n_neurons))
            init.normal_(self.w_hh_linear_rl.weight, 0.0, self.init_cross_hemi_rel_factor/math.sqrt(self.n_neurons))


        init.constant_(self.w_hh_linear_ll.bias, 0.0)
        init.constant_(self.w_hh_linear_rr.bias, 0.0)


    def full_recurrent(self, h_pre):
        h_pre_left = h_pre[:,:self.n_neurons//2]
        h_pre_right = h_pre[:,self.n_neurons//2:]

        h1 = torch.cat([self.w_hh_linear_ll(h_pre_left), self.w_hh_linear_lr(h_pre_left)], 1) # (n_neurons)
        h2 = torch.cat([self.w_hh_linear_rl(h_pre_right), self.w_hh_linear_rr(h_pre_right)], 1) # (n_neurons)

        return h1 + h2


    def forward(self, x_injected, h_pre):
        '''
        Input:
        x_injected: (n_trials, n_neurons)
        h_pre: (n_trials, n_neurons)

        Output:
        h: (n_trials, n_neurons)
        '''
        noise = math.sqrt(2/self.a)*self.sigma*torch.randn_like(x_injected)

        h = (1-self.a)*h_pre + self.a*(self.full_recurrent(h_pre) + x_injected + noise)

        return h


















class TwoHemiRNNTanh(nn.Module):

    def __init__(self, configs, a, pert_begin, pert_end, zero_init_cross_hemi=False, return_input=False):
        super().__init__()

        self.return_input = return_input

        self.configs = configs

        self.a = a
        self.pert_begin = pert_begin
        self.pert_end = pert_end
        self.zero_init_cross_hemi = zero_init_cross_hemi
        self.init_cross_hemi_rel_factor = configs['init_cross_hemi_rel_factor']

        self.uni_pert_trials_prob = configs['uni_pert_trials_prob']
        self.left_alm_pert_prob = configs['left_alm_pert_prob']

        # bi stim pert
        self.bi_pert_trials_prob = None

        self.n_neurons = configs['n_neurons']
        self.n_left_neurons = self.n_neurons//2
        self.n_right_neurons = self.n_neurons - self.n_neurons//2

        self.sigma_input_noise = configs['sigma_input_noise']
        self.sigma_rec_noise = configs['sigma_rec_noise']

        # Define left and right ALM
        self.left_alm_inds = np.arange(self.n_neurons//2)
        self.right_alm_inds = np.arange(self.n_neurons//2, self.n_neurons)


        self.rnn_cell = TwoHemiRNNCellGeneral(n_neurons=self.n_neurons, a=self.a, sigma=self.sigma_rec_noise, nonlinearity=nn.Tanh(),
            zero_init_cross_hemi=self.zero_init_cross_hemi, init_cross_hemi_rel_factor=self.init_cross_hemi_rel_factor)
        
        self.w_xh_linear_left_alm = nn.Linear(1, self.n_neurons//2, bias=False)
        self.w_xh_linear_right_alm = nn.Linear(1, self.n_neurons-self.n_neurons//2, bias=False)

        self.readout_linear_left_alm = nn.Linear(self.n_neurons//2, 1)
        self.readout_linear_right_alm = nn.Linear(self.n_neurons-self.n_neurons//2, 1)

        self.init_params()

        self.drop_p_min = configs['drop_p_min']
        self.drop_p_max = configs['drop_p_max']


        self.xs_left_alm_drop_p = configs['xs_left_alm_drop_p']
        self.xs_right_alm_drop_p = configs['xs_right_alm_drop_p']

        self.xs_left_alm_amp = configs['xs_left_alm_amp']
        self.xs_right_alm_amp = configs['xs_right_alm_amp']


    def get_w_hh(self):
        w_hh = torch.zeros((self.n_neurons, self.n_neurons))
        w_hh[:self.n_neurons//2,:self.n_neurons//2] = self.rnn_cell.w_hh_linear_ll.weight
        w_hh[self.n_neurons//2:,self.n_neurons//2:] = self.rnn_cell.w_hh_linear_rr.weight
        w_hh[self.n_neurons//2:,:self.n_neurons//2] = self.rnn_cell.w_hh_linear_lr.weight
        w_hh[:self.n_neurons//2,self.n_neurons//2:] = self.rnn_cell.w_hh_linear_rl.weight

        return w_hh




    def init_params(self):
        init.normal_(self.w_xh_linear_left_alm.weight, 0.0, 1)
        init.normal_(self.w_xh_linear_right_alm.weight, 0.0, 1)


        init.normal_(self.readout_linear_left_alm.weight, 0.0, 1.0/math.sqrt(self.n_neurons//2))
        init.constant_(self.readout_linear_left_alm.bias, 0.0)

        init.normal_(self.readout_linear_right_alm.weight, 0.0, 1.0/math.sqrt(self.n_neurons-self.n_neurons//2))
        init.constant_(self.readout_linear_right_alm.bias, 0.0)


    def apply_pert(self, h, left_pert_trial_inds, right_pert_trial_inds):
        '''
        For each trial, we sample drop_p from [drop_p_min, drop_p_max]. Then, sample drop_p fraction of neurons to silence during the stim period.
        '''
        n_trials, n_neurons = h.size()


        '''
        Construct left_pert_mask
        '''
        n_left_pert_trials = len(left_pert_trial_inds)

        left_pert_drop_ps = np.random.uniform(self.drop_p_min, self.drop_p_max, n_left_pert_trials) # (n_left_per_trials)


        left_pert_mask = np.zeros((n_trials, n_neurons), dtype=bool)

        for i in range(n_left_pert_trials):
            cur_drop_p = left_pert_drop_ps[i]
            left_pert_neuron_inds = np.random.permutation(self.n_left_neurons)[:int(self.n_left_neurons*cur_drop_p)]
            left_pert_mask[left_pert_trial_inds[i],self.left_alm_inds[left_pert_neuron_inds]] = True


        '''
        Construct right_pert_mask
        '''
        n_right_pert_trials = len(right_pert_trial_inds)

        right_pert_drop_ps = np.random.uniform(self.drop_p_min, self.drop_p_max, n_right_pert_trials) # (n_right_per_trials)

        right_pert_mask = np.zeros((n_trials, n_neurons), dtype=bool)

        for i in range(n_right_pert_trials):
            cur_drop_p = right_pert_drop_ps[i]
            right_pert_neuron_inds = np.random.permutation(self.n_right_neurons)[:int(self.n_right_neurons*cur_drop_p)]
            right_pert_mask[right_pert_trial_inds[i],self.right_alm_inds[right_pert_neuron_inds]] = True


        # left pertubation
        h[np.nonzero(left_pert_mask)] = 0

        # right pertubation
        h[np.nonzero(right_pert_mask)] = 0



    def forward(self, xs):
        '''
        Input:
        xs: (n_trials, T, 1)

        Output:
        hs: (n_trials, T, n_neurons)
        zs: (n_trials, T, 2)
        '''
        n_trials = xs.size(0)
        T = xs.size(1)
        h_pre = xs.new_zeros(n_trials, self.n_neurons)
        hs = []

        # input noise
        xs_noise_left_alm = math.sqrt(2/self.a)*self.sigma_input_noise*torch.randn_like(xs)
        xs_noise_right_alm = math.sqrt(2/self.a)*self.sigma_input_noise*torch.randn_like(xs)

        # input trial mask
        n_trials = xs.size(0)
        xs_left_alm_mask = (torch.rand(n_trials,1,1) >= self.xs_left_alm_drop_p).float().to(xs.device)  # (n_trials, 1, 1)
        xs_right_alm_mask = (torch.rand(n_trials,1,1) >= self.xs_right_alm_drop_p).float().to(xs.device)  # (n_trials, 1, 1)


        xs_injected_left_alm = self.w_xh_linear_left_alm(xs*xs_left_alm_mask*self.xs_left_alm_amp + xs_noise_left_alm)
        xs_injected_right_alm = self.w_xh_linear_right_alm(xs*xs_right_alm_mask*self.xs_right_alm_amp + xs_noise_right_alm)

        xs_injected = torch.cat([xs_injected_left_alm, xs_injected_right_alm], 2)

        # Determine trials in which we apply uni pert.
        n_trials = xs.size(0)
        pert_trial_inds = np.random.permutation(n_trials)[:int(self.uni_pert_trials_prob*n_trials)]
        left_pert_trial_inds = pert_trial_inds[:int(self.left_alm_pert_prob*len(pert_trial_inds))]
        right_pert_trial_inds = pert_trial_inds[int(self.left_alm_pert_prob*len(pert_trial_inds)):]

        # Bi stim pert
        if self.bi_pert_trials_prob is not None:
            n_trials = xs.size(0)
            bi_pert_trial_inds = np.random.permutation(n_trials)[:int(self.bi_pert_trials_prob*n_trials)]


        for t in range(T):
            h = self.rnn_cell(xs_injected[:,t], h_pre) # (n_trials, n_neurons)
            

            # Apply perturbation.
            if t >= self.pert_begin and t <= self.pert_end:
                if self.bi_pert_trials_prob is None:
                    self.apply_pert(h, left_pert_trial_inds, right_pert_trial_inds)
                else:
                    self.apply_bi_pert(h, bi_pert_trial_inds)


            hs.append(h)
            h_pre = h

        hs = torch.stack(hs, 1)
        
        zs_left_alm = self.readout_linear_left_alm(hs[...,self.left_alm_inds])
        zs_right_alm = self.readout_linear_right_alm(hs[...,self.right_alm_inds])
        zs = torch.cat([zs_left_alm, zs_right_alm], 2)

        if self.return_input:
            return xs*xs_left_alm_mask*self.xs_left_alm_amp + xs_noise_left_alm, hs, zs
        else:
            return hs, zs




class TwoHemiRNNCellGeneral(nn.Module):
    '''
    Same as TwoHemiRNNCellGeneral except that we separately store within-hemi and cross-hemi weights, so that
    they can easily trained separately.
    '''

    def __init__(self, n_neurons=128, a=0.2, sigma=0.05, nonlinearity=nn.Tanh(), zero_init_cross_hemi=False,\
        init_cross_hemi_rel_factor=1, bias=True):
        super().__init__()
        self.n_neurons = n_neurons
        self.a = a
        self.sigma = sigma

        self.nonlinearity = nonlinearity
        self.zero_init_cross_hemi = zero_init_cross_hemi
        self.init_cross_hemi_rel_factor = init_cross_hemi_rel_factor

        self.bias = bias

        self.w_hh_linear_ll = nn.Linear(n_neurons//2, n_neurons//2, bias=self.bias)
        self.w_hh_linear_rr = nn.Linear(n_neurons-n_neurons//2, n_neurons-n_neurons//2, bias=self.bias)

        self.w_hh_linear_lr = nn.Linear(n_neurons//2, n_neurons-n_neurons//2, bias=False)
        self.w_hh_linear_rl = nn.Linear(n_neurons-n_neurons//2, n_neurons//2, bias=False)


        self.init_params()

    def init_params(self):
        init.normal_(self.w_hh_linear_ll.weight, 0.0, 1.0/math.sqrt(self.n_neurons))
        init.normal_(self.w_hh_linear_rr.weight, 0.0, 1.0/math.sqrt(self.n_neurons))

        if self.zero_init_cross_hemi:
            init.constant_(self.w_hh_linear_lr.weight, 0.0)
            init.constant_(self.w_hh_linear_rl.weight, 0.0)

        else:
            init.normal_(self.w_hh_linear_lr.weight, 0.0, self.init_cross_hemi_rel_factor/math.sqrt(self.n_neurons))
            init.normal_(self.w_hh_linear_rl.weight, 0.0, self.init_cross_hemi_rel_factor/math.sqrt(self.n_neurons))


        if self.bias:
            init.constant_(self.w_hh_linear_ll.bias, 0.0)
            init.constant_(self.w_hh_linear_rr.bias, 0.0)


    def full_recurrent(self, h_pre):
        h_pre_left = h_pre[:,:self.n_neurons//2]
        h_pre_right = h_pre[:,self.n_neurons//2:]

        h1 = torch.cat([self.w_hh_linear_ll(h_pre_left), self.w_hh_linear_lr(h_pre_left)], 1) # (n_neurons)
        h2 = torch.cat([self.w_hh_linear_rl(h_pre_right), self.w_hh_linear_rr(h_pre_right)], 1) # (n_neurons)

        return h1 + h2


    def forward(self, x_injected, h_pre):
        '''
        Input:
        x_injected: (n_trials, n_neurons)
        h_pre: (n_trials, n_neurons)

        Output:
        h: (n_trials, n_neurons)
        '''
        noise = math.sqrt(2/self.a)*self.sigma*torch.randn_like(x_injected)

        if self.nonlinearity is not None:
            h = (1-self.a)*h_pre + self.a*self.nonlinearity(self.full_recurrent(h_pre) + x_injected + noise)
        else:
            h = (1-self.a)*h_pre + self.a*(self.full_recurrent(h_pre) + x_injected + noise)

        return h




