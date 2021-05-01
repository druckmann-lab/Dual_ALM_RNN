import torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as T 
import torch.nn as nn
import itertools
import math

from sklearn.metrics import accuracy_score

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dual_alm_rnn_models import *

import json

import scipy

import contextlib
import io

cat = np.concatenate

from scipy import stats


import random

class DualALMRNNExp(object):

    def __init__(self):

        # Load pred_configs
        with open('dual_alm_rnn_configs.json','r') as read_file:
            self.configs = json.load(read_file)

        # Create directories to save results.
        os.makedirs(self.configs['data_dir'], exist_ok=True)
        os.makedirs(self.configs['logs_dir'], exist_ok=True)
        os.makedirs(self.configs['models_dir'], exist_ok=True)


        self.n_trial_types = 2
        self.n_loc_names = 2
        self.n_loc_names_list = ['left_ALM', 'right_ALM']
        self.loc_name_list = self.n_loc_names_list
        self.n_trial_types_list = range(self.n_trial_types)

        self.n_neurons = self.configs['n_neurons']
        self.neural_unit_location = np.zeros((self.n_neurons,), dtype=object)

        self.neural_unit_location[:self.n_neurons//2] = 'left_ALM'
        self.neural_unit_location[self.n_neurons//2:] = 'right_ALM'

        self.init_exp_setting()
        self.init_sub_path(self.configs['train_type'])


    def init_sub_path(self, train_type):

        self.sub_path = os.path.join(train_type, 'n_neurons_{}_random_seed_{}'.format(self.configs['n_neurons'], self.configs['random_seed']),\
            'n_epochs_{}'.format(self.configs['n_epochs']),\
            'lr_{:.1e}_bs_{}'.format(self.configs['lr'], self.configs['bs']),\
            'sigma_input_noise_{:.2f}_sigma_rec_noise_{:.2f}'.format(self.configs['sigma_input_noise'], self.configs['sigma_rec_noise']),\
            'xs_left_alm_amp_{:.2f}_right_alm_amp_{:.2f}'.format(self.configs['xs_left_alm_amp'], self.configs['xs_right_alm_amp']),\
            'init_cross_hemi_rel_factor_{:.2f}'.format(self.configs['init_cross_hemi_rel_factor']))




    def init_exp_setting(self):


        self.trial_begin_t = -3100 # in ms
        self.sample_begin_t = -3000 # in ms, from the response onset.
        self.delay_begin_t = -1700 # in ms, from the response onset.
        self.total_duration = -self.trial_begin_t

        self.t_step = 25 # in ms
        self.tau = 50 # The neuronal time constant in ms.
        self.a = self.t_step/self.tau


        self.T = self.total_duration//self.t_step + 1
        self.sample_begin = (self.sample_begin_t - self.trial_begin_t)//self.t_step
        self.delay_begin = (self.delay_begin_t - self.trial_begin_t)//self.t_step


        '''
        Uni perturbation
        '''
        self.pert_begin_t = -1700
        self.pert_end_t = -900


        # Perturbation is applied in [pert_begin,pert_end], inclusive at both ends.
        self.pert_begin =  (self.pert_begin_t - self.trial_begin_t)//self.t_step
        self.pert_end = (self.pert_end_t - self.trial_begin_t)//self.t_step



        self.sensory_input_means = np.zeros((self.n_trial_types,))
        self.sensory_input_means[0] = -0.15
        self.sensory_input_means[1] = 0.15

        self.sensory_input_stds = np.zeros((self.n_trial_types,))
        self.sensory_input_stds[0] = 1
        self.sensory_input_stds[1] = 1


        # Convert time from ms to s.

        self.trial_begin_t_in_sec = self.trial_begin_t/1000 
        self.sample_begin_t_in_sec = self.sample_begin_t/1000 
        self.delay_begin_t_in_sec = self.delay_begin_t/1000 
        self.pert_begin_t_in_sec = self.pert_begin_t/1000 
        self.pert_end_t_in_sec = self.pert_end_t/1000 
        self.t_step_in_sec = self.t_step/1000






    '''
    ###
    Dataset generation.
    ###
    '''


    def generate_dataset(self):

        random_seed = self.configs['dataset_random_seed']

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)



        T = self.T
        sample_begin = self.sample_begin
        delay_begin = self.delay_begin

        presample_mask = np.zeros((T,), dtype=bool)
        presample_mask[:sample_begin] = True
        presample_inds = np.arange(0,sample_begin)

        sample_mask = np.zeros((T,), dtype=bool)
        sample_mask[sample_begin:delay_begin] = True
        sample_inds = np.arange(sample_begin,delay_begin)


        delay_mask = np.zeros((T,), dtype=bool)
        delay_mask[delay_begin:] = True
        delay_inds = np.arange(delay_begin,T)


        n_train_trials = 5000
        n_val_trials = 1000
        n_test_trials = 1000

        sensory_input_means = self.sensory_input_means
        sensory_input_stds = self.sensory_input_stds

        '''
        Generate the train set.
        '''

        train_sensory_inputs = np.zeros((n_train_trials, T, 1), dtype=np.float32)
        train_trial_type_labels = np.zeros((n_train_trials,), dtype=int)

        shuffled_inds = np.random.permutation(n_train_trials)
        train_trial_type_labels[shuffled_inds[:n_train_trials//2]] = 1

        for i in range(self.n_trial_types):
            cur_trial_type_inds = np.nonzero(train_trial_type_labels==i)[0]

            gaussian_samples = np.random.randn(len(cur_trial_type_inds), len(sample_inds), 1)

            train_sensory_inputs[np.ix_(cur_trial_type_inds, sample_inds)] = \
            sensory_input_means[i] + sensory_input_stds[i]*gaussian_samples


        '''
        Generate the val set.
        '''

        val_sensory_inputs = np.zeros((n_val_trials, T, 1), dtype=np.float32)
        val_trial_type_labels = np.zeros((n_val_trials,), dtype=int)

        shuffled_inds = np.random.permutation(n_val_trials)
        val_trial_type_labels[shuffled_inds[:n_val_trials//2]] = 1

        for i in range(self.n_trial_types):
            cur_trial_type_inds = np.nonzero(val_trial_type_labels==i)[0]

            gaussian_samples = np.random.randn(len(cur_trial_type_inds), len(sample_inds), 1)

            val_sensory_inputs[np.ix_(cur_trial_type_inds, sample_inds)] = \
            sensory_input_means[i] + sensory_input_stds[i]*gaussian_samples



        '''
        Generate the test set.
        '''

        test_sensory_inputs = np.zeros((n_test_trials, T, 1), dtype=np.float32)
        test_trial_type_labels = np.zeros((n_test_trials,), dtype=int)

        shuffled_inds = np.random.permutation(n_test_trials)
        test_trial_type_labels[shuffled_inds[:n_test_trials//2]] = 1

        for i in range(self.n_trial_types):
            cur_trial_type_inds = np.nonzero(test_trial_type_labels==i)[0]

            gaussian_samples = np.random.randn(len(cur_trial_type_inds), len(sample_inds), 1)

            test_sensory_inputs[np.ix_(cur_trial_type_inds, sample_inds)] = \
            sensory_input_means[i] + sensory_input_stds[i]*gaussian_samples




        '''
        Save.
        '''
        train_save_path = os.path.join(self.configs['data_dir'], 'train')
        os.makedirs(train_save_path, exist_ok=True)
        np.save(os.path.join(train_save_path, 'sensory_inputs.npy'), train_sensory_inputs)
        np.save(os.path.join(train_save_path, 'trial_type_labels.npy'), train_trial_type_labels)

        val_save_path = os.path.join(self.configs['data_dir'], 'val')
        os.makedirs(val_save_path, exist_ok=True)
        np.save(os.path.join(val_save_path, 'sensory_inputs.npy'), val_sensory_inputs)
        np.save(os.path.join(val_save_path, 'trial_type_labels.npy'), val_trial_type_labels)

        test_save_path = os.path.join(self.configs['data_dir'], 'test')
        os.makedirs(test_save_path, exist_ok=True)
        np.save(os.path.join(test_save_path, 'sensory_inputs.npy'), test_sensory_inputs)
        np.save(os.path.join(test_save_path, 'trial_type_labels.npy'), test_trial_type_labels)


        sample_inds = np.random.permutation(n_train_trials)[:10]
        sample_train_inputs = train_sensory_inputs[sample_inds]
        sample_train_labels = train_trial_type_labels[sample_inds]


        '''
        Sanity check.
        '''

        fig = plt.figure()
        ax = fig.add_subplot(111)


        color = ['r', 'b']
        for i in range(2):
            ax.plot(sample_train_inputs[sample_train_labels==i][...,0].T, c=color[i])
        ax.axvline(self.sample_begin, c='k')
        ax.axvline(self.delay_begin, c='k')

        fig.savefig(os.path.join(train_save_path, 'sample.png'))

        plt.show()






    def train_type_uniform(self):

        random_seed = self.configs['random_seed']

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        model_type = self.configs['model_type']

        self.init_sub_path('train_type_uniform')


        model_save_path = os.path.join(self.configs['models_dir'], model_type, self.sub_path)


        logs_save_path = os.path.join(self.configs['logs_dir'], model_type, self.sub_path)

        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(logs_save_path, exist_ok=True)




        # Detect devices
        use_cuda = bool(self.configs['use_cuda'])
        if use_cuda and not torch.cuda.is_available():
            use_cuda = False
        device = torch.device("cuda:{}".format(self.configs['gpu_ids'][0]) if use_cuda else "cpu")

        # Data loading parameters
        if use_cuda:
            params = {'batch_size': self.configs['bs'], 'shuffle': True, 'num_workers': self.configs['num_workers'], \
            'pin_memory': bool(self.configs['pin_memory'])}
        else:
            params = {'batch_size': self.configs['bs'], 'shuffle': True}

        '''
        Load the dataset and wrap it with Pytorch Dataset.
        '''

        # train
        train_save_path = os.path.join(self.configs['data_dir'], 'train')

        train_sensory_inputs = np.load(os.path.join(train_save_path, 'sensory_inputs.npy'))
        train_trial_type_labels = np.load(os.path.join(train_save_path, 'trial_type_labels.npy'))

        train_set = data.TensorDataset(torch.tensor(train_sensory_inputs), torch.tensor(train_trial_type_labels))

        train_loader = data.DataLoader(train_set, **params, drop_last=True)

        # val
        val_save_path = os.path.join(self.configs['data_dir'], 'val')

        val_sensory_inputs = np.load(os.path.join(val_save_path, 'sensory_inputs.npy'))
        val_trial_type_labels = np.load(os.path.join(val_save_path, 'trial_type_labels.npy'))

        val_set = data.TensorDataset(torch.tensor(val_sensory_inputs), torch.tensor(val_trial_type_labels))

        val_loader = data.DataLoader(val_set, **params)



        '''
        Initialize the model.
        '''

        import sys
        model = getattr(sys.modules[__name__], model_type)(self.configs, \
            self.a, self.pert_begin, self.pert_end).to(device)


        '''
        We only train the recurrent weights.
        '''
        trainable_params = []
        for name, param in model.named_parameters():
            if 'rnn_cell' in name:
                trainable_params.append(param)


        optimizer = optim.Adam(trainable_params, lr=self.configs['lr'], weight_decay=self.configs['l2_weight_decay'])

        loss_fct = nn.BCEWithLogitsLoss()




        '''
        Train the model.
        '''



        all_epoch_train_losses = []
        all_epoch_train_scores = []
        all_epoch_val_losses = []
        all_epoch_val_scores = []

        best_val_score = float('-inf')

        for epoch in range(self.configs['n_epochs']):
            epoch_begin_time = time.time()


            model.uni_pert_trials_prob = self.configs['uni_pert_trials_prob']
            
            train_losses, train_scores = self.train_helper(model, device, train_loader, optimizer, epoch, loss_fct) # Per each training batch.


            val_loss, val_score = self.val_helper(model, device, val_loader, loss_fct) # On the entire val set.

            if val_score > best_val_score:
                best_val_score = val_score
                model_save_name = 'best_model.pth'

                torch.save(model.state_dict(), os.path.join(model_save_path, model_save_name))  # save model


            all_epoch_train_losses.extend(train_losses)
            all_epoch_train_scores.extend(train_scores)
            all_epoch_val_losses.append(val_loss)
            all_epoch_val_scores.append(val_score)

            A = np.array(all_epoch_train_losses)
            B = np.array(all_epoch_train_scores)
            C = np.array(all_epoch_val_losses)
            D = np.array(all_epoch_val_scores)

            np.save(os.path.join(logs_save_path, 'all_epoch_train_losses.npy'), A)
            np.save(os.path.join(logs_save_path, 'all_epoch_train_scores.npy'), B)
            np.save(os.path.join(logs_save_path, 'all_epoch_val_losses.npy'), C)
            np.save(os.path.join(logs_save_path, 'all_epoch_val_scores.npy'), D)

            epoch_end_time = time.time()

            print('Epoch {} total time: {:.3f} s'.format(epoch+1, epoch_end_time - epoch_begin_time))
            print('')









    def train_type_within_hemi_only(self):


        random_seed = self.configs['random_seed']

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)


        model_type = self.configs['model_type']


        self.init_sub_path('train_type_within_hemi_only')


        model_save_path = os.path.join(self.configs['models_dir'], model_type, self.sub_path)


        logs_save_path = os.path.join(self.configs['logs_dir'], model_type, self.sub_path)

        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(logs_save_path, exist_ok=True)




        # Detect devices
        use_cuda = bool(self.configs['use_cuda'])
        if use_cuda and not torch.cuda.is_available():
            use_cuda = False
        device = torch.device("cuda:{}".format(self.configs['gpu_ids'][0]) if use_cuda else "cpu")

        # Data loading parameters
        if use_cuda:
            params = {'batch_size': self.configs['bs'], 'shuffle': True, 'num_workers': self.configs['num_workers'], \
            'pin_memory': bool(self.configs['pin_memory'])}
        else:
            params = {'batch_size': self.configs['bs'], 'shuffle': True}

        '''
        Load the dataset and wrap it with Pytorch Dataset.
        '''

        # train
        train_save_path = os.path.join(self.configs['data_dir'], 'train')

        train_sensory_inputs = np.load(os.path.join(train_save_path, 'sensory_inputs.npy'))
        train_trial_type_labels = np.load(os.path.join(train_save_path, 'trial_type_labels.npy'))

        train_set = data.TensorDataset(torch.tensor(train_sensory_inputs), torch.tensor(train_trial_type_labels))

        train_loader = data.DataLoader(train_set, **params, drop_last=True)

        # val
        val_save_path = os.path.join(self.configs['data_dir'], 'val')

        val_sensory_inputs = np.load(os.path.join(val_save_path, 'sensory_inputs.npy'))
        val_trial_type_labels = np.load(os.path.join(val_save_path, 'trial_type_labels.npy'))

        val_set = data.TensorDataset(torch.tensor(val_sensory_inputs), torch.tensor(val_trial_type_labels))

        val_loader = data.DataLoader(val_set, **params)



        '''
        Initialize the model.
        '''

        import sys
        model = getattr(sys.modules[__name__], model_type)(self.configs, \
            self.a, self.pert_begin, self.pert_end).to(device)



        '''
        We only train the recurrent weights.
        '''
        params_within_hemi = []
        params_cross_hemi = []
        n_neurons = self.configs['n_neurons']


        for name, param in model.named_parameters():
            if ('w_hh_linear_ll' in name) or ('w_hh_linear_rr' in name):
                params_within_hemi.append(param)
            elif ('w_hh_linear_lr' in name) or ('w_hh_linear_rl' in name):
                params_cross_hemi.append(param)


        optimizer_within_hemi = optim.Adam(params_within_hemi, lr=self.configs['lr'], weight_decay=self.configs['l2_weight_decay'])


        loss_fct = nn.BCEWithLogitsLoss()




        '''
        Train the model.
        '''



        all_epoch_train_losses = []
        all_epoch_train_scores = []
        all_epoch_val_losses = []
        all_epoch_val_scores = []

        best_val_score = float('-inf')


        for epoch in range(self.configs['n_epochs']):
            epoch_begin_time = time.time()


            print('')
            print('Within-hemi training')

            model.uni_pert_trials_prob = self.configs['uni_pert_trials_prob']

            train_losses, train_scores = self.train_helper(model, device, train_loader, optimizer_within_hemi, epoch, loss_fct) # Per each training batch.


            val_loss, val_score = self.val_helper(model, device, val_loader, loss_fct) # On the entire val set.

            if val_score > best_val_score:
                best_val_score = val_score
                model_save_name = 'best_model.pth'

                torch.save(model.state_dict(), os.path.join(model_save_path, model_save_name))  # save model


            all_epoch_train_losses.extend(train_losses)
            all_epoch_train_scores.extend(train_scores)
            all_epoch_val_losses.append(val_loss)
            all_epoch_val_scores.append(val_score)

            A = np.array(all_epoch_train_losses)
            B = np.array(all_epoch_train_scores)
            C = np.array(all_epoch_val_losses)
            D = np.array(all_epoch_val_scores)

            np.save(os.path.join(logs_save_path, 'all_epoch_train_losses.npy'), A)
            np.save(os.path.join(logs_save_path, 'all_epoch_train_scores.npy'), B)
            np.save(os.path.join(logs_save_path, 'all_epoch_val_losses.npy'), C)
            np.save(os.path.join(logs_save_path, 'all_epoch_val_scores.npy'), D)

            epoch_end_time = time.time()

            print('Epoch {} total time: {:.3f} s'.format(epoch+1, epoch_end_time - epoch_begin_time))
            print('')







    '''
    Add losses randomly after stim period.
    '''
    def train_helper(self, model, device, train_loader, optimizer, epoch, loss_fct):

        model.train()

        losses = []
        scores = []

        trial_count = 0

        begin_time = time.time()
        for batch_idx, data in enumerate(train_loader):

            inputs, labels  = data
            inputs, labels = inputs.to(device), labels.to(device)

            trial_count += len(labels)

            optimizer.zero_grad()


            '''
            hs: (n_trials, T, n_neurons)
            zs: (n_trials, T, 2) # 2 because we have a readout at each hemisphere.
            '''
            _, zs = model(inputs)


            assert self.T == inputs.shape[1]

            dec_begin = self.delay_begin            


            # BCEWithLogitsLoss requires that the target be float between 0 and 1.
            loss_left_alm = loss_fct(zs[:,dec_begin:,0], labels.float()[:,None].expand(-1,self.T-dec_begin))
            loss_right_alm = loss_fct(zs[:,dec_begin:,1], labels.float()[:,None].expand(-1,self.T-dec_begin))


            loss = loss_left_alm + loss_right_alm


            loss.backward()

            optimizer.step()


            # Evaluate the score.
            preds_left_alm = (zs[:,-1,0] >= 0).long()
            preds_right_alm = (zs[:,-1,1] >= 0).long()

            score_left_alm = accuracy_score(labels.cpu().data.numpy(), preds_left_alm.cpu().data.numpy())
            score_right_alm = accuracy_score(labels.cpu().data.numpy(), preds_right_alm.cpu().data.numpy())

            score = (score_left_alm+score_right_alm)/2

            losses.append(loss)
            scores.append(score)

            if (batch_idx + 1) % self.configs['log_interval'] == 0:
                cur_time = time.time()
                print('Train Epoch: {} [{}/{} ({:.0f}%)] loss: {:.6f}, fraction correct: {:.1f}% ({:.3f} s)'.format(
                    epoch + 1, trial_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), \
                    loss.item(), 100. * score, cur_time - begin_time))
                begin_time = time.time()


        return losses, scores




    def val_helper(self, model, device, val_loader, loss_fct):

        model.eval()

        total_loss = 0
        total_score = 0

        trial_count = 0

        begin_time = time.time()
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):

                inputs, labels  = data
                inputs, labels = inputs.to(device), labels.to(device)

                trial_count += len(labels)

                '''
                hs: (n_trials, T, n_neurons)
                zs: (n_trials, T, 2)
                '''
                _, zs = model(inputs)

                loss_left_alm = loss_fct(zs[:,-1,0], labels.float()).item()*len(labels) # BCEWithLogitsLoss requires that the target be float between 0 and 1.
                loss_right_alm = loss_fct(zs[:,-1,1], labels.float()).item()*len(labels) # BCEWithLogitsLoss requires that the target be float between 0 and 1.

                loss = loss_left_alm + loss_right_alm


                total_loss += loss


                # Evaluate the score.


                preds_left_alm = (zs[:,-1,0] >= 0).long()
                preds_right_alm = (zs[:,-1,1] >= 0).long()

                n_correct_left_alm = accuracy_score(labels.cpu().data.numpy(), preds_left_alm.cpu().data.numpy(), normalize=False)
                n_correct_right_alm = accuracy_score(labels.cpu().data.numpy(), preds_right_alm.cpu().data.numpy(), normalize=False)

                total_score += (n_correct_left_alm+n_correct_right_alm)/2


        total_loss /= trial_count
        total_score /= trial_count

        cur_time = time.time()

        print('Test set ({:d} samples): loss: {:.4f}, fraction correct: {:.1f}% ({:.3f} s)'.format(trial_count, total_loss, \
            100. * total_score, cur_time - begin_time))


        return total_loss, total_score










    def plot_cd_traces(self):
        '''
        Main differences from plot_cd_traces:
        1. Add light blue time window for pert period.
        '''

        train_type_str = self.configs['train_type']
        init_cross_hemi_rel_factor = self.configs['init_cross_hemi_rel_factor']
        random_seed = self.configs['random_seed']     


        model_type = self.configs['model_type']
        
        uni_pert_trials_prob = self.configs['uni_pert_trials_prob']


        test_random_seed = self.configs['test_random_seed']

        np.random.seed(test_random_seed)
        torch.manual_seed(test_random_seed)

        self.init_sub_path(train_type_str)



        # Detect devices
        use_cuda = bool(self.configs['use_cuda'])
        if use_cuda and not torch.cuda.is_available():
            use_cuda = False
        device = torch.device("cuda:{}".format(self.configs['gpu_ids'][0]) if use_cuda else "cpu")

        # Data loading parameters
        if use_cuda:
            params = {'batch_size': self.configs['bs'], 'shuffle': False, 'num_workers': self.configs['num_workers'], \
            'pin_memory': bool(self.configs['pin_memory'])}
        else:
            params = {'batch_size': self.configs['bs'], 'shuffle': False}

        '''
        Load the dataset and wrap it with Pytorch Dataset.
        '''

        # test

        test_save_path = os.path.join(self.configs['data_dir'], 'test')

        test_sensory_inputs = np.load(os.path.join(test_save_path, 'sensory_inputs.npy'))
        test_trial_type_labels = np.load(os.path.join(test_save_path, 'trial_type_labels.npy'))

        test_set = data.TensorDataset(torch.tensor(test_sensory_inputs), torch.tensor(test_trial_type_labels))

        test_loader = data.DataLoader(test_set, **params)


        '''
        Load the saved model.
        '''

        import sys
        model = getattr(sys.modules[__name__], model_type)(self.configs, \
            self.a, self.pert_begin, self.pert_end).to(device)

        model_save_path = os.path.join(self.configs['models_dir'], model_type, self.sub_path)

        state_dict = torch.load(os.path.join(model_save_path, 'best_model.pth'))

        model.load_state_dict(state_dict)


        # Unless otherwise specified, we set drop_p_min and max = 1
        model.drop_p_min = 1.0
        model.drop_p_max = 1.0


        '''
        noise.
        '''
        model.sigma_input_noise = self.configs['test_sigma_input_noise']
        model.rnn_cell.sigma = self.configs['test_sigma_rec_noise']


        '''
        Compute cd proj
        '''
        # train
        train_save_path = os.path.join(self.configs['data_dir'], 'train')

        train_sensory_inputs = np.load(os.path.join(train_save_path, 'sensory_inputs.npy'))
        train_trial_type_labels = np.load(os.path.join(train_save_path, 'trial_type_labels.npy'))

        train_set = data.TensorDataset(torch.tensor(train_sensory_inputs), torch.tensor(train_trial_type_labels))

        train_loader = data.DataLoader(train_set, **params, drop_last=True)



        cds = self.get_cds(model, device, train_loader, model_type) # cds[j] = (T, n_neurons in a given hemi)

        old_cds = cds

        # Average cd over the delay period.
        cds = np.zeros((self.n_loc_names,), dtype=object)
        for j in range(self.n_loc_names):
            cds[j] = old_cds[j][self.delay_begin:].mean(0)
            cds[j] = cds[j]/np.linalg.norm(cds[j]) # (n_neurons in a given hemi)

        cd_dbs = self.get_cd_dbs(cds, model, device, train_loader, model_type) # (n_loc_names)


        # Control trials
        model.uni_pert_trials_prob = 0
        no_stim_hs, no_stim_labels = self.get_neurons_trace(model, device, test_loader, model_type, hemi_type='all')


        # left_stim trials
        model.uni_pert_trials_prob = 1
        model.left_alm_pert_prob = 1

        left_stim_hs, left_stim_labels = self.get_neurons_trace(model, device, test_loader, model_type, hemi_type='all')


        # right_stim trials
        model.uni_pert_trials_prob = 1
        model.left_alm_pert_prob = 0

        right_stim_hs, right_stim_labels = self.get_neurons_trace(model, device, test_loader, model_type, hemi_type='all')


        n_neurons = no_stim_hs.shape[2]

        no_stim_cd_projs = np.zeros((self.n_loc_names,), dtype=object)
        left_stim_cd_projs = np.zeros((self.n_loc_names,), dtype=object)
        right_stim_cd_projs = np.zeros((self.n_loc_names,), dtype=object)

        for j in range(self.n_loc_names):
            if j == 0:
                no_stim_cd_projs[j] = no_stim_hs[...,:n_neurons//2].dot(cds[j]) # (n_trials, T)
                left_stim_cd_projs[j] = left_stim_hs[...,:n_neurons//2].dot(cds[j]) # (n_trials, T)
                right_stim_cd_projs[j] = right_stim_hs[...,:n_neurons//2].dot(cds[j]) # (n_trials, T)

            elif j == 1:
                no_stim_cd_projs[j] = no_stim_hs[...,n_neurons//2:].dot(cds[j]) # (n_trials, T)
                left_stim_cd_projs[j] = left_stim_hs[...,n_neurons//2:].dot(cds[j]) # (n_trials, T)
                right_stim_cd_projs[j] = right_stim_hs[...,n_neurons//2:].dot(cds[j]) # (n_trials, T)

            # Center by db.
            no_stim_cd_projs[j] = no_stim_cd_projs[j] - cd_dbs[j]
            left_stim_cd_projs[j] = left_stim_cd_projs[j] - cd_dbs[j]
            right_stim_cd_projs[j] = right_stim_cd_projs[j] - cd_dbs[j]


        '''
        Plot
        '''


        import mimic_alpha as ma
        alpha = 0.3
        alpha_r = ma.colorAlpha_to_rgb('r', alpha=alpha)
        alpha_b = ma.colorAlpha_to_rgb('b', alpha=alpha)

        from matplotlib import colors
        skyblue_rgb = colors.to_rgb('skyblue') # Directly inputting skyblue in the below line didn't work.
        alpha_skyblue = ma.colorAlpha_to_rgb(skyblue_rgb, alpha=0.5)[0] # [0] because the output is like [array([r, g, b])].

        fig = plt.figure(figsize=(15,15))
        gs = gridspec.GridSpec(2,2, wspace=0.4, hspace=0.4)

        T = no_stim_cd_projs[0].shape[1]


        timesteps = self.trial_begin_t_in_sec + self.t_step_in_sec*np.arange(T)

        for j in range(self.n_loc_names):
            for k in range(2):
                ax = fig.add_subplot(gs[j,k])

                # lick left
                ax.plot(timesteps, no_stim_cd_projs[j][no_stim_labels==0].mean(0), color='r', ls='--', lw=5)
                if k == 0:
                    ax.plot(timesteps, left_stim_cd_projs[j][left_stim_labels==0].mean(0), color='r', ls='-', lw=5)
                    ax.fill_between(timesteps, left_stim_cd_projs[j][left_stim_labels==0].mean(0) - left_stim_cd_projs[j][left_stim_labels==0].std(0),\
                        left_stim_cd_projs[j][left_stim_labels==0].mean(0) + left_stim_cd_projs[j][left_stim_labels==0].std(0), color=alpha_r)
                else:
                    ax.plot(timesteps, right_stim_cd_projs[j][right_stim_labels==0].mean(0), color='r', ls='-', lw=5)
                    ax.fill_between(timesteps, right_stim_cd_projs[j][right_stim_labels==0].mean(0) - right_stim_cd_projs[j][right_stim_labels==0].std(0),\
                        right_stim_cd_projs[j][right_stim_labels==0].mean(0) + right_stim_cd_projs[j][right_stim_labels==0].std(0), color=alpha_r)


                # lick right
                ax.plot(timesteps, no_stim_cd_projs[j][no_stim_labels==1].mean(0), color='b', ls='--', lw=5)
                if k == 0:
                    ax.plot(timesteps, left_stim_cd_projs[j][left_stim_labels==1].mean(0), color='b', ls='-', lw=5)
                    ax.fill_between(timesteps, left_stim_cd_projs[j][left_stim_labels==1].mean(0) - left_stim_cd_projs[j][left_stim_labels==1].std(0),\
                        left_stim_cd_projs[j][left_stim_labels==1].mean(0) + left_stim_cd_projs[j][left_stim_labels==1].std(0), color=alpha_b)

                else:
                    ax.plot(timesteps, right_stim_cd_projs[j][right_stim_labels==1].mean(0), color='b', ls='-', lw=5)
                    ax.fill_between(timesteps, right_stim_cd_projs[j][right_stim_labels==1].mean(0) - right_stim_cd_projs[j][right_stim_labels==1].std(0),\
                        right_stim_cd_projs[j][right_stim_labels==1].mean(0) + right_stim_cd_projs[j][right_stim_labels==1].std(0), color=alpha_b)

                # Find y max, y min values for y_lim and yticks.
                y_agg = cat([left_stim_cd_projs[j][left_stim_labels==0].mean(0) + left_stim_cd_projs[j][left_stim_labels==0].std(0),
                    left_stim_cd_projs[j][left_stim_labels==0].mean(0) - left_stim_cd_projs[j][left_stim_labels==0].std(0),
                    right_stim_cd_projs[j][right_stim_labels==0].mean(0) + right_stim_cd_projs[j][right_stim_labels==0].std(0),
                    right_stim_cd_projs[j][right_stim_labels==0].mean(0) - right_stim_cd_projs[j][right_stim_labels==0].std(0),
                    left_stim_cd_projs[j][left_stim_labels==1].mean(0) + left_stim_cd_projs[j][left_stim_labels==1].std(0),
                    left_stim_cd_projs[j][left_stim_labels==1].mean(0) - left_stim_cd_projs[j][left_stim_labels==1].std(0),
                    right_stim_cd_projs[j][right_stim_labels==1].mean(0) + right_stim_cd_projs[j][right_stim_labels==1].std(0),
                    right_stim_cd_projs[j][right_stim_labels==1].mean(0) - right_stim_cd_projs[j][right_stim_labels==1].std(0),
                    ], 0)

                y_abs_max = np.max(np.abs(y_agg))    



                ax.axvline(self.sample_begin_t_in_sec, ls='--', color='k', lw=2)
                ax.axvline(self.delay_begin_t_in_sec, ls='--', color='k', lw=2)

                ax.axvspan(self.pert_begin_t_in_sec, self.pert_end_t_in_sec, color=alpha_skyblue, zorder=-10)

                # spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                ax.spines['bottom'].set_linewidth(2)
                ax.spines['left'].set_linewidth(2)

                # ticks
                ax.tick_params(length=4, width=2, labelsize=30)

                ax.set_xticks([-3, -2, -1, 0])
                ax.set_yticks([-np.rint(y_abs_max), 0, np.rint(y_abs_max)])

                ax.set_xlabel('Time from movement (s)', fontsize=30)
                ax.set_ylabel('CD projection', fontsize=30)



        if self.configs['sigma_input_noise'] == self.configs['test_sigma_input_noise'] and \
        self.configs['sigma_rec_noise'] == self.configs['test_sigma_rec_noise']:
            noise_str = 'test_sigma_input_noise_{:.2f}_sigma_rec_noise_{:.2f}'.format(model.sigma_input_noise, model.rnn_cell.sigma)

        else:
            noise_str = \
            'train_sigma_input_noise_{:.2f}_sigma_rec_noise_{:.2f}_test_sigma_input_noise_{:.2f}_sigma_rec_noise_{:.2f}'.format(\
                self.configs['sigma_input_noise'], self.configs['sigma_rec_noise'], model.sigma_input_noise, model.rnn_cell.sigma)





        fig_save_path = os.path.join(self.configs['plots_dir'], 'plot_cd_traces', train_type_str, \
            'init_cross_hemi_rel_factor_{:.2f}'.format(init_cross_hemi_rel_factor),\
            noise_str, \
            'random_seed_{}'.format(random_seed), \
            'xs_left_alm_amp_{:.2f}_right_alm_amp_{:.2f}'.format(self.configs['xs_left_alm_amp'], self.configs['xs_right_alm_amp']))
        os.makedirs(fig_save_path, exist_ok=True)

        fig.savefig(os.path.join(fig_save_path, 'plot_cd_traces_model_type_{}.png'.format(model_type)))        
        fig.savefig(os.path.join(fig_save_path, 'plot_cd_traces_model_type_{}.svg'.format(model_type)))        




    def get_cds(self, model, device, loader, model_type, recompute=True):
        '''
        Note added: we only compute cd using correct trials.

        Return
        cds: (n_loc_names,) cds[j] is a numpy array of shape (T, n_neurons in hemi j).
        '''

        save_path = os.path.join(self.configs['results_dir'], 'misc', 'get_cds', model_type,\
            self.sub_path)


        os.makedirs(save_path, exist_ok=True)

        if not recompute and os.path.isfile(os.path.join(save_path, 'cds.npy')):
            cds = np.load(os.path.join(save_path, 'cds.npy'))
            return cds 

        else:
            model.uni_pert_trials_prob = 0
            model.left_alm_pert_prob = 1

            control_hs, control_labels, control_pred_labels = self.get_neurons_trace(model, device, loader, model_type, hemi_type='all', return_pred_labels=True) # np array (n_trials, T, n_neurons), (n_trials)

            control_suc_labels = (control_labels==control_pred_labels).astype(int)

            lick_right_avg = control_hs[(control_labels==1)*(control_suc_labels==1)].mean(0) # (T, n_neurons)
            lick_left_avg = control_hs[(control_labels==0)*(control_suc_labels==1)].mean(0) # (T, n_neurons)

            cd_raw = lick_right_avg - lick_left_avg
            # Now we separate left and right ALM.
            n_loc_names = 2
            n_neurons = cd_raw.shape[1]
            cds = np.zeros((n_loc_names,), dtype=object)

            for j in range(n_loc_names):
                if j == 0:
                    cur_cd = cd_raw[:,:n_neurons//2]
                else:
                    cur_cd = cd_raw[:,n_neurons//2:]

                cds[j] = cur_cd/np.linalg.norm(cur_cd, axis=1, keepdims=True) # (T, n_neurons in a given hemi)

            if not recompute:
                np.save(os.path.join(save_path, 'cds.npy'), cds)

            return cds




    def get_cd_dbs(self, cds, model, device, loader, model_type, recompute=True):
        '''
        Return
        cd_dbs: (n_loc_names,) cd_dbs[j] is a number.
        '''

        save_path = os.path.join(self.configs['results_dir'], 'misc', 'get_cd_dbs', model_type,\
            self.sub_path)


        os.makedirs(save_path, exist_ok=True)

        if not recompute and os.path.isfile(os.path.join(save_path, 'cd_dbs.npy')):
            cd_dbs = np.load(os.path.join(save_path, 'cd_dbs.npy'))
            return cd_dbs 

        else:
            model.uni_pert_trials_prob = 0
            model.left_alm_pert_prob = 1

            control_hs, control_labels, control_pred_labels = self.get_neurons_trace(model, device, loader, model_type, hemi_type='all', return_pred_labels=True) # np array (n_trials, T, n_neurons), (n_trials)

            control_suc_labels = (control_labels==control_pred_labels).astype(int)

            n_loc_names = 2
            n_neurons = control_hs.shape[2]

            # Take last time bin.
            lick_left_h = control_hs[control_labels==0][:,-1] # (n_trials of i, n_neurons)
            lick_right_h = control_hs[control_labels==1][:,-1] # (n_trials of i, n_neurons)

            cd_dbs = np.zeros((n_loc_names,), dtype=object)

            for j in range(n_loc_names):
                cur_cd = cds[j] # (n_neurons//2)

                if j == 0:
                    cur_lick_left_cd_proj = lick_left_h[:,:n_neurons//2].dot(cur_cd) # (n_trials of i)
                    cur_lick_right_cd_proj = lick_right_h[:,:n_neurons//2].dot(cur_cd)
                else:
                    cur_lick_left_cd_proj = lick_left_h[:,n_neurons//2:].dot(cur_cd)
                    cur_lick_right_cd_proj = lick_right_h[:,n_neurons//2:].dot(cur_cd)

                cur_lick_left_avg = cur_lick_left_cd_proj.mean()
                cur_lick_left_var = np.var(cur_lick_left_cd_proj, ddof=1)

                cur_lick_right_avg = cur_lick_right_cd_proj.mean()
                cur_lick_right_var = np.var(cur_lick_right_cd_proj, ddof=1)

                cd_dbs[j] = (cur_lick_left_avg/cur_lick_left_var + cur_lick_right_avg/cur_lick_right_var)/(1/cur_lick_left_var+1/cur_lick_right_var)


            if not recompute:
                np.save(os.path.join(save_path, 'cd_dbs.npy'), cd_dbs)

            return cd_dbs









    def get_neurons_trace(self, model, device, loader, model_type, hemi_type='left_ALM', recompute=False, return_pred_labels=False):
        '''
        Return:
        numpy arrays
        hs: (n_trials, T, 1)
        labels: (n_trials)
        '''
        random_seed = self.configs['test_random_seed']

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)


        total_hs = []
        total_labels = []

        model.eval()

        trial_count = 0

        if return_pred_labels:
            total_pred_labels = []

        begin_time = time.time()
        with torch.no_grad():
            for batch_idx, data in enumerate(loader):

                inputs, labels  = data
                inputs, labels = inputs.to(device), labels.to(device)

                trial_count += len(labels)

                '''
                hs: (n_trials, T, n_neurons)
                zs: (n_trials, T, 2)
                '''

                hs, zs = model(inputs)


                n_neurons = hs.shape[2]
                
                if hemi_type == 'left_ALM':
                    hs = hs[...,:n_neurons//2]
                elif hemi_type == 'right_ALM':
                    hs = hs[...,n_neurons//2:]
                elif hemi_type == 'all':
                    pass


                total_hs.append(hs.cpu().data.numpy())
                total_labels.append(labels.cpu().data.numpy())

                if return_pred_labels:
                    preds_left_alm = (zs[:,-1,0] >= 0).long().cpu().data.numpy()
                    preds_right_alm = (zs[:,-1,1] >= 0).long().cpu().data.numpy()

                    # We take only trials for which both hemispheres agree in pred_labels. The other trials have value of -1.
                    agree_mask = (preds_left_alm == preds_right_alm)
                    pred_labels = np.zeros_like(preds_left_alm)
                    pred_labels[agree_mask] = preds_left_alm[agree_mask]
                    pred_labels[~agree_mask] = -1

                    total_pred_labels.append(pred_labels)


        total_hs = cat(total_hs, 0)
        total_labels = cat(total_labels, 0)

        if return_pred_labels:
            total_pred_labels = cat(total_pred_labels, 0)

        if return_pred_labels:
            return total_hs, total_labels, total_pred_labels

        else:
            return total_hs, total_labels










