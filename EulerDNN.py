"""
@author: LXA
 Date: 2022 年 9 月 10 日
"""
import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib
import platform
import shutil
import time

import dataUtils
import DNN_Class_base
import saveData
import plotData
import logUtils


class EulerDNN(object):
    def __init__(self, input_dim=4, out_dim=1, hidden_layer=None, Model_name='DNN', name2actIn='relu',
                 name2actHidden='relu', name2actOut='linear', opt2regular_WB='L2', type2numeric='float32',
                 factor2freq=None, sFourier=1.0):
        super(EulerDNN, self).__init__()
        if 'DNN' == str.upper(Model_name):
            self.DNN = DNN_Class_base.Pure_Dense_Net(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric)
        elif 'SCALE_DNN' == str.upper(Model_name) or 'DNN_SCALE' == str.upper(Model_name):
            self.DNN = DNN_Class_base.Dense_ScaleNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric)
        elif 'FOURIER_DNN' == str.upper(Model_name) or 'DNN_FOURIERBASE' == str.upper(Model_name):
            self.DNN = DNN_Class_base.Dense_FourierNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric)

        if type2numeric == 'float32':
            self.float_type = tf.float32
        elif type2numeric == 'float64':
            self.float_type = tf.float64
        elif type2numeric == 'float16':
            self.float_type = tf.float16

        self.input_dim = input_dim
        self.factor2freq = factor2freq
        self.sFourier = sFourier
        self.opt2regular_WB = opt2regular_WB

    # 拟 Euler 迭代方法(左边减右边格式)
    def Quasi_Euler(self, t=None,            # shape(input_size, 1)
                   input_size=100,
                   s_obs=None,               # shape(input_size, 1)
                   i_obs=None,               # shape(input_size, 1)
                   r_obs=None,               # shape(input_size, 1)
                   d_obs=None,               # shape(input_size, 1)
                   loss_type='l2_loss',
                   scale2lncosh=0.5):
        assert (t is not None)
        assert (s_obs is not None)
        assert (i_obs is not None)
        assert (r_obs is not None)
        assert (d_obs is not None)

        shape2t = t.get_shape().as_list()
        lenght2t_shape = len(shape2t)
        assert (lenght2t_shape == 2)
        assert (shape2t[-1] == 1)

        ParamsNN = self.DNN(t, scale=self.factor2freq, sFourier=self.sFourier)

        AI = tf.eye(input_size, dtype=tf.float32) * (-2)
        Ones_mat = tf.ones([input_size, input_size], dtype=tf.float32)
        A_diag = tf.linalg.band_part(Ones_mat, 0, 1)
        Amat = AI + A_diag

        dS2dt_obs = tf.matmul(Amat[0:-1, :], s_obs)
        dI2dt_obs = tf.matmul(Amat[0:-1, :], i_obs)
        dR2dt_obs = tf.matmul(Amat[0:-1, :], r_obs)
        dD2dt_obs = tf.matmul(Amat[0:-1, :], d_obs)

        nn_beta = tf.reshape(ParamsNN[:, 0], shape=[-1, 1])
        nn_gamma = tf.reshape(ParamsNN[:, 1], shape=[-1, 1])
        nn_mu = tf.reshape(ParamsNN[:, 2], shape=[-1, 1])

        nn_s2t = -nn_beta[0:-1, 0] * s_obs[0:-1, 0] * i_obs[0:-1, 0] / (s_obs[0:-1, 0] + i_obs[0:-1, 0])
        nn_i2t = nn_beta[0:-1, 0] * s_obs[0:-1, 0] * i_obs[0:-1, 0] / (s_obs[0:-1, 0] + i_obs[0:-1, 0]) - \
                    nn_gamma[0:-1, 0] * i_obs[0:-1, 0] - nn_mu[0:-1, 0] * i_obs[0:-1, 0]
        nn_r2t = nn_gamma[0:-1, 0] * i_obs[0:-1, 0]
        nn_d2t = nn_mu[0:-1, 0] * i_obs[0:-1, 0]

        diff2S = dS2dt_obs - tf.reshape(nn_s2t, shape=[-1, 1])
        diff2I = dI2dt_obs - tf.reshape(nn_i2t, shape=[-1, 1])
        diff2R = dR2dt_obs - tf.reshape(nn_r2t, shape=[-1, 1])
        diff2D = dD2dt_obs - tf.reshape(nn_d2t, shape=[-1, 1])
        if str.lower(loss_type) == 'l2_loss':
            Loss2S = tf.reduce_mean(tf.square(diff2S))
            Loss2I = tf.reduce_mean(tf.square(diff2I))
            Loss2R = tf.reduce_mean(tf.square(diff2R))
            Loss2D = tf.reduce_mean(tf.square(diff2D))
        elif str.lower(loss_type) == 'lncosh_loss':
            Loss2S = (1 / scale2lncosh) * tf.reduce_mean(tf.log(tf.cosh(scale2lncosh * diff2S)))
            Loss2I = (1 / scale2lncosh) * tf.reduce_mean(tf.log(tf.cosh(scale2lncosh * diff2I)))
            Loss2R = (1 / scale2lncosh) * tf.reduce_mean(tf.log(tf.cosh(scale2lncosh * diff2R)))
            Loss2D = (1 / scale2lncosh) * tf.reduce_mean(tf.log(tf.cosh(scale2lncosh * diff2D)))

        return ParamsNN, Loss2S, Loss2I, Loss2R, Loss2D

    # 向前 Euler 迭代方法(纯正的向前迭代格式)
    def Forward_Euler(self, t=None,          # shape(input_size, 1)
                      input_size=100,
                      s_obs=None,            # shape(input_size, 1)
                      i_obs=None,            # shape(input_size, 1)
                      r_obs=None,            # shape(input_size, 1)
                      d_obs=None,            # shape(input_size, 1)
                      loss_type='L2_loss',
                      scale2lncosh=0.5):
        assert (t is not None)
        assert (s_obs is not None)
        assert (i_obs is not None)
        assert (r_obs is not None)
        assert (d_obs is not None)

        shape2t = t.get_shape().as_list()
        lenght2t_shape = len(shape2t)
        assert (lenght2t_shape == 2)
        assert (shape2t[-1] == 1)

        ParamsNN = self.DNN(t, scale=self.factor2freq, sFourier=self.sFourier)     # shape(input_size, 3)

        nn_beta = tf.reshape(ParamsNN[:, 0], shape=[-1, 1])
        nn_gamma = tf.reshape(ParamsNN[:, 1], shape=[-1, 1])
        nn_mu = tf.reshape(ParamsNN[:, 2], shape=[-1, 1])

        # s_obs[0:-1, 0] 这样得到的数据形状为(size-1,) 并不是 (size-1, 1), 即使 s_obs 的形状为(size, 1)
        s_pre = s_obs[0:-1, 0] - nn_beta[0:-1, 0] * s_obs[0:-1, 0] * i_obs[0:-1, 0] / (s_obs[0:-1, 0] + i_obs[0:-1, 0])
        i_pre = i_obs[0:-1, 0] + nn_beta[0:-1, 0] * s_obs[0:-1, 0] * i_obs[0:-1, 0] / (s_obs[0:-1, 0] + i_obs[0:-1, 0]) \
                - nn_gamma[0:-1, 0] * i_obs[0:-1, 0] - nn_mu[0:-1, 0] * i_obs[0:-1, 0]
        r_pre = r_obs[0:-1, 0] + nn_gamma[0:-1, 0] * i_obs[0:-1, 0]
        d_pre = d_obs[0:-1, 0] + nn_mu[0:-1, 0] * i_obs[0:-1, 0]

        diff2S = tf.reshape(s_obs[1: input_size], shape=[-1, 1]) - tf.reshape(s_pre, shape=[-1, 1])
        diff2I = tf.reshape(i_obs[1: input_size], shape=[-1, 1]) - tf.reshape(i_pre, shape=[-1, 1])
        diff2R = tf.reshape(r_obs[1: input_size], shape=[-1, 1]) - tf.reshape(r_pre, shape=[-1, 1])
        diff2D = tf.reshape(d_obs[1: input_size], shape=[-1, 1]) - tf.reshape(d_pre, shape=[-1, 1])

        if str.lower(loss_type) == 'l2_loss':
            Loss2S = tf.reduce_mean(tf.square(diff2S))
            Loss2I = tf.reduce_mean(tf.square(diff2I))
            Loss2R = tf.reduce_mean(tf.square(diff2R))
            Loss2D = tf.reduce_mean(tf.square(diff2D))
        elif str.lower(loss_type) == 'lncosh_loss':
            Loss2S = (1 / scale2lncosh) * tf.reduce_mean(tf.log(tf.cosh(scale2lncosh * diff2S)))
            Loss2I = (1 / scale2lncosh) * tf.reduce_mean(tf.log(tf.cosh(scale2lncosh * diff2I)))
            Loss2R = (1 / scale2lncosh) * tf.reduce_mean(tf.log(tf.cosh(scale2lncosh * diff2R)))
            Loss2D = (1 / scale2lncosh) * tf.reduce_mean(tf.log(tf.cosh(scale2lncosh * diff2D)))

        return ParamsNN, Loss2S, Loss2I, Loss2R, Loss2D

    # 修正的 Euler 迭代方法
    def Modify_Euler(self, t=None,           # shape(input_size, 1)
                     input_size=100,
                     s_obs=None,             # shape(input_size, 1)
                     i_obs=None,             # shape(input_size, 1)
                     r_obs=None,             # shape(input_size, 1)
                     d_obs=None,             # shape(input_size, 1)
                     loss_type='l2_loss',
                     scale2lncosh=0.5):
        assert (t is not None)
        assert (s_obs is not None)
        assert (i_obs is not None)
        assert (r_obs is not None)
        assert (d_obs is not None)

        shape2t = t.get_shape().as_list()
        lenght2t_shape = len(shape2t)
        assert (lenght2t_shape == 2)
        assert (shape2t[-1] == 1)

        ParamsNN = self.DNN(t, scale=self.factor2freq, sFourier=self.sFourier)

        nn_beta = tf.reshape(ParamsNN[:, 0], shape=[-1, 1])
        nn_gamma = tf.reshape(ParamsNN[:, 1], shape=[-1, 1])
        nn_mu = tf.reshape(ParamsNN[:, 2], shape=[-1, 1])

        # s_obs[0:-1, 0] 这样得到的数据形状为(size-1,) 并不是 (size-1, 1), 即使 s_obs 的形状为(size, 1)
        s_bar = s_obs[0:-1, 0] - nn_beta[0:-1, 0] * s_obs[0:-1, 0] * i_obs[0:-1, 0] / (s_obs[0:-1, 0] + i_obs[0:-1, 0])
        i_bar = i_obs[0:-1, 0] + nn_beta[0:-1, 0] * s_obs[0:-1, 0] * i_obs[0:-1, 0] / (s_obs[0:-1, 0] + i_obs[0:-1, 0]) \
                - nn_gamma[0:-1, 0] * i_obs[0:-1, 0] - nn_mu[0:-1, 0] * i_obs[0:-1, 0]
        r_bar = r_obs[0:-1, 0] + nn_gamma[0:-1, 0] * i_obs[0:-1, 0]
        d_bar = d_obs[0:-1, 0] + nn_mu[0:-1, 0] * i_obs[0:-1, 0]

        s_bar = tf.reshape(s_bar, shape=[-1, 1])
        i_bar = tf.reshape(i_bar, shape=[-1, 1])
        r_bar = tf.reshape(r_bar, shape=[-1, 1])
        d_bar = tf.reshape(d_bar, shape=[-1, 1])

        # s_bar[:, 0] 这样得到的数据形状为(size,) 并不是 (size, 1), 即使 s_bar 的形状为(size, 1)
        s_tide = s_obs[0:-1, 0] - nn_beta[0:-1, 0] * s_bar[:, 0] * i_bar[:, 0] / (s_bar[:, 0] + i_bar[:, 0])
        i_tide = i_obs[0:-1, 0] + nn_beta[0:-1, 0] * s_bar[:, 0] * i_bar[:, 0] / (s_bar[:, 0] + i_bar[:, 0]) \
                 - nn_gamma[0:-1, 0] * i_bar[:, 0] - nn_mu[0:-1, 0] * i_bar[:, 0]
        r_tide = r_obs[0:-1, 0] + nn_gamma[0:-1, 0] * i_bar[:, 0]
        d_tide = d_obs[0:-1, 0] + nn_mu[0:-1, 0] * i_bar[:, 0]

        s_pre = 0.5 * (s_bar + tf.reshape(s_tide, shape=[-1, 1]))
        i_pre = 0.5 * (i_bar + tf.reshape(i_tide, shape=[-1, 1]))
        r_pre = 0.5 * (r_bar + tf.reshape(r_tide, shape=[-1, 1]))
        d_pre = 0.5 * (d_bar + tf.reshape(d_tide, shape=[-1, 1]))

        diff2S = tf.reshape(s_obs[1: input_size, 0], shape=[-1, 1]) - s_pre
        diff2I = tf.reshape(i_obs[1: input_size, 0], shape=[-1, 1]) - i_pre
        diff2R = tf.reshape(r_obs[1: input_size, 0], shape=[-1, 1]) - r_pre
        diff2D = tf.reshape(d_obs[1: input_size, 0], shape=[-1, 1]) - d_pre

        if str.lower(loss_type) == 'l2_loss':
            Loss2S = tf.reduce_mean(tf.square(diff2S))
            Loss2I = tf.reduce_mean(tf.square(diff2I))
            Loss2R = tf.reduce_mean(tf.square(diff2R))
            Loss2D = tf.reduce_mean(tf.square(diff2D))
        elif str.lower(loss_type) == 'lncosh_loss':
            Loss2S = (1 / scale2lncosh) * tf.reduce_mean(tf.log(tf.cosh(scale2lncosh * diff2S)))
            Loss2I = (1 / scale2lncosh) * tf.reduce_mean(tf.log(tf.cosh(scale2lncosh * diff2I)))
            Loss2R = (1 / scale2lncosh) * tf.reduce_mean(tf.log(tf.cosh(scale2lncosh * diff2R)))
            Loss2D = (1 / scale2lncosh) * tf.reduce_mean(tf.log(tf.cosh(scale2lncosh * diff2D)))

        return ParamsNN, Loss2S, Loss2I, Loss2R, Loss2D

    def get_regularSum2WB(self):
        sum2WB = self.DNN.get_regular_sum2WB(self.opt2regular_WB)
        return sum2WB

    def evalue_EulerDNN(self, t=None, s_init=10.0, i_init=10.0, r_init=10.0, d_init=10.0, size2predict=7,
                        opt2itera='forward_euler'):
        assert (t is not None)              # 该处的t是训练过程的时间， size2predict 是测试的规模大小
        shape2t = t.get_shape().as_list()
        lenght2t_shape = len(shape2t)
        assert (lenght2t_shape == 2)
        assert (shape2t[-1] == 1)

        ParamsNN = self.DNN(t, scale=self.factor2freq, sFourier=self.sFourier)
        nn_beta = tf.reshape(ParamsNN[:, 0], shape=[-1, 1])
        nn_gamma = tf.reshape(ParamsNN[:, 1], shape=[-1, 1])
        nn_mu = tf.reshape(ParamsNN[:, 2], shape=[-1, 1])

        s_base = s_init
        i_base = i_init
        r_base = r_init
        d_base = d_init
        S_list, I_list, R_list, D_list = [], [], [], []
        if str.lower(opt2itera) == 'forward_euler' or str.lower(opt2itera) == 'quasi_euler':
            for i in range(size2predict):
                # temp2beta = nn_beta[i, 0]      # 这样得到的数据形状为shape=()
                # temp2gamma = nn_gamma[i, 0]
                # temp2mu = nn_mu[i, 0]
                s_update = s_base - nn_beta[i, 0] * s_base * i_base / (s_base + i_base)
                i_update = i_base + nn_beta[i, 0] * s_base * i_base / (s_base + i_base) - nn_gamma[i, 0] * i_base - nn_mu[i, 0] * i_base
                r_update = r_base + nn_gamma[i, 0] * i_base
                d_update = d_base + nn_mu[i, 0] * i_base

                S_list.append(tf.reshape(s_update, shape=[-1, 1]))
                I_list.append(tf.reshape(i_update, shape=[-1, 1]))
                R_list.append(tf.reshape(r_update, shape=[-1, 1]))
                D_list.append(tf.reshape(d_update, shape=[-1, 1]))

                s_base = s_update
                i_base = i_update
                r_base = r_update
                d_base = d_update
        elif str.lower(opt2itera) == 'modify_euler':
            for i in range(size2predict):
                s_bar = s_base - nn_beta[i, 0] * s_base * i_base / (s_base + i_base)
                i_bar = i_base + nn_beta[i, 0] * s_base * i_base / (s_base + i_base) - nn_gamma[i, 0] * i_base - nn_mu[i, 0] * i_base
                r_bar = r_base + nn_gamma[i, 0] * i_base
                d_bar = d_base + nn_mu[i, 0] * i_base

                s_tide = s_base - nn_beta[i, 0] * s_bar * i_bar / (s_bar + i_bar)
                i_tide = i_base + nn_beta[i, 0] * s_bar * i_bar / (s_bar + i_bar) - nn_gamma[i, 0] * i_bar - nn_mu[i, 0] * i_bar
                r_tide = r_base + nn_gamma[i, 0] * i_bar
                d_tide = d_base + nn_mu[i, 0] * i_bar

                s_update = 0.5 * (s_bar + s_tide)
                i_update = 0.5 * (i_bar + i_tide)
                r_update = 0.5 * (r_bar + r_tide)
                d_update = 0.5 * (d_bar + d_tide)

                S_list.append(tf.reshape(s_update, shape=[-1, 1]))
                I_list.append(tf.reshape(i_update, shape=[-1, 1]))
                R_list.append(tf.reshape(r_update, shape=[-1, 1]))
                D_list.append(tf.reshape(d_update, shape=[-1, 1]))

                s_base = s_update
                i_base = i_update
                r_base = r_update
                d_base = d_update
        S = tf.concat(S_list, axis=0)
        I = tf.concat(I_list, axis=0)
        R = tf.concat(R_list, axis=0)
        D = tf.concat(D_list, axis=0)
        return ParamsNN, S, I, R, D

    def evalue_EulerDNN_FixedParas(self, t=None, s_init=10.0, i_init=10.0, r_init=10.0, d_init=10.0, size2predict=7,
                        opt2itera='forward_euler', opt2fixed_paras='last2train', mean2para=3):
        assert (t is not None)
        shape2t = t.get_shape().as_list()
        lenght2t_shape = len(shape2t)
        assert (lenght2t_shape == 2)
        assert (shape2t[-1] == 1)

        ParamsNN = self.DNN(t, scale=self.factor2freq, sFourier=self.sFourier)

        # 训练过程中最后一天的参数作为固定参数
        if opt2fixed_paras == 'last2train':
            nn_beta = ParamsNN[0, 0]
            nn_gamma = ParamsNN[0, 1]
            nn_mu = ParamsNN[0, 2]
        else:  # 训练过程中最后几天的参数的均值作为固定参数，如三天的参数均值作为固定参数
            nn_beta = tf.reduce_mean(ParamsNN[:, 0], axis=0)
            nn_gamma = tf.reduce_mean(ParamsNN[:, 1], axis=0)
            nn_mu = tf.reduce_mean(ParamsNN[:, 2], axis=0)

        s_base = s_init
        i_base = i_init
        r_base = r_init
        d_base = d_init
        S_list, I_list, R_list, D_list = [], [], [], []
        if str.lower(opt2itera) == 'forward_euler' or str.lower(opt2itera) == 'quasi_euler':
            for i in range(size2predict):
                s_update = s_base - nn_beta * s_base * i_base / (s_base + i_base)
                i_update = i_base + nn_beta * s_base * i_base / (s_base + i_base) - nn_gamma * i_base - nn_mu * i_base
                r_update = r_base + nn_gamma * i_base
                d_update = d_base + nn_mu * i_base

                S_list.append(tf.reshape(s_update, shape=[-1, 1]))
                I_list.append(tf.reshape(i_update, shape=[-1, 1]))
                R_list.append(tf.reshape(r_update, shape=[-1, 1]))
                D_list.append(tf.reshape(d_update, shape=[-1, 1]))

                s_base = s_update
                i_base = i_update
                r_base = r_update
                d_base = d_update
        elif str.lower(opt2itera) == 'modify_euler':
            for i in range(size2predict):
                s_bar = s_base - nn_beta * s_base * i_base / (s_base + i_base)
                i_bar = i_base + nn_beta * s_base * i_base / (s_base + i_base) - nn_gamma * i_base - nn_mu * i_base
                r_bar = r_base + nn_gamma * i_base
                d_bar = d_base + nn_mu * i_base

                s_tide = s_base - nn_beta * s_bar * i_bar / (s_bar + i_bar)
                i_tide = i_base + nn_beta * s_bar * i_bar / (s_bar + i_bar) - nn_gamma * i_bar - nn_mu * i_bar
                r_tide = r_base + nn_gamma * i_bar
                d_tide = d_base + nn_mu * i_bar

                s_update = 0.5 * (s_bar + s_tide)
                i_update = 0.5 * (i_bar + i_tide)
                r_update = 0.5 * (r_bar + r_tide)
                d_update = 0.5 * (d_bar + d_tide)

                S_list.append(tf.reshape(s_update, shape=[-1, 1]))
                I_list.append(tf.reshape(i_update, shape=[-1, 1]))
                R_list.append(tf.reshape(r_update, shape=[-1, 1]))
                D_list.append(tf.reshape(d_update, shape=[-1, 1]))

                s_base = s_update
                i_base = i_update
                r_base = r_update
                d_base = d_update
        S = tf.concat(S_list, axis=0)
        I = tf.concat(I_list, axis=0)
        R = tf.concat(R_list, axis=0)
        D = tf.concat(D_list, axis=0)
        return ParamsNN, S, I, R, D


def solve_SIRD(R):
    log_out_path = R['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    logfile_name = '%s_%s.txt' % ('log2train', R['name2act_hidden'])
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    logUtils.dictionary_out2file(R, log_fileout)

    # 问题需要的设置
    trainSet_szie = R['size2train']                  # 训练集大小,给定一个数据集，拆分训练集和测试集时，需要多大规模的训练集
    batchsize2train = R['batch_size2train']
    batchsize2test = R['batch_size2test']
    penalty2WB = R['penalty2weight_biases']          # Regularization parameter for weights and biases
    lr_decay = R['learning_rate_decay']
    learning_rate = R['learning_rate']
    act_func = R['name2act_hidden']

    input_dim = R['input_dim']  # 1

    SIRDmodel = EulerDNN(input_dim=R['input_dim'], out_dim=R['output_dim'], hidden_layer=R['hidden_layers'],
                         Model_name=R['model2NN'], name2actIn=R['name2act_in'], name2actHidden=R['name2act_hidden'],
                         name2actOut=R['name2act_out'], opt2regular_WB='L0', type2numeric='float32',
                         factor2freq=R['freq'], sFourier=R['sfourier'])

    global_steps = tf.compat.v1.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.compat.v1.variable_scope('vscope', reuse=tf.compat.v1.AUTO_REUSE):
            t_train = tf.compat.v1.placeholder(tf.float32, name='t_train', shape=[batchsize2train, input_dim])
            S_observe = tf.compat.v1.placeholder(tf.float32, name='S_observe', shape=[batchsize2train, input_dim])
            I_observe = tf.compat.v1.placeholder(tf.float32, name='I_observe', shape=[batchsize2train, input_dim])
            R_observe = tf.compat.v1.placeholder(tf.float32, name='R_observe', shape=[batchsize2train, input_dim])
            D_observe = tf.compat.v1.placeholder(tf.float32, name='D_observe', shape=[batchsize2train, input_dim])

            t_test = tf.compat.v1.placeholder(tf.float32, name='t_test', shape=[batchsize2test, input_dim])
            init2S = tf.compat.v1.placeholder(tf.float32, name='init2S', shape=[])
            init2I = tf.compat.v1.placeholder(tf.float32, name='init2I', shape=[])
            init2R = tf.compat.v1.placeholder(tf.float32, name='init2R', shape=[])
            init2D = tf.compat.v1.placeholder(tf.float32, name='init2D', shape=[])
            t_test2fixed_paras = tf.compat.v1.placeholder(tf.float32, name='t_test2fixed_paras', shape=[1, input_dim])
            in_learning_rate = tf.compat.v1.placeholder_with_default(input=1e-5, shape=[], name='lr')

            regularSum2WB = SIRDmodel.get_regularSum2WB()
            Paras_PWB = penalty2WB * regularSum2WB
            if R['opt2Euler'] == 'quasi_euler':
                NN2Params, Loss_S, Loss_I, Loss_R, Loss_D = SIRDmodel.Quasi_Euler(
                    t=t_train, input_size=batchsize2train, s_obs=S_observe, i_obs=I_observe, r_obs=R_observe,
                    d_obs=D_observe, loss_type=R['loss_type'], scale2lncosh=R['lambda2lncosh'])
            elif R['opt2Euler'] == 'forward_euler':
                NN2Params, Loss_S, Loss_I, Loss_R, Loss_D = SIRDmodel.Forward_Euler(
                    t=t_train, input_size=batchsize2train, s_obs=S_observe, i_obs=I_observe, r_obs=R_observe,
                    d_obs=D_observe, loss_type=R['loss_type'], scale2lncosh=R['lambda2lncosh'])
            elif R['opt2Euler'] == 'modify_euler':
                NN2Params, Loss_S, Loss_I, Loss_R, Loss_D = SIRDmodel.Modify_Euler(
                    t=t_train, input_size=batchsize2train, s_obs=S_observe, i_obs=I_observe, r_obs=R_observe,
                    d_obs=D_observe, loss_type=R['loss_type'], scale2lncosh=R['lambda2lncosh'])

            Loss_ALL = Loss_S + Loss_I + Loss_R + Loss_D + Paras_PWB

            my_optimizer = tf.compat.v1.train.AdamOptimizer(in_learning_rate)
            if R['train_model'] == 'group3_training':
                train_op1 = my_optimizer.minimize(Loss_S, global_step=global_steps)
                train_op2 = my_optimizer.minimize(Loss_I, global_step=global_steps)
                train_op3 = my_optimizer.minimize(Loss_R, global_step=global_steps)
                train_op4 = my_optimizer.minimize(Loss_D, global_step=global_steps)
                train_op5 = my_optimizer.minimize(Loss_ALL, global_step=global_steps)
                train_Losses = tf.group(train_op1, train_op2, train_op3, train_op4, train_op5)
            elif R['train_model'] == 'union_training':
                train_Losses = my_optimizer.minimize(Loss_ALL, global_step=global_steps)

            LBFGS_optimizer = tf.contrib.opt.ScipyOptimizerInterface(Loss_ALL,
                                                                     var_list=None,
                                                                     method='L-BFGS-B',
                                                                     options={'maxiter': 100000,
                                                                     'maxfun': 100000,
                                                                     'maxcor': 50,
                                                                     'maxls': 50,
                                                                     'ftol': 1*np.finfo(float).eps})

            NN2Para_test, S_test, I_test, R_test, D_test = SIRDmodel.evalue_EulerDNN(
                t=t_test, s_init=init2S, i_init=init2I, r_init=init2R, d_init=init2D, size2predict=batchsize2test,
                opt2itera=R['opt2Euler'])

            NN2Para_test2fixed, S_test2fixed, I_test2fixed, R_test2fixed, D_test2fixed = \
                SIRDmodel.evalue_EulerDNN_FixedParas(t=t_test2fixed_paras, s_init=init2S, i_init=init2I, r_init=init2R,
                                                     d_init=init2D, size2predict=batchsize2test,
                                                     opt2itera=R['opt2Euler'], opt2fixed_paras='last2train')

    t0 = time.time()
    loss_all, loss_s_all, loss_i_all, loss_r_all, loss_d_all = [], [], [], [], []  # 空列表, 使用 append() 添加元素
    test_epoch = []
    test_mse2s_all, test_mse2i_all, test_mse2r_all, test_mse2d_all = [], [], [], []
    test_rel2s_all, test_rel2i_all, test_rel2r_all, test_rel2d_all = [], [], [], []

    test_mse2s_Fix_all, test_mse2i_Fix_all, test_mse2r_Fix_all, test_mse2d_Fix_all = [], [], [], []
    test_rel2s_Fix_all, test_rel2i_Fix_all, test_rel2r_Fix_all, test_rel2d_Fix_all = [], [], [], []

    # filename = 'data2csv/Wuhan.csv'
    # filename = 'data2csv/Italia_data.csv'
    # filename = 'data2csv/Korea_data.csv'
    # filename = 'data2csv/minnesota.csv'
    # filename = 'data2csv/minnesota2.csv'
    filename = 'data/minnesota3.csv'
    date, data2S, data2I, data2R, data2D = dataUtils.load_4csvData_cal_S(
        datafile=filename, total_population=R['total_population'])

    assert (trainSet_szie + batchsize2test <= len(data2I))
    if R['normalize_population'] == 1:
        # 不归一化数据
        train_date, train_data2s, train_data2i, train_data2r, train_data2d, test_date, test_data2s, test_data2i, \
        test_data2r, test_data2d = dataUtils.split_5csvData2train_test(date, data2S, data2I, data2R, data2D,
                                                                       size2train=trainSet_szie, normalFactor=1.0)
    elif (R['total_population'] != R['normalize_population']) and R['normalize_population'] != 1:
        # 归一化数据，使用的归一化数值小于总“人口”
        train_date, train_data2s, train_data2i, train_data2r, train_data2d, test_date, test_data2s, test_data2i, \
        test_data2r, test_data2d = dataUtils.split_5csvData2train_test(date, data2S, data2I, data2R, data2D,
                                                                       size2train=trainSet_szie,
                                                                       normalFactor=R['normalize_population'])
    elif (R['total_population'] == R['normalize_population']) and R['normalize_population'] != 1:
        # 归一化数据，使用总“人口”归一化数据
        train_date, train_data2s, train_data2i, train_data2r, train_data2d, test_date, test_data2s, test_data2i, \
        test_data2r, test_data2d = dataUtils.split_5csvData2train_test(date, data2S, data2I, data2R, data2D,
                                                                       size2train=trainSet_szie,
                                                                       normalFactor=R['total_population'])
    # 对于时间数据来说，验证模型的合理性，要用连续的时间数据验证.
    test_t_bach = dataUtils.sample_testDays_serially(test_date, batchsize2test)

    # 由于将数据拆分为训练数据和测试数据时，进行了归一化处理，故这里不用归一化
    s_obs_test = dataUtils.sample_testData_serially(test_data2s, batchsize2test, normalFactor=1.0)
    i_obs_test = dataUtils.sample_testData_serially(test_data2i, batchsize2test, normalFactor=1.0)
    r_obs_test = dataUtils.sample_testData_serially(test_data2r, batchsize2test, normalFactor=1.0)
    d_obs_test = dataUtils.sample_testData_serially(test_data2d, batchsize2test, normalFactor=1.0)

    # 测试过程的初始值，选为训练集的最后一天的值
    init2S_test = train_data2s[trainSet_szie-1]
    init2I_test = train_data2i[trainSet_szie-1]
    init2R_test = train_data2r[trainSet_szie-1]
    init2D_test = train_data2d[trainSet_szie-1]

    # 将训练集的最后一天和测试集的前n天连接起来，作为新的测试批大小
    last_train_ts = np.reshape(train_date[trainSet_szie - 5:-1], newshape=[1, 1])
    last_train_t = np.reshape(train_date[trainSet_szie - 1], newshape=[1, 1])
    new_test_t_bach = np.concatenate([last_train_t, np.reshape(test_t_bach[0:-1, 0], newshape=[-1, 1])], axis=0)
    tmp_lr = learning_rate
    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        for epoch in range(R['max_epoch'] + 1):
            tmp_lr = tmp_lr * (1 - lr_decay)
            if batchsize2train == trainSet_szie:
                t_batch = np.reshape(train_date, newshape=[-1, 1])
                s_obs = np.reshape(train_data2s, newshape=[-1, 1])
                i_obs = np.reshape(train_data2i, newshape=[-1, 1])
                r_obs = np.reshape(train_data2r, newshape=[-1, 1])
                d_obs = np.reshape(train_data2d, newshape=[-1, 1])
            else:
                t_batch, s_obs, i_obs, r_obs, d_obs = \
                    dataUtils.randSample_Normalize_5existData(
                        train_date, train_data2s, train_data2i, train_data2r, train_data2d, batchsize=batchsize2train,
                        normalFactor=1.0, sampling_opt=R['opt2sample'])

            _, loss2s, loss2i, loss2r, loss2d, loss, pwb2paras = sess.run(
                [train_Losses, Loss_S, Loss_I, Loss_R, Loss_D, Loss_ALL, Paras_PWB],
                feed_dict={t_train: t_batch, S_observe: s_obs, I_observe: i_obs, R_observe: r_obs, D_observe: d_obs})

            loss_s_all.append(loss2s)
            loss_i_all.append(loss2i)
            loss_r_all.append(loss2r)
            loss_d_all.append(loss2d)
            loss_all.append(loss)

            if epoch % 1000 == 0:
                logUtils.print_training2OneNet(epoch, time.time() - t0, tmp_lr, pwb2paras, loss2s, loss2i, loss2r,
                                               loss2d, loss, log_out=log_fileout)

                # ---------------------------   test network ----------------------------------------------
                test_epoch.append(epoch / 1000)

                paras_nn, S_predict, I_predict, R_predict, D_predict = sess.run(
                    [NN2Para_test, S_test, I_test, R_test, D_test], feed_dict={t_test: new_test_t_bach,
                     init2S: init2S_test, init2I: init2I_test, init2R: init2R_test, init2D: init2D_test})
                test_mse2S = np.mean(np.square(s_obs_test - S_predict))
                test_mse2I = np.mean(np.square(i_obs_test - I_predict))
                test_mse2R = np.mean(np.square(r_obs_test - R_predict))
                test_mse2D = np.mean(np.square(d_obs_test - D_predict))

                test_rel2S = test_mse2S / np.mean(np.square(s_obs_test))
                test_rel2I = test_mse2I / np.mean(np.square(i_obs_test))
                test_rel2R = test_mse2R / np.mean(np.square(r_obs_test))
                test_rel2D = test_mse2D / np.mean(np.square(d_obs_test))

                test_mse2s_all.append(test_mse2S)
                test_mse2i_all.append(test_mse2I)
                test_mse2r_all.append(test_mse2R)
                test_mse2d_all.append(test_mse2D)

                test_rel2s_all.append(test_rel2S)
                test_rel2i_all.append(test_rel2I)
                test_rel2r_all.append(test_rel2R)
                test_rel2d_all.append(test_rel2D)
                logUtils.print_test2OneNet(test_mse2S, test_mse2I, test_mse2R, test_mse2D, test_rel2S, test_rel2I,
                                           test_rel2R, test_rel2D, log_out=log_fileout)

                fix_paras_nn, S_predict2fix, I_predict2fix, R_predict2fix, D_predict2fix = sess.run(
                    [NN2Para_test2fixed, S_test2fixed, I_test2fixed, R_test2fixed, D_test2fixed],
                    feed_dict={t_test2fixed_paras: last_train_t, init2S: init2S_test, init2I: init2I_test,
                               init2R: init2R_test, init2D: init2D_test})

                test_mse2S_fix = np.mean(np.square(s_obs_test - S_predict2fix))
                test_mse2I_fix = np.mean(np.square(i_obs_test - I_predict2fix))
                test_mse2R_fix = np.mean(np.square(r_obs_test - R_predict2fix))
                test_mse2D_fix = np.mean(np.square(d_obs_test - D_predict2fix))

                test_rel2S_fix = test_mse2S_fix / np.mean(np.square(s_obs_test))
                test_rel2I_fix = test_mse2I_fix / np.mean(np.square(i_obs_test))
                test_rel2R_fix = test_mse2R_fix / np.mean(np.square(r_obs_test))
                test_rel2D_fix = test_mse2D_fix / np.mean(np.square(d_obs_test))

                test_mse2s_Fix_all.append(test_mse2S_fix)
                test_mse2i_Fix_all.append(test_mse2I_fix)
                test_mse2r_Fix_all.append(test_mse2R_fix)
                test_mse2d_Fix_all.append(test_mse2D_fix)

                test_rel2s_Fix_all.append(test_rel2S_fix)
                test_rel2i_Fix_all.append(test_rel2I_fix)
                test_rel2r_Fix_all.append(test_rel2R_fix)
                test_rel2d_Fix_all.append(test_rel2D_fix)

                logUtils.print_testFix_paras2OneNet(test_mse2S_fix, test_mse2I_fix, test_mse2R_fix, test_mse2D_fix,
                                                    test_rel2S_fix, test_rel2I_fix, test_rel2R_fix, test_rel2D_fix,
                                                    log_out=log_fileout)

    # ------------------- save the training results into mat file and plot them -------------------------
    saveData.save_SIRD_trainLoss2mat_no_N(loss_s_all, loss_i_all, loss_r_all, loss_d_all, actName=act_func,
                                          outPath=R['FolderName'])

    plotData.plotTrain_loss_1act_func(loss_s_all, lossType='loss2s', seedNo=R['seed'], outPath=R['FolderName'],
                                      yaxis_scale=True)
    plotData.plotTrain_loss_1act_func(loss_i_all, lossType='loss2i', seedNo=R['seed'], outPath=R['FolderName'],
                                      yaxis_scale=True)
    plotData.plotTrain_loss_1act_func(loss_r_all, lossType='loss2r', seedNo=R['seed'], outPath=R['FolderName'],
                                      yaxis_scale=True)
    plotData.plotTrain_loss_1act_func(loss_d_all, lossType='loss2d', seedNo=R['seed'], outPath=R['FolderName'],
                                      yaxis_scale=True)

    # ------------------- save the testing results into mat file and plot them -------------------------
    plotData.plotTest_MSE_REL(test_mse2s_all, test_rel2s_all, test_epoch, actName='S', seedNo=R['seed'],
                              outPath=R['FolderName'], xaxis_scale=False, yaxis_scale=True)
    plotData.plotTest_MSE_REL(test_mse2i_all, test_rel2i_all, test_epoch, actName='I', seedNo=R['seed'],
                              outPath=R['FolderName'], xaxis_scale=False, yaxis_scale=True)
    plotData.plotTest_MSE_REL(test_mse2r_all, test_rel2r_all, test_epoch, actName='R', seedNo=R['seed'],
                              outPath=R['FolderName'], xaxis_scale=False, yaxis_scale=True)
    plotData.plotTest_MSE_REL(test_mse2d_all, test_rel2d_all, test_epoch, actName='D', seedNo=R['seed'],
                              outPath=R['FolderName'], xaxis_scale=False, yaxis_scale=True)

    plotData.plotTest_MSE_REL(test_mse2s_Fix_all, test_rel2s_Fix_all, test_epoch, actName='S_Fix', seedNo=R['seed'],
                              outPath=R['FolderName'], xaxis_scale=False, yaxis_scale=True)
    plotData.plotTest_MSE_REL(test_mse2i_Fix_all, test_rel2i_Fix_all, test_epoch, actName='I_Fix', seedNo=R['seed'],
                              outPath=R['FolderName'], xaxis_scale=False, yaxis_scale=True)
    plotData.plotTest_MSE_REL(test_mse2r_Fix_all, test_rel2r_Fix_all, test_epoch, actName='R_Fix', seedNo=R['seed'],
                              outPath=R['FolderName'], xaxis_scale=False, yaxis_scale=True)
    plotData.plotTest_MSE_REL(test_mse2d_Fix_all, test_rel2d_Fix_all, test_epoch, actName='D_Fix', seedNo=R['seed'],
                              outPath=R['FolderName'], xaxis_scale=False, yaxis_scale=True)

    plotData.plot_3solus2SIRD_test(s_obs_test, S_predict, S_predict2fix, exact_name='S_true', solu1_name='S_pre2time',
                                   solu2_name='S_pre2fix',  file_name='S_solu', coord_points=test_t_bach,
                                   outPath=R['FolderName'])
    plotData.plot_3solus2SIRD_test(i_obs_test, I_predict, I_predict2fix, exact_name='I_true', solu1_name='I_pre2time',
                                   solu2_name='I_pre2fix', file_name='I_solu', coord_points=test_t_bach,
                                   outPath=R['FolderName'])
    plotData.plot_3solus2SIRD_test(r_obs_test, R_predict, R_predict2fix, exact_name='R_true', solu1_name='R_pre2time',
                                   solu2_name='R_pre2fix', file_name='R_solu', coord_points=test_t_bach,
                                   outPath=R['FolderName'])
    plotData.plot_3solus2SIRD_test(d_obs_test, D_predict, D_predict2fix, exact_name='D_true', solu1_name='D_pre2time',
                                   solu2_name='D_pre2fix', file_name='D_solu', coord_points=test_t_bach,
                                   outPath=R['FolderName'])

    saveData.save_SIRD_testSolus2mat(S_predict, I_predict, R_predict, D_predict, name2solus1='S_pre',
                                     name2solus2='I_pre', name2solus3='R_pre', name2solus4='D_pre',
                                     file_name='timeParas', outPath=R['FolderName'])

    saveData.save_SIRD_testSolus2mat(S_predict2fix, I_predict2fix, R_predict2fix, D_predict2fix, name2solus1='S_pre',
                                     name2solus2='I_pre', name2solus3='R_pre', name2solus4='D_pre',
                                     file_name='fixParas', outPath=R['FolderName'])

    saveData.save_SIRD_testParas2mat(paras_nn[:, 0], paras_nn[:, 1], paras_nn[:, 2], name2para1='Beta',
                                     name2para2='Gamma', name2para3='Mu', outPath=R['FolderName'])


if __name__ == "__main__":
    R = {}
    R['gpuNo'] = 0
    if platform.system() == 'Windows':
        os.environ["CDUA_VISIBLE_DEVICES"] = "%s" % (R['gpuNo'])
    else:
        print('-------------------------------------- linux -----------------------------------------------')
        # Linux终端没有GUI, 需要添加如下代码，而且必须添加在 import matplotlib.pyplot 之前，否则无效。
        matplotlib.use('Agg')

        if tf.test.is_gpu_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 设置当前使用的GPU设备仅为第 0,1,2,3 块GPU, 设备名称为'/gpu:0'
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # The path of saving files
    store_file = 'SIRD_Euler'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    OUT_DIR = os.path.join(BASE_DIR, store_file)
    if not os.path.exists(OUT_DIR):
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    R['seed'] = np.random.randint(1e5)
    seed_str = str(R['seed'])  # int 型转为字符串型
    FolderName = os.path.join(OUT_DIR, seed_str)  # 路径连接
    R['FolderName'] = FolderName
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
        os.mkdir(FolderName)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Copy and save this file to given path %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if platform.system() == 'Windows':
        tf.compat.v1.reset_default_graph()
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    # step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
    # R['activate_stop'] = int(step_stop_flag)
    R['activate_stop'] = int(0)
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    R['max_epoch'] = 100000
    # R['max_epoch'] = 200000
    if 0 != R['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        R['max_epoch'] = int(epoch_stop)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setups of problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    R['input_dim'] = 1                                    # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 3                                   # 输出维数

    R['ODE_type'] = 'SIRD'
    R['equa_name'] = 'minnesota'

    # R['opt2Euler'] = 'quasi_euler'
    R['opt2Euler'] = 'forward_euler'
    # R['opt2Euler'] = 'modify_euler'

    R['total_population'] = 3450000  # 总的“人口”数量

    # R['normalize_population'] = 3450000                # 归一化时使用的“人口”数值
    R['normalize_population'] = 10000
    # R['normalize_population'] = 5000
    # R['normalize_population'] = 2000
    # R['normalize_population'] = 1000
    # R['normalize_population'] = 1

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup of DNN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    R['size2train'] = 280                               # 训练集的大小
    # R['batch_size2train'] = 30                        # 训练数据的批大小
    # R['batch_size2train'] = 80                        # 训练数据的批大小
    R['batch_size2train'] = 280  # 训练数据的批大小
    R['batch_size2test'] = 7  # 训练数据的批大小
    # R['opt2sample'] = 'random_sample'                 # 训练集的选取方式--随机采样
    # R['opt2sample'] = 'rand_sample_sort'              # 训练集的选取方式--随机采样后按时间排序
    R['opt2sample'] = 'windows_rand_sample'             # 训练集的选取方式--随机窗口采样(以随机点为基准，然后滑动窗口采样)

    # The types of loss function
    R['loss_type'] = 'L2_loss'
    # R['loss_type'] = 'lncosh_loss'
    # R['lambda2lncosh'] = 0.01
    R['lambda2lncosh'] = 0.05                           # 这个因子效果很好
    # R['lambda2lncosh'] = 0.075
    # R['lambda2lncosh'] = 0.1
    # R['lambda2lncosh'] = 0.5
    # R['lambda2lncosh'] = 1.0
    # R['lambda2lncosh'] = 50.0

    # The options of optimizers, learning rate, the decay of learning rate and the model of training network
    R['optimizer_name'] = 'Adam'                       # 优化器

    # R['learning_rate'] = 1e-2                          # 学习率
    # R['learning_rate_decay'] = 2e-4                    # 学习率 decay

    R['learning_rate'] = 5e-3                         # 学习率
    R['learning_rate_decay'] = 12e-5                  # 学习率 decay
    # R['learning_rate_decay'] = 15e-5                  # 学习率 decay

    # R['learning_rate'] = 2e-4                          # 学习率
    # R['learning_rate_decay'] = 5e-5                    # 学习率 decay
    R['train_model'] = 'union_training'
    # R['train_model'] = 'group2_training'
    # R['train_model'] = 'group3_training'

    # 正则化权重和偏置的模式
    # R['regular_wb_model'] = 'L0'
    # R['regular_wb_model'] = 'L1'
    R['regular_wb_model'] = 'L2'
    # R['penalty2weight_biases'] = 0.000                    # Regularization parameter for weights
    R['penalty2weight_biases'] = 0.00005                  # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.0001                     # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.0005                    # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.001                  # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.0025                 # Regularization parameter for weights

    # 边界的惩罚处理方式,以及边界的惩罚因子
    R['activate_penalty2pt_increase'] = 1
    # R['init_penalty2predict_true'] = 1000               # Regularization factor for the  prediction and true
    # R['init_penalty2predict_true'] = 100                # Regularization factor for the  prediction and true
    R['init_penalty2predict_true'] = 10                   # Regularization factor for the  prediction and true

    # &&&&&&& The option fo Network model, the setups of hidden-layers and the option of activation function &&&&&&&&&&&
    # R['model2NN'] = 'DNN'
    # R['model2NN'] = 'Scale_DNN'
    # R['model2NN'] = 'Adapt_scale_DNN'
    R['model2NN'] = 'Fourier_DNN'

    if R['model2NN'] == 'Fourier_DNN':
        R['hidden_layers'] = (125, 200, 100, 100, 80)
    else:
        R['hidden_layers'] = (250, 200, 100, 100, 80)

    # R['name2act_in'] = 'relu'
    # R['name2act_in'] = 'leaky_relu'
    # R['name2act_in'] = 'elu'
    # R['name2act_in'] = 'gelu'
    # R['name2act_in'] = 'mgelu'
    R['name2act_in'] = 'tanh'
    # R['name2act_in'] = 'sin'
    # R['name2act_in'] = 'sinAddcos'
    # R['name2act_in'] = 's2relu'

    # R['name2act_hidden'] = 'relu'
    R['name2act_hidden'] = 'tanh'
    # R['name2act_hidden']' = leaky_relu'
    # R['name2act_hidden'] = 'srelu'
    # R['name2act_hidden'] = 's2relu'
    # R['name2act_hidden'] = 'scsrelu'
    # R['name2act_hidden'] = 'sin'
    # R['name2act_hidden'] = 'sinAddcos'
    # R['name2act_hidden'] = 'elu'

    R['name2act_out'] = 'linear'
    # R['name2act_out'] = 'sigmoid'

    # &&&&&&&&&&&&&&&&&&&&& some other factors for network &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    R['freq'] = np.concatenate(([1], np.arange(1, 30 - 1)), axis=0)  # 网络的频率范围设置

    if R['model2NN'] == 'DNN_FourierBase' or R['model2NN'] == 'Scale_DNN' or R['model2NN'] == 'Fourier_DNN':
        R['if_repeat_High_freq'] = False

    if R['model2NN'] == 'Fourier_DNN' and R['name2act_hidden'] == 'tanh':
        # R['sfourier'] = 0.5
        R['sfourier'] = 1.0
    elif R['model2NN'] == 'Fourier_DNN' and R['name2act_hidden'] == 's2relu':
        R['sfourier'] = 0.5
        # R['sfourier'] = 1.0
    elif R['model2NN'] == 'Fourier_DNN' and R['name2act_hidden'] == 'sinAddcos':
        # R['sfourier'] = 0.5
        R['sfourier'] = 1.0
    elif R['model2NN'] == 'Fourier_DNN' and R['name2act_hidden'] == 'sin':
        # R['sfourier'] = 0.5
        R['sfourier'] = 1.0
    else:
        R['sfourier'] = 1.0
        # R['sfourier'] = 5.0
        # R['sfourier'] = 0.75

    solve_SIRD(R)
