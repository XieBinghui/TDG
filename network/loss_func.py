import pdb

import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
from .types_ import *


def kl_divergence(latent_space_a, latent_space_b):
    return torch.mean(torch.distributions.kl_divergence(latent_space_a, latent_space_b))


def temporal_smooth_loss(latent_variables: Tensor, batch_first=True):
    if batch_first:
        return F.l1_loss(latent_variables[:, 1:, :], latent_variables[:, :-1, :], reduction='mean')
    else:
        return F.l1_loss(latent_variables[1:, :, :], latent_variables[-1:, :, :], reduction='mean')


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        raise ValueError('Must specify the dimension.')


def log_density(sample, mu, logsigma):
    mu = mu.type_as(sample)
    logsigma = logsigma.type_as(sample)
    c = torch.Tensor([np.log(2 * np.pi)]).type_as(sample.data)

    inv_sigma = torch.exp(-logsigma)
    tmp = (sample - mu) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * logsigma + c)

def matrix_log_density_gaussian(x, mu, logvar):
    batch_size, dim = x.shape
    x = x.view(batch_size, 1, dim)
    mu = mu.view(1, batch_size, dim)
    logvar = logvar.view(1, batch_size, dim)
    return log_density_gaussian(x, mu, logvar)

def log_density_gaussian(x, mu, logvar):
    normalization = - 0.5 * (math.log(2 * math.pi) + logvar)
    inv_var = torch.exp(-logvar)
    log_density = normalization - 0.5 * ((x - mu)**2 * inv_var)
    return log_density

def log_importance_weight_matrix(batch_size, dataset_size):
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[::M + 1] = 1 / N
    W.view(-1)[1::M + 1] = strat_weight
    W[M - 1, 0] = strat_weight
    return W.log()

def compute_mi(latent_sample, latent_dist):
    log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(latent_sample,
                                                                         latent_dist,
                                                                         None,
                                                                         is_mss=False)
    mi_loss = (log_q_zCx - log_qz).mean()

    return mi_loss


def _get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, n_data, is_mss=True):
    batch_size, hidden_dim = latent_sample.shape

    # calculate log q(z|x)
    log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)

    if is_mss:
        # use stratification
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(latent_sample.device)
        mat_log_qz = mat_log_qz + log_iw_mat.view(batch_size, batch_size, 1)

    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
    return None, log_qz, None, log_q_zCx

