# -*- coding: utf-8 -*-
"""Training losses. y, pred: (B, 1) in 0..1; w: (B,) sample weights (mean 1 over the training set)."""
import numpy as np
import tensorflow as tf


def weighted_mse(y, pred, w):
    return tf.reduce_mean(w * tf.reduce_sum(tf.square(pred - y), axis=-1))


def weighted_huber(y, pred, w, delta):
    e = tf.abs(pred - y)
    h = tf.where(e <= delta, 0.5 * tf.square(e), delta * (e - 0.5 * delta))
    return tf.reduce_mean(w * tf.reduce_sum(h, axis=-1))


def balanced_mse_bmc(y, pred, sigma):
    """Balanced MSE, Batch-based Monte-Carlo version (Ren et al., CVPR 2022, official code):
    logits_ij = -(pred_i - y_j)^2 / (2 sigma^2); loss = CE(logits, i) * (2 sigma^2) (constant scale).
    sigma (label noise) is learned."""
    var = tf.square(sigma)
    logits = -0.5 * tf.square(pred - tf.transpose(y)) / var          # (B, B)
    labels = tf.range(tf.shape(y)[0])
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(ce) * tf.stop_gradient(2.0 * var)


def dldl_targets(y, bins, sigma):
    """Gaussian label distribution over fixed bin centres 0..1 (DLDL)."""
    c = tf.constant(np.linspace(0, 1, bins, dtype=np.float32))[None, :]
    t = tf.exp(-0.5 * tf.square((c - y) / sigma))
    return t / tf.reduce_sum(t, axis=-1, keepdims=True)


def dldl_v2(y, probs, pred, w, bins, sigma, l1_weight):
    """DLDL-v2 (Gao et al., IJCAI 2018): KL(label distribution || predicted distribution) +
    L1 between the expected value and the label. The L1 term is measured in percent of power loss
    (0..100), the same scale as the 0..100 age bins of the original paper, so l1_weight = 1 as there."""
    t = dldl_targets(y, bins, sigma)
    kl = tf.reduce_sum(t * (tf.math.log(t + 1e-12) - tf.math.log(probs + 1e-12)), axis=-1)
    l1 = 100.0 * tf.reduce_sum(tf.abs(pred - y), axis=-1)
    return tf.reduce_mean(w * (kl + l1_weight * l1))


def _rank_rows(x):
    """Rank of every element within its row, scaled to (0, 1]."""
    r = tf.argsort(tf.argsort(x, axis=-1, stable=True), axis=-1, stable=True)
    return (tf.cast(r, tf.float32) + 1.0) / tf.cast(tf.shape(x)[-1], tf.float32)


def ranksim(feat, y, lam):
    """RankSim (Gong et al., ICML 2022, official code): for each image in the batch, the ranking of its
    cosine similarities to the other images (features) should match the ranking of label similarities
    (-|y_i - y_j|). Sorting is made differentiable with the blackbox method (Vlastelica et al., ICLR 2020)."""

    @tf.custom_gradient
    def true_rank(s):
        r = _rank_rows(s)

        def grad(dy):
            r2 = _rank_rows(s + lam * dy)
            return -(r - r2) / (lam + 1e-8)
        return r, grad

    f = tf.math.l2_normalize(feat, axis=-1)
    sim_f = tf.matmul(f, f, transpose_b=True)
    sim_y = -tf.abs(y - tf.transpose(y))
    return tf.reduce_mean(tf.square(true_rank(sim_f) - _rank_rows(sim_y)))
