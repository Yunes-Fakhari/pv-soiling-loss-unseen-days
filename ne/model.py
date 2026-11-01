# -*- coding: utf-8 -*-
"""MobileNet v1 regressor, identical to the paper model:
image -> MobileNet v1 (ImageNet) -> global average pooling -> concat(irradiance) -> FC128 -> FC32 -> 1 (sigmoid).
DLDL keeps the same body; only the last layer becomes 101 softmax bins and the prediction is their
expected value (a fixed, non-trainable layer), so prediction is still one number from one forward pass."""
import re

import keras
import numpy as np
import tensorflow as tf

from .common import ensure_imagenet_weights


def build(cfg, imagenet=True):
    """Returns (infer, train_net, full).
    infer:     inputs {'image','irr'} -> loss prediction (N, 1). This is what is saved and used.
    train_net: same layers, outputs {'pred', 'feat'(FC32 features, for RankSim)[, 'probs' (DLDL)]}."""
    size = int(cfg["img_size"])
    seed = int(cfg["seed"])
    if imagenet:
        ensure_imagenet_weights()
    img_in = tf.keras.Input(shape=(size, size, 3), name="image")
    # "irr" = irradiance, plus the panel features (panel_index, sun) for recipe cmixup_feat
    irr_in = tf.keras.Input(shape=(1 + len(cfg.get("panel_features") or []),), name="irr")
    base = tf.keras.applications.MobileNet(input_shape=(size, size, 3), alpha=float(cfg["alpha"]),
                                           include_top=False, weights="imagenet" if imagenet else None)
    base.trainable = True
    feat_map = base(img_in)                                   # 6 x 6 x 1024
    if cfg.get("attention") == "eca":
        feat_map = eca_attention(feat_map, int(cfg.get("eca_kernel", 5)))
    elif cfg.get("attention") == "cbam":
        feat_map = cbam_attention(feat_map, int(cfg.get("cbam_reduction", 16)), int(cfg.get("cbam_kernel", 3)))
    elif cfg.get("attention") == "se":
        feat_map = se_attention(feat_map, int(cfg.get("se_reduction", 16)))
    elif cfg.get("attention") == "ca":
        feat_map = coord_attention(feat_map, int(cfg.get("ca_reduction", 32)))
    elif cfg.get("attention"):
        raise ValueError(f"unknown attention: {cfg.get('attention')}")
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(feat_map)
    if cfg.get("coverage_input"):
        # extra feature measured from the SAME image inside the network: share of the panel covered by soiling
        area = SoiledArea(size, name="soiled_area_feature")(img_in)
        x = tf.keras.layers.Concatenate(name="concat_irradiance")([x, irr_in, area])
    else:
        x = tf.keras.layers.Concatenate(name="concat_irradiance")([x, irr_in])
    feat = None
    for i, units in enumerate(cfg["dense_units"]):
        x = tf.keras.layers.Dense(int(units), activation="relu", name=f"fc{i + 1}_{units}",
                                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + i))(x)
        feat = x
        if float(cfg["dropout"]) > 0:
            x = tf.keras.layers.Dropout(float(cfg["dropout"]))(x)
    outs = {"feat": feat}
    if cfg["loss"] == "dldl":
        K = int(cfg["dldl"]["bins"])
        probs = tf.keras.layers.Dense(K, activation="softmax", name="power_loss_bins",
                                      kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 99))(x)
        expect = tf.keras.layers.Dense(1, use_bias=False, trainable=False, name="power_loss")
        pred = expect(probs)
        expect.set_weights([np.linspace(0, 1, K, dtype=np.float32).reshape(K, 1)])   # bin centres 0..1
        outs["probs"] = probs
    else:
        pred = tf.keras.layers.Dense(1, activation="sigmoid", name="power_loss",
                                     kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 99))(x)
    outs["pred"] = pred
    aux = cfg.get("aux_targets")
    if aux:
        # training-only extra outputs (not part of the saved prediction model):
        #   heat: soiling of each of the 6 x 6 image blocks, from the last feature map (1 x 1 conv)
        #   aux_idx: regional soiling index, from the same 32 features as the power-loss output
        if float(aux.get("heat_weight", 0)) > 0:
            h = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", name="aux_heatmap",
                                       kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 55))(feat_map)
            outs["heat"] = tf.keras.layers.Reshape((-1,), name="aux_heatmap_flat")(h)
        if float(aux.get("index_weight", 0)) > 0:
            outs["aux_idx"] = tf.keras.layers.Dense(1, activation="sigmoid", name="aux_regional_index",
                                                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 66))(x)
    inputs = {"image": img_in, "irr": irr_in}
    full = None
    if cfg.get("coverage_head"):
        # second output: share of the panel covered by soiling (0..1), from the same 32 features
        cov = tf.keras.layers.Dense(1, activation="sigmoid", name="soiled_area",
                                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 77))(x)
        outs["cov"] = cov
        full = tf.keras.Model(inputs, {"power_loss": pred, "soiled_area": cov}, name="SoilingEdge_two_outputs")
    infer = tf.keras.Model(inputs, pred, name="SoilingEdge_mobilenet_v1")
    train_net = tf.keras.Model(inputs, outs, name="SoilingEdge_train")
    return infer, train_net, full


@keras.saving.register_keras_serializable(package="soiling")
class SoiledArea(tf.keras.layers.Layer):
    """Fixed (not trained) layer: share of the soiled panel that is NOT clearly blue in the input image.
    Same rule as ne/coverage.py (soft step): a pixel is clean when blue > red + 25 and blue > green + 5 (0-255),
    counted inside the panel (frame excluded). Input: the normalised image (-1..1). Output: (batch, 1) in 0..1.
    Because it is computed from the image itself, prediction still needs only the image and the irradiance."""

    def __init__(self, size=192, **kwargs):
        kwargs["trainable"] = False
        super().__init__(**kwargs)
        self.size = int(size)

    def build(self, input_shape):
        from .coverage import inner_panel_mask
        mk = inner_panel_mask(self.size).astype("float32")
        self.mask = tf.constant(mk[None, ...])
        self.n = float(mk.sum())
        super().build(input_shape)

    def call(self, img):
        x = (tf.cast(img, tf.float32) + 1.0) * 127.5
        r, g, b = x[..., 0], x[..., 1], x[..., 2]
        # soft version of (b > r + 25) and (b > g + 5): plain arithmetic, so it also runs in int8 TFLite
        clean = tf.sigmoid((b - r - 25.0) / 2.0) * tf.sigmoid((b - g - 5.0) / 2.0)
        covered = tf.reduce_sum((1.0 - clean) * self.mask, axis=[1, 2]) / self.n
        return tf.expand_dims(covered, -1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

    def get_config(self):
        return {**super().get_config(), "size": self.size}


def eca_attention(feature, kernel_size=5):
    """ECA channel attention (Wang et al., ECA-Net, CVPR 2020) after the last MobileNet block:
    GAP -> 1-D convolution across neighbouring channels -> sigmoid -> one weight per channel (1024).
    The convolution starts at zero and the gate is 2*sigmoid, so at the start every weight is exactly 1
    and the network begins identical to the one without attention. Adds only `kernel_size` parameters."""
    ch = int(feature.shape[-1])
    g = tf.keras.layers.GlobalAveragePooling2D(name="eca_gap")(feature)
    g = tf.keras.layers.Reshape((ch, 1), name="eca_sequence")(g)
    g = tf.keras.layers.Conv1D(1, kernel_size, padding="same", use_bias=False,
                               kernel_initializer="zeros", name="eca_conv")(g)
    g = tf.keras.layers.Activation("sigmoid", name="eca_sigmoid")(g)
    g = tf.keras.layers.Rescaling(2.0, name="eca_scale")(g)
    g = tf.keras.layers.Reshape((1, 1, ch), name="eca_gate")(g)
    return tf.keras.layers.Multiply(name="eca_apply")([feature, g])


def cbam_attention(feature, reduction=16, kernel_size=3):
    """CBAM (Woo et al., ECCV 2018) after the last MobileNet block: first WHICH channels matter
    (shared MLP on average- and max-pooled channels), then WHERE on the 6x6 map (convolution on the
    channel-mean and channel-max maps). The last layer of each gate starts at zero and each gate is
    2*sigmoid, so at the start every weight is exactly 1 and the network begins as without attention."""
    ch = int(feature.shape[-1])
    hidden = max(8, ch // reduction)
    down = tf.keras.layers.Dense(hidden, activation="relu", name="cbam_down")
    up = tf.keras.layers.Dense(ch, kernel_initializer="zeros", name="cbam_up")
    a = up(down(tf.keras.layers.GlobalAveragePooling2D(name="cbam_avg")(feature)))
    b = up(down(tf.keras.layers.GlobalMaxPooling2D(name="cbam_max")(feature)))
    c = tf.keras.layers.Activation("sigmoid", name="cbam_channel_sigmoid")(tf.keras.layers.Add(name="cbam_add")([a, b]))
    c = tf.keras.layers.Rescaling(2.0, name="cbam_channel_scale")(c)
    c = tf.keras.layers.Reshape((1, 1, ch), name="cbam_channel_gate")(c)
    x = tf.keras.layers.Multiply(name="cbam_channel_apply")([feature, c])
    pooled = tf.keras.layers.Concatenate(axis=-1, name="cbam_spatial_pool")([
        tf.keras.ops.mean(x, axis=-1, keepdims=True), tf.keras.ops.max(x, axis=-1, keepdims=True)])
    s = tf.keras.layers.Conv2D(1, kernel_size, padding="same", kernel_initializer="zeros",
                               name="cbam_spatial_conv")(pooled)
    s = tf.keras.layers.Activation("sigmoid", name="cbam_spatial_sigmoid")(s)
    s = tf.keras.layers.Rescaling(2.0, name="cbam_spatial_scale")(s)
    return tf.keras.layers.Multiply(name="cbam_spatial_apply")([x, s])


def se_attention(feature, reduction=16):
    """Squeeze-and-Excitation (Hu et al., CVPR 2018): GAP -> FC (1024/16) -> FC (1024) -> one weight per channel.
    The last FC starts at zero and the gate is 2*sigmoid, so at the start every weight is exactly 1."""
    ch = int(feature.shape[-1])
    g = tf.keras.layers.GlobalAveragePooling2D(name="se_gap")(feature)
    g = tf.keras.layers.Dense(max(8, ch // reduction), activation="relu", name="se_down")(g)
    g = tf.keras.layers.Dense(ch, kernel_initializer="zeros", name="se_up")(g)
    g = tf.keras.layers.Activation("sigmoid", name="se_sigmoid")(g)
    g = tf.keras.layers.Rescaling(2.0, name="se_scale")(g)
    g = tf.keras.layers.Reshape((1, 1, ch), name="se_gate")(g)
    return tf.keras.layers.Multiply(name="se_apply")([feature, g])


def coord_attention(feature, reduction=32):
    """Coordinate Attention (Hou et al., CVPR 2021), designed for mobile networks: channel weights that also
    know the ROW and the COLUMN of the 6x6 map (pooling along width and along height separately), so the
    network can stress the part of the panel where the soiling is. Gates start at exactly 1 (zero init, 2*sigmoid)."""
    h, w, ch = int(feature.shape[1]), int(feature.shape[2]), int(feature.shape[3])
    mid = max(8, ch // reduction)
    xh = tf.keras.layers.AveragePooling2D(pool_size=(1, w), name="ca_pool_w")(feature)          # h x 1 x ch
    xw = tf.keras.layers.AveragePooling2D(pool_size=(h, 1), name="ca_pool_h")(feature)          # 1 x w x ch
    xw = tf.keras.layers.Permute((2, 1, 3), name="ca_permute_in")(xw)                         # w x 1 x ch
    y = tf.keras.layers.Concatenate(axis=1, name="ca_concat")([xh, xw])                       # (h+w) x 1 x ch
    y = tf.keras.layers.Conv2D(mid, 1, name="ca_reduce")(y)
    y = tf.keras.layers.BatchNormalization(name="ca_bn")(y)
    y = tf.keras.layers.Activation("hard_swish", name="ca_act")(y)
    yh = tf.keras.layers.Cropping2D(((0, w), (0, 0)), name="ca_split_h")(y)                    # h x 1 x mid
    yw = tf.keras.layers.Cropping2D(((h, 0), (0, 0)), name="ca_split_w")(y)                    # w x 1 x mid
    ah = tf.keras.layers.Conv2D(ch, 1, kernel_initializer="zeros", activation="sigmoid", name="ca_gate_h")(yh)
    aw = tf.keras.layers.Conv2D(ch, 1, kernel_initializer="zeros", activation="sigmoid", name="ca_gate_w")(yw)
    ah = tf.keras.layers.Rescaling(2.0, name="ca_scale_h")(ah)                                  # h x 1 x ch
    aw = tf.keras.layers.Rescaling(2.0, name="ca_scale_w")(aw)
    aw = tf.keras.layers.Permute((2, 1, 3), name="ca_permute_out")(aw)                        # 1 x w x ch
    return tf.keras.layers.Multiply(name="ca_apply")([feature, ah, aw])


def backbone_of(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name.startswith("mobilenet"):
            return layer
    raise ValueError("MobileNet backbone not found")


def lr_groups(infer, cfg):
    """Variables split into learning-rate groups. Without LLRD: one group, factor 1.
    With LLRD (discriminative fine-tuning, Howard & Ruder 2018): each lower group gets lr / factor:
      group 0 (factor 1):       head + MobileNet blocks 12-13
      group 1 (1/2.6):          blocks 7-11
      group 2 (1/2.6^2):        stem + blocks 1-6"""
    allv = list(infer.trainable_variables)
    if not cfg.get("llrd"):
        return [(1.0, allv)]
    f = float(cfg["llrd"]["factor"])
    base = backbone_of(infer)
    gid = {}
    for layer in base.layers:
        m = re.match(r"conv_(?:dw|pw)_(\d+)", layer.name)
        block = int(m.group(1)) if m else (0 if layer.name.startswith("conv1") else 99)
        g = 0 if block >= 12 else (1 if block >= 7 else 2)
        for v in layer.trainable_weights:
            gid[id(v)] = g
    groups = [[], [], []]
    for v in allv:
        groups[gid.get(id(v), 0)].append(v)
    return [(1.0, groups[0]), (1.0 / f, groups[1]), (1.0 / f ** 2, groups[2])]


def average_weights(paths, cfg):
    """Stochastic weight averaging of saved epochs (plain mean of all weights, as in the paper)."""
    models = [tf.keras.models.load_model(p, compile=False) for p in paths]
    ws = [m.get_weights() for m in models]
    avg = [np.mean([w[i] for w in ws], axis=0).astype(ws[0][i].dtype) for i in range(len(ws[0]))]
    out = models[-1]
    out.set_weights(avg)
    return out
