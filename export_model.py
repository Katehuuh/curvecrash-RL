"""
Export trained CurveCrash PPO actor weights to JSON for browser inference.

Usage:
    python export_model.py checkpoints_v5_1_main/agent_final.pt
    python export_model.py checkpoints_v5_1_main/agent_final.pt --verify
    python export_model.py checkpoints_impala_voronoi/agent_1000448.pt  # auto-detects IMPALA

Auto-detects checkpoint version by presence of keys:
  - v4 (legacy): no gru.* keys
  - v5:          gru.* keys, no cbam.* keys
  - v5.1:        gru.* + cbam.* keys, optionally spatial_attn.* keys
  - v6:          v5.1 with 6 input channels (GS++)
  - impala:      conv.0.conv.* keys (IMPALA-CNN residual backbone)

Outputs JSON with weights in TF.js-compatible format:
  Conv: [kH, kW, Cin, Cout]  (transposed from PyTorch [Cout, Cin, kH, kW])
  Linear: [in, out]          (transposed from PyTorch [out, in])
  GRU: split by gate (reset, update, new) with transposed kernels
"""
import argparse
import base64
import json

import numpy as np
import torch

from curvecrash_env_ffa import OBS_SIZE
from train_selfplay import detect_checkpoint_type, get_gru_hidden_size, get_input_channels


def _build_nhwc_perm(C, H, W):
    """Build permutation: NHWC flat index -> NCHW flat index."""
    perm = np.zeros(C * H * W, dtype=int)
    for c in range(C):
        for h in range(H):
            for w_idx in range(W):
                nhwc_idx = h * W * C + w_idx * C + c
                nchw_idx = c * H * W + h * W + w_idx
                perm[nhwc_idx] = nchw_idx
    return perm


def _is_impala(state_dict):
    """Detect IMPALA-CNN architecture from weight keys."""
    return any(k.startswith("conv.0.conv.") for k in state_dict)


def _get_impala_channels(state_dict):
    """Get IMPALA channel sizes from weight shapes."""
    channels = []
    for i in range(10):
        k = f"conv.{i}.conv.weight"
        if k in state_dict:
            channels.append(state_dict[k].shape[0])
        else:
            break
    return tuple(channels)


def _conv_to_tfjs(w_pt):
    """Convert PyTorch conv weight [Cout, Cin, kH, kW] to TF.js [kH, kW, Cin, Cout]."""
    return w_pt.cpu().numpy().transpose(2, 3, 1, 0)


def export_actor_json(state_dict, output_path, compact=True):
    """Extract actor weights and save as JSON in TF.js format."""
    is_imp = _is_impala(state_dict)

    if is_imp:
        _export_impala(state_dict, output_path, compact)
    else:
        _export_naturecnn(state_dict, output_path, compact)


def _export_impala(state_dict, output_path, compact=True):
    """Export IMPALA-CNN architecture."""
    channels = _get_impala_channels(state_dict)
    n_input_channels = state_dict["conv.0.conv.weight"].shape[1]
    gru_hidden = get_gru_hidden_size(state_dict)
    has_cbam = any(k.startswith("cbam.") for k in state_dict)
    has_spatial = any(k.startswith("spatial_attn.") for k in state_dict)

    layers = []

    # IMPALA ConvSequences
    # Each stage: conv(3x3,same) -> maxpool(3,s2,same) -> res1(2 convs) -> res2(2 convs)
    for si in range(len(channels)):
        stage = {}
        # Main conv
        stage["conv_weight"] = _conv_to_tfjs(state_dict[f"conv.{si}.conv.weight"]).tolist()
        stage["conv_bias"] = state_dict[f"conv.{si}.conv.bias"].cpu().numpy().tolist()
        # ResBlock 1
        stage["res1_conv1_weight"] = _conv_to_tfjs(state_dict[f"conv.{si}.res1.conv1.weight"]).tolist()
        stage["res1_conv1_bias"] = state_dict[f"conv.{si}.res1.conv1.bias"].cpu().numpy().tolist()
        stage["res1_conv2_weight"] = _conv_to_tfjs(state_dict[f"conv.{si}.res1.conv2.weight"]).tolist()
        stage["res1_conv2_bias"] = state_dict[f"conv.{si}.res1.conv2.bias"].cpu().numpy().tolist()
        # ResBlock 2
        stage["res2_conv1_weight"] = _conv_to_tfjs(state_dict[f"conv.{si}.res2.conv1.weight"]).tolist()
        stage["res2_conv1_bias"] = state_dict[f"conv.{si}.res2.conv1.bias"].cpu().numpy().tolist()
        stage["res2_conv2_weight"] = _conv_to_tfjs(state_dict[f"conv.{si}.res2.conv2.weight"]).tolist()
        stage["res2_conv2_bias"] = state_dict[f"conv.{si}.res2.conv2.bias"].cpu().numpy().tolist()
        stage["out_channels"] = channels[si]
        layers.append({"type": "impala_conv_sequence", **stage})

    # CBAM (if present, applied after all conv stages + ReLU)
    if has_cbam:
        fc0_w = state_dict["cbam.fc.0.weight"].cpu().numpy()
        fc2_w = state_dict["cbam.fc.2.weight"].cpu().numpy()
        sp_w = state_dict["cbam.spatial_conv.weight"].cpu().numpy()
        sp_w_tfjs = sp_w.transpose(2, 3, 1, 0)
        layers.append({
            "type": "cbam",
            "channel_fc0_weight": fc0_w.T.tolist(),
            "channel_fc2_weight": fc2_w.T.tolist(),
            "spatial_conv_weight": sp_w_tfjs.tolist(),
            "reduction": fc0_w.shape[0],
            "channels": fc0_w.shape[1],
        })

    # Spatial self-attention (if present)
    if has_spatial:
        qkv_w = state_dict["spatial_attn.qkv.weight"].cpu().numpy()
        proj_w = state_dict["spatial_attn.proj.weight"].cpu().numpy()
        proj_b = state_dict["spatial_attn.proj.bias"].cpu().numpy()
        C = qkv_w.shape[1]
        H = W = 16  # IMPALA 128->64->32->16
        layers.append({
            "type": "spatial_self_attention",
            "qkv_weight": qkv_w.T.tolist(),
            "proj_weight": proj_w.T.tolist(),
            "proj_bias": proj_b.tolist(),
            "num_heads": 4,
            "dim": C,
            "tokens": H * W,
        })

    # FC hidden (after flatten)
    # IMPALA: feature map is channels[-1] * 16 * 16
    C_feat = channels[-1]
    H_feat = W_feat = 16  # after 3 stages of stride-2 from 128
    w = state_dict["fc.1.weight"].cpu().numpy()  # [256, 8192]
    b = state_dict["fc.1.bias"].cpu().numpy()
    perm = _build_nhwc_perm(C_feat, H_feat, W_feat)
    w_permuted = w[:, perm]
    layers.append({
        "type": "linear",
        "weight": w_permuted.T.tolist(),
        "bias": b.tolist(),
        "shape_pt": list(w.shape),
        "shape_tfjs": list(w_permuted.T.shape),
    })

    # GRU
    _append_gru(state_dict, layers, gru_hidden)

    # Actor head
    _append_actor(state_dict, layers)

    model_json = {
        "version": "impala",
        "arch": "impala",
        "obs_size": OBS_SIZE,
        "n_input_channels": n_input_channels,
        "impala_channels": list(channels),
        "has_cbam": has_cbam,
        "has_spatial_attn": has_spatial,
        "gru_hidden": gru_hidden,
        "layers": layers,
    }
    _save_and_report(model_json, output_path, compact)


def _export_naturecnn(state_dict, output_path, compact=True):
    """Export NatureCNN architecture (v4/v5/v5.1/v6)."""
    version = detect_checkpoint_type(state_dict)
    has_gru = version in ("v5", "v5_1", "v6")
    has_cbam = version in ("v5_1", "v6")
    has_spatial = any(k.startswith("spatial_attn.") for k in state_dict)
    gru_hidden = get_gru_hidden_size(state_dict) if has_gru else 0
    n_input_channels = get_input_channels(state_dict)

    layers = []

    # 3 Conv layers (indices 0, 2, 4 in nn.Sequential)
    conv_configs = [
        (0, 4, 0),
        (2, 2, 0),
        (4, 2, 0),
    ]
    for idx, stride, padding in conv_configs:
        w = state_dict[f"network.{idx}.weight"].cpu().numpy()
        w_tfjs = w.transpose(2, 3, 1, 0)
        b = state_dict[f"network.{idx}.bias"].cpu().numpy()
        layers.append({
            "type": "conv2d",
            "weight": w_tfjs.tolist(),
            "bias": b.tolist(),
            "stride": stride,
            "padding": padding,
            "shape_pt": list(w.shape),
            "shape_tfjs": list(w_tfjs.shape),
        })

    # CBAM
    if has_cbam:
        fc0_w = state_dict["cbam.fc.0.weight"].cpu().numpy()
        fc2_w = state_dict["cbam.fc.2.weight"].cpu().numpy()
        sp_w = state_dict["cbam.spatial_conv.weight"].cpu().numpy()
        sp_w_tfjs = sp_w.transpose(2, 3, 1, 0)
        layers.append({
            "type": "cbam",
            "channel_fc0_weight": fc0_w.T.tolist(),
            "channel_fc2_weight": fc2_w.T.tolist(),
            "spatial_conv_weight": sp_w_tfjs.tolist(),
            "reduction": fc0_w.shape[0],
            "channels": fc0_w.shape[1],
        })

    # Spatial self-attention
    if has_spatial:
        qkv_w = state_dict["spatial_attn.qkv.weight"].cpu().numpy()
        proj_w = state_dict["spatial_attn.proj.weight"].cpu().numpy()
        proj_b = state_dict["spatial_attn.proj.bias"].cpu().numpy()
        layers.append({
            "type": "spatial_self_attention",
            "qkv_weight": qkv_w.T.tolist(),
            "proj_weight": proj_w.T.tolist(),
            "proj_bias": proj_b.tolist(),
            "num_heads": 4,
            "dim": 64,
            "tokens": 36,
        })

    # FC hidden
    C, H, W = 64, 6, 6
    fc_key = "fc.1" if has_cbam else "network.7"
    w = state_dict[f"{fc_key}.weight"].cpu().numpy()
    b = state_dict[f"{fc_key}.bias"].cpu().numpy()
    perm = _build_nhwc_perm(C, H, W)
    w_permuted = w[:, perm]
    layers.append({
        "type": "linear",
        "weight": w_permuted.T.tolist(),
        "bias": b.tolist(),
        "shape_pt": list(w.shape),
        "shape_tfjs": list(w_permuted.T.shape),
    })

    # GRU
    if has_gru:
        _append_gru(state_dict, layers, gru_hidden)

    # Actor head
    _append_actor(state_dict, layers)

    model_json = {
        "version": version,
        "arch": "naturecnn",
        "obs_size": OBS_SIZE,
        "n_input_channels": n_input_channels,
        "has_cbam": has_cbam,
        "has_spatial_attn": has_spatial,
        "gru_hidden": gru_hidden,
        "layers": layers,
    }
    _save_and_report(model_json, output_path, compact)


def _append_gru(state_dict, layers, hidden_size):
    """Append GRU layer to layers list."""
    w_ih = state_dict["gru.weight_ih_l0"].cpu().numpy()
    w_hh = state_dict["gru.weight_hh_l0"].cpu().numpy()
    b_ih = state_dict["gru.bias_ih_l0"].cpu().numpy()
    b_hh = state_dict["gru.bias_hh_l0"].cpu().numpy()

    gate_names = ["reset", "update", "new"]
    gates = {}
    for gi, gname in enumerate(gate_names):
        s = gi * hidden_size
        e = (gi + 1) * hidden_size
        gates[gname] = {
            "kernel_input": w_ih[s:e].T.tolist(),
            "kernel_hidden": w_hh[s:e].T.tolist(),
            "bias_input": b_ih[s:e].tolist(),
            "bias_hidden": b_hh[s:e].tolist(),
        }

    layers.append({
        "type": "gru",
        "units": hidden_size,
        "input_size": w_ih.shape[1],
        "gates": gates,
        "gate_order": "reset,update,new",
        "formula": "r=sig(x@Wi_r+bi_r+h@Wh_r+bh_r); "
                   "z=sig(x@Wi_z+bi_z+h@Wh_z+bh_z); "
                   "n=tanh(x@Wi_n+bi_n+r*(h@Wh_n+bh_n)); "
                   "h_new=(1-z)*n+z*h",
    })


def _append_actor(state_dict, layers):
    """Append actor head to layers list."""
    w = state_dict["actor.weight"].cpu().numpy()
    b = state_dict["actor.bias"].cpu().numpy()
    layers.append({
        "type": "linear",
        "weight": w.T.tolist(),
        "bias": b.tolist(),
        "shape_pt": list(w.shape),
        "shape_tfjs": list(w.T.shape),
    })


def _compact_value(v):
    """Convert a list/nested-list of floats to base64 float32 string + shape."""
    arr = np.array(v, dtype=np.float32)
    return base64.b64encode(arr.tobytes()).decode('ascii'), list(arr.shape)


def _compact_layer(layer):
    """Convert all weight arrays in a layer dict to base64."""
    out = {}
    for k, v in layer.items():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (list, float, int)):
            b64, shape = _compact_value(v)
            out[k] = b64
            out[k + '_shape'] = shape
        elif isinstance(v, dict):
            out[k] = {}
            for gk, gv in v.items():
                if isinstance(gv, dict):
                    out[k][gk] = {}
                    for kk, vv in gv.items():
                        if isinstance(vv, list) and len(vv) > 0:
                            b64, shape = _compact_value(vv)
                            out[k][gk][kk] = b64
                            out[k][gk][kk + '_shape'] = shape
                        else:
                            out[k][gk][kk] = vv
                else:
                    out[k][gk] = gv
        else:
            out[k] = v
    return out


def _save_and_report(model_json, output_path, compact=True):
    """Save JSON and print stats."""
    if compact:
        model_json['encoding'] = 'base64_float32'
        model_json['layers'] = [_compact_layer(l) for l in model_json['layers']]

    with open(output_path, "w") as f:
        json.dump(model_json, f)

    size_bytes = len(json.dumps(model_json))
    total_params = 0
    for l in model_json["layers"]:
        lt = l["type"]
        if lt == "gru":
            hs = l["units"]
            ins = l["input_size"]
            total_params += 3 * (ins * hs + hs + hs * hs + hs)
        elif lt == "cbam":
            red = l["reduction"]
            ch = l["channels"]
            total_params += ch * red + red * ch + 7 * 7 * 2
        elif lt == "spatial_self_attention":
            dim = l["dim"]
            total_params += dim * dim * 3 + dim * dim + dim
        elif lt == "impala_conv_sequence":
            # main conv + 2 res blocks * 2 convs each = 5 conv layers
            oc = l["out_channels"]
            # main conv: count from weight shape
            w = np.array(l["conv_weight"])
            total_params += np.prod(w.shape) + oc
            for prefix in ["res1_conv1", "res1_conv2", "res2_conv1", "res2_conv2"]:
                w = np.array(l[f"{prefix}_weight"])
                total_params += np.prod(w.shape) + oc
        elif lt == "conv2d":
            w_shape = l["shape_pt"]
            total_params += np.prod(w_shape) + len(l["bias"])
        elif lt == "linear":
            w_shape = l["shape_pt"]
            total_params += np.prod(w_shape) + len(l["bias"])

    arch = model_json.get("arch", model_json["version"])
    version = model_json["version"]
    n_ch = model_json["n_input_channels"]
    has_cbam = model_json["has_cbam"]
    has_spatial = model_json.get("has_spatial_attn", False)
    gru_h = model_json["gru_hidden"]
    print(f"Exported {arch}/{version}: {len(model_json['layers'])} layers, {total_params:,} params")
    print(f"  Input channels: {n_ch}, CBAM: {has_cbam}, "
          f"Spatial attn: {has_spatial}, GRU hidden: {gru_h}")
    if "impala_channels" in model_json:
        print(f"  IMPALA channels: {model_json['impala_channels']}")
    print(f"JSON size: {size_bytes / 1024:.0f} KB ({size_bytes / 1024 / 1024:.1f} MB)")
    print(f"Saved to: {output_path}")


def verify_forward_pass(state_dict, output_path):
    """Load JSON and verify forward pass matches PyTorch."""
    is_imp = _is_impala(state_dict)
    has_cbam = any(k.startswith("cbam.") for k in state_dict)
    has_spatial = any(k.startswith("spatial_attn.") for k in state_dict)
    gru_hidden = get_gru_hidden_size(state_dict)

    if is_imp:
        n_input_channels = state_dict["conv.0.conv.weight"].shape[1]
    else:
        n_input_channels = get_input_channels(state_dict)

    with open(output_path) as f:
        model_json = json.load(f)

    x = np.random.rand(1, n_input_channels, OBS_SIZE, OBS_SIZE).astype(np.float32)

    # --- PyTorch forward ---
    if is_imp:
        from experiments import ImpalaCNNAgent
        channels = _get_impala_channels(state_dict)
        agent = ImpalaCNNAgent(
            n_input_channels=n_input_channels,
            gru_hidden=gru_hidden,
            channels=channels,
            use_cbam=has_cbam,
        )
    else:
        version = detect_checkpoint_type(state_dict)
        if version in ("v5_1", "v6"):
            from train_selfplay import Agent
            agent = Agent(use_spatial_attn=has_spatial, n_input_channels=n_input_channels)
        elif version == "v5":
            from train_selfplay import V5Agent
            agent = V5Agent()
        else:
            from train_selfplay import LegacyAgent
            agent = LegacyAgent()

    agent.load_state_dict(state_dict)
    agent.eval()

    with torch.no_grad():
        xt = torch.tensor(x)
        if is_imp:
            feat = agent.conv(xt)
            feat = torch.nn.functional.relu(feat)
            if has_cbam:
                feat = agent.cbam(feat)
            hidden = agent.fc(feat)
            gru_state = torch.zeros(1, 1, gru_hidden)
            _, gru_state = agent.gru(hidden.unsqueeze(0), gru_state)
            logits_pt = agent.actor(gru_state.squeeze(0)).numpy().flatten()
        elif hasattr(agent, 'cbam'):
            feat = agent.network(xt)
            feat = agent.cbam(feat)
            if has_spatial:
                feat = agent.spatial_attn(feat)
            hidden = agent.fc(feat)
            gru_state = torch.zeros(1, 1, gru_hidden)
            _, gru_state = agent.gru(hidden.unsqueeze(0), gru_state)
            logits_pt = agent.actor(gru_state.squeeze(0)).numpy().flatten()
        elif gru_hidden > 0:
            hidden = agent.network(xt)
            gru_state = torch.zeros(1, 1, gru_hidden)
            _, gru_state = agent.gru(hidden.unsqueeze(0), gru_state)
            logits_pt = agent.actor(gru_state.squeeze(0)).numpy().flatten()
        else:
            hidden = agent.network(xt)
            logits_pt = agent.actor(hidden).numpy().flatten()

    # --- Numpy (simulates TF.js NHWC forward pass) ---
    def relu(a):
        return np.maximum(a, 0)

    def sigmoid(a):
        return 1.0 / (1.0 + np.exp(-a))

    def conv2d_nhwc(inp, w, b, stride, padding='valid'):
        _, h_in, w_in, c_in = inp.shape
        kh, kw, _, c_out = w.shape
        if padding == 'same':
            pad_h = (kh - 1) // 2
            pad_w = (kw - 1) // 2
            inp = np.pad(inp, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)))
            _, h_in, w_in, c_in = inp.shape
        h_out = (h_in - kh) // stride + 1
        w_out = (w_in - kw) // stride + 1
        out = np.zeros((1, h_out, w_out, c_out), dtype=np.float32)
        for co in range(c_out):
            for i in range(h_out):
                for j in range(w_out):
                    patch = inp[0, i*stride:i*stride+kh, j*stride:j*stride+kw, :]
                    out[0, i, j, co] = np.sum(patch * w[:, :, :, co]) + b[co]
        return out

    def maxpool_same(inp, pool_size=3, stride=2):
        _, h_in, w_in, c = inp.shape
        pad = (pool_size - 1) // 2
        inp = np.pad(inp, ((0, 0), (pad, pad), (pad, pad), (0, 0)), constant_values=-1e9)
        _, h_p, w_p, _ = inp.shape
        h_out = (h_in + 2 * pad - pool_size) // stride + 1
        w_out = (w_in + 2 * pad - pool_size) // stride + 1
        out = np.zeros((1, h_out, w_out, c), dtype=np.float32)
        for i in range(h_out):
            for j in range(w_out):
                patch = inp[0, i*stride:i*stride+pool_size, j*stride:j*stride+pool_size, :]
                out[0, i, j, :] = patch.max(axis=(0, 1))
        return out

    # Convert NCHW -> NHWC
    a = x.transpose(0, 2, 3, 1)
    json_layers = model_json["layers"]
    layer_idx = 0

    if is_imp:
        # IMPALA ConvSequences
        for si in range(len(model_json.get("impala_channels", [16, 32, 32]))):
            stage = json_layers[layer_idx]
            # Main conv (same padding)
            w = np.array(stage["conv_weight"], dtype=np.float32)
            b = np.array(stage["conv_bias"], dtype=np.float32)
            a = conv2d_nhwc(a, w, b, 1, 'same')
            # MaxPool
            a = maxpool_same(a, 3, 2)
            # ResBlock 1: relu -> conv -> relu -> conv -> +residual
            residual = a
            a = relu(a)
            w = np.array(stage["res1_conv1_weight"], dtype=np.float32)
            b = np.array(stage["res1_conv1_bias"], dtype=np.float32)
            a = conv2d_nhwc(a, w, b, 1, 'same')
            a = relu(a)
            w = np.array(stage["res1_conv2_weight"], dtype=np.float32)
            b = np.array(stage["res1_conv2_bias"], dtype=np.float32)
            a = conv2d_nhwc(a, w, b, 1, 'same')
            a = a + residual
            # ResBlock 2
            residual = a
            a = relu(a)
            w = np.array(stage["res2_conv1_weight"], dtype=np.float32)
            b = np.array(stage["res2_conv1_bias"], dtype=np.float32)
            a = conv2d_nhwc(a, w, b, 1, 'same')
            a = relu(a)
            w = np.array(stage["res2_conv2_weight"], dtype=np.float32)
            b = np.array(stage["res2_conv2_bias"], dtype=np.float32)
            a = conv2d_nhwc(a, w, b, 1, 'same')
            a = a + residual
            layer_idx += 1
        # ReLU after all conv stages
        a = relu(a)
    else:
        # NatureCNN Conv layers
        for i in range(3):
            l = json_layers[layer_idx]
            w = np.array(l["weight"], dtype=np.float32)
            b = np.array(l["bias"], dtype=np.float32)
            a = relu(conv2d_nhwc(a, w, b, l["stride"]))
            layer_idx += 1

    # CBAM
    if has_cbam:
        l = json_layers[layer_idx]
        fc0_w = np.array(l["channel_fc0_weight"], dtype=np.float32)
        fc2_w = np.array(l["channel_fc2_weight"], dtype=np.float32)
        avg_pool = a.mean(axis=(1, 2), keepdims=False).reshape(1, -1)
        max_pool = a.max(axis=1).max(axis=1, keepdims=False).reshape(1, -1)
        avg_fc = relu(avg_pool @ fc0_w) @ fc2_w
        max_fc = relu(max_pool @ fc0_w) @ fc2_w
        ch_att = sigmoid(avg_fc + max_fc).reshape(1, 1, 1, -1)
        a = a * ch_att
        avg_sp = a.mean(axis=3, keepdims=True)
        max_sp = a.max(axis=3, keepdims=True)
        sp_cat = np.concatenate([avg_sp, max_sp], axis=3)
        sp_w = np.array(l["spatial_conv_weight"], dtype=np.float32)
        sp_padded = np.pad(sp_cat, ((0, 0), (3, 3), (3, 3), (0, 0)))
        sp_conv = conv2d_nhwc(sp_padded, sp_w, np.zeros(1, dtype=np.float32), 1)
        sp_att = sigmoid(sp_conv)
        a = a * sp_att
        layer_idx += 1

    # Spatial self-attention
    if has_spatial:
        l = json_layers[layer_idx]
        qkv_w = np.array(l["qkv_weight"], dtype=np.float32)
        proj_w = np.array(l["proj_weight"], dtype=np.float32)
        proj_b = np.array(l["proj_bias"], dtype=np.float32)
        num_heads = l["num_heads"]
        dim = l["dim"]
        head_dim = dim // num_heads
        scale = head_dim ** -0.5
        _, H, W, C = a.shape
        tokens = a.reshape(1, H * W, C)
        qkv = tokens @ qkv_w
        qkv = qkv.reshape(1, H * W, 3, num_heads, head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale
        attn_max = attn.max(axis=-1, keepdims=True)
        attn_exp = np.exp(attn - attn_max)
        attn = attn_exp / attn_exp.sum(axis=-1, keepdims=True)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(1, H * W, C)
        out = out @ proj_w + proj_b
        tokens = tokens + out
        a = tokens.reshape(1, H, W, C)
        layer_idx += 1

    # Flatten + FC hidden
    a = a.reshape(1, -1)
    fc_layer = json_layers[layer_idx]
    w = np.array(fc_layer["weight"], dtype=np.float32)
    b = np.array(fc_layer["bias"], dtype=np.float32)
    a = relu(a @ w + b)
    layer_idx += 1

    # GRU
    if gru_hidden > 0:
        gru_layer = json_layers[layer_idx]
        gates = gru_layer["gates"]
        h_prev = np.zeros((1, gru_hidden), dtype=np.float32)
        Wi_r = np.array(gates["reset"]["kernel_input"], dtype=np.float32)
        Wh_r = np.array(gates["reset"]["kernel_hidden"], dtype=np.float32)
        bi_r = np.array(gates["reset"]["bias_input"], dtype=np.float32)
        bh_r = np.array(gates["reset"]["bias_hidden"], dtype=np.float32)
        Wi_z = np.array(gates["update"]["kernel_input"], dtype=np.float32)
        Wh_z = np.array(gates["update"]["kernel_hidden"], dtype=np.float32)
        bi_z = np.array(gates["update"]["bias_input"], dtype=np.float32)
        bh_z = np.array(gates["update"]["bias_hidden"], dtype=np.float32)
        Wi_n = np.array(gates["new"]["kernel_input"], dtype=np.float32)
        Wh_n = np.array(gates["new"]["kernel_hidden"], dtype=np.float32)
        bi_n = np.array(gates["new"]["bias_input"], dtype=np.float32)
        bh_n = np.array(gates["new"]["bias_hidden"], dtype=np.float32)
        r = sigmoid(a @ Wi_r + bi_r + h_prev @ Wh_r + bh_r)
        z = sigmoid(a @ Wi_z + bi_z + h_prev @ Wh_z + bh_z)
        n = np.tanh(a @ Wi_n + bi_n + r * (h_prev @ Wh_n + bh_n))
        a = (1 - z) * n + z * h_prev
        layer_idx += 1

    # Actor head
    actor_layer = json_layers[layer_idx]
    w = np.array(actor_layer["weight"], dtype=np.float32)
    b = np.array(actor_layer["bias"], dtype=np.float32)
    logits_np = (a @ w + b).flatten()

    max_diff = np.max(np.abs(logits_pt - logits_np))
    arch = "impala" if is_imp else detect_checkpoint_type(state_dict)
    print(f"\nVerification ({arch}):")
    print(f"  PyTorch logits:  {logits_pt}")
    print(f"  Numpy logits:    {logits_np}")
    print(f"  Max abs diff:    {max_diff:.8f}")
    if max_diff < 1e-3:
        print("  PASS")
    else:
        print("  WARN -- difference exceeds tolerance")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", help="Path to .pt checkpoint")
    p.add_argument("--output", "-o", default=None,
                   help="Output path (default: auto based on version)")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--no-compact", action="store_true",
                   help="Disable base64 compact encoding (outputs raw JSON arrays)")
    args = p.parse_args()

    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    if args.output is None:
        if _is_impala(state_dict):
            args.output = "curvecrash_impala_weights.json"
        else:
            version = detect_checkpoint_type(state_dict)
            version_suffix = {"legacy": "v4", "v5": "v5", "v5_1": "v5_1", "v6": "v6"}[version]
            args.output = f"curvecrash_{version_suffix}_weights.json"

    export_actor_json(state_dict, args.output, compact=not args.no_compact)

    if args.verify:
        verify_forward_pass(state_dict, args.output)


if __name__ == "__main__":
    main()
