"""Reusable network pieces and policy wrappers for TableTop imitation learning."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def random_shift(x, pad=8):
    """DrQ-style augmentation: replicate-pad then randomly crop back to the original size (per sample)."""
    n, c, h, w = x.shape
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    eps = 1.0 / (h + 2 * pad)
    arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad, device=x.device)[:h]
    base = torch.stack(torch.meshgrid(arange, arange, indexing="xy"), -1).unsqueeze(0).repeat(n, 1, 1, 1)
    shift = torch.randint(0, 2 * pad + 1, (n, 1, 1, 2), device=x.device).float() * 2.0 / (h + 2 * pad)
    return F.grid_sample(x, base + shift, padding_mode="zeros", align_corners=False)


def group_norm(module, group_size=16):
    """Replace every BatchNorm2d with GroupNorm (Diffusion Policy recipe): no batch statistics, so small/noisy
    batches and train-vs-eval mode behave the same, which matters for closed-loop control."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(max(1, child.num_features // group_size), child.num_features))
        else:
            group_norm(child, group_size)
    return module


class ImageEncoder(nn.Module):
    """ResNet-18 (ImageNet weights).
    spatial=False: stages up to layer2 with GroupNorm -> 16x16 feature map at 128 px -> spatial-softmax keypoints
        (x, y per keypoint) -> MLP. Global average pooling would discard *where* things are; keypoints keep it
        (robomimic / Diffusion Policy). The 16x16 map (not layer3's 8x8) gives the precision a 4 cm cube needs.
    spatial=True : layer-4 feature map (B, 512, 4, 4) for token-based heads (ACT)."""

    def __init__(self, pretrained=True, spatial=False, num_kp=32, out_dim=256):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=weights)
        children = list(r.children())
        self.spatial = spatial
        if spatial:
            self.trunk = nn.Sequential(*children[:-3])
            self.layer4 = children[-3]
            self.out_dim = 512
        else:
            self.trunk = group_norm(nn.Sequential(*children[:6]))      # conv1, bn1, relu, maxpool, layer1, layer2
            self.kp = SpatialSoftmax(128, num_kp)
            self.proj = nn.Sequential(nn.Linear(2 * num_kp, out_dim), nn.ReLU())
            self.out_dim = out_dim

    def forward(self, x):
        x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
        f = self.trunk(x)
        if self.spatial:
            return self.layer4(f)
        return self.proj(self.kp(f))


class SpatialSoftmax(nn.Module):
    """1x1 conv to K heatmaps -> softmax over pixels -> expected (x, y) of each heatmap in [-1, 1] (Levine et al. 2016)."""

    def __init__(self, in_ch, num_kp=32):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_kp, 1)
        self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(self, f):
        B, _, H, W = f.shape
        h = self.conv(f).flatten(2) / self.log_temp.exp()
        p = h.softmax(-1)
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, H, device=f.device), torch.linspace(-1, 1, W, device=f.device), indexing="ij")
        return torch.cat([(p * xs.reshape(1, 1, -1)).sum(-1), (p * ys.reshape(1, 1, -1)).sum(-1)], -1)


class PixelBCNet(nn.Module):
    """Shared encoder over each camera + state + task one-hot -> action (or action chunk)."""

    def __init__(self, cameras=("front", "wrist"), state_dim=8, task_dim=15, act_dim=7, horizon=1, hidden=512, pretrained=True):
        super().__init__()
        self.cameras, self.horizon, self.act_dim = cameras, horizon, act_dim
        self.encoder = ImageEncoder(pretrained)
        d = self.encoder.out_dim * len(cameras) + state_dim + task_dim
        self.head = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, act_dim * horizon))

    def features(self, batch, augment=False):
        feats = []
        for c in self.cameras:
            x = batch[c]
            if augment:
                x = random_shift(x)
            feats.append(self.encoder(x))
        return torch.cat(feats + [batch["state"], batch["task"]], -1)

    def forward(self, batch, augment=False):
        out = self.head(self.features(batch, augment))
        return out if self.horizon == 1 else out.view(-1, self.horizon, self.act_dim)


class TorchPolicy:
    """Wraps a network trained on TableTopDataset tensors into the env policy interface (act(obs) -> 7-D action)."""

    def __init__(self, net, dataset, device="cuda", execute_steps=1, env=None):
        self.net, self.ds, self.device, self.k = net.eval(), dataset, device, execute_steps
        self.env = env
        self.queue = []

    def reset(self):
        self.queue = []

    def batch_from_obs(self, obs):
        from .torch_data import task_vector
        b = {"state": torch.from_numpy(self.ds.state_norm.norm(obs["state"]).astype(np.float32))[None].to(self.device),
             "task": torch.from_numpy(task_vector(self.env.task))[None].to(self.device)}
        for c in self.net.cameras:
            b[c] = torch.from_numpy(np.ascontiguousarray(obs["images"][c])).permute(2, 0, 1).float().div(255)[None].to(self.device)
        return b

    def act(self, obs):
        if not self.queue:
            with torch.no_grad():
                out = self.net(self.batch_from_obs(obs)).cpu().numpy()[0]
            out = np.stack([self.ds.act_norm.unnorm(a) for a in out.reshape(-1, 7)])
            out = self.ds.to_absolute(out, obs) if hasattr(self.ds, "to_absolute") else out
            self.queue = list(out[: self.k])
        a = self.queue.pop(0)
        a = self.ds.to_command(a, self.env) if hasattr(self.ds, "to_command") else a.copy()
        a[6] = float(a[6] > 0.5)
        return a
