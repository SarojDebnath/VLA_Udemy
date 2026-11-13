"""Language conditioning for TableTop policies (S09): frozen CLIP text embeddings, paraphrases, FiLM vs concat."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .chunk_policies import ChunkPolicy
from .models import IMAGENET_MEAN, IMAGENET_STD, random_shift

TRAIN_TEMPLATES = {
    "pick": ["pick up the {o} cube", "lift the {o} cube", "grab the {o} cube"],
    "place": ["put the {o} cube in the {t} tray", "place the {o} cube into the {t} tray", "move the {o} cube to the {t} tray"],
    "stack": ["stack the {o} cube on the {t} cube", "put the {o} cube on top of the {t} cube", "place the {o} cube onto the {t} cube"],
}
HELDOUT_TEMPLATES = {
    "pick": ["raise the {o} block off the table", "could you pick the {o} block up"],
    "place": ["drop the {o} block in the {t} container", "the {o} block goes into the {t} tray"],
    "stack": ["build a tower with the {o} block above the {t} block", "set the {o} block on the {t} block"],
}


def render_instruction(task, template):
    return template.format(o=task["obj"], t=task.get("target", ""))


class TextEmbedder:
    """Frozen CLIP text tower with a cache (instructions repeat a lot)."""

    def __init__(self, model_id="openai/clip-vit-base-patch32", device="cpu"):
        from transformers import CLIPModel, CLIPTokenizer
        self.tok = CLIPTokenizer.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.device, self.cache = device, {}
        self.dim = self.model.config.projection_dim

    @torch.no_grad()
    def __call__(self, texts):
        missing = [t for t in texts if t not in self.cache]
        if missing:
            x = self.tok(missing, padding=True, return_tensors="pt").to(self.device)
            out = self.model.get_text_features(**x)
            out = getattr(out, "pooler_output", out)
            for t, e in zip(missing, F.normalize(out, dim=-1).cpu()):
                self.cache[t] = e
        return torch.stack([self.cache[t] for t in texts])


class LangDataset(torch.utils.data.Dataset):
    """Wraps TableTopDataset: replaces the one-hot 'task' by a CLIP embedding of a (paraphrased) instruction."""

    def __init__(self, base, embedder, templates=TRAIN_TEMPLATES, paraphrase=True, seed=0):
        self.base, self.templates, self.paraphrase = base, templates, paraphrase
        self.rng = np.random.default_rng(seed)
        texts = set()
        for ep in base.eps:
            t = ep["task"]
            texts.add(ep["instruction"])
            texts.update(render_instruction(t, tpl) for tpl in templates[t["type"]])
        self.emb = {t: e for t, e in zip(sorted(texts), embedder(sorted(texts)))}

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        e, _ = self.base.index[i]
        task = self.base.eps[e]["task"]
        text = render_instruction(task, self.rng.choice(self.templates[task["type"]])) if self.paraphrase else self.base.eps[e]["instruction"]
        item["lang"] = self.emb[text]
        return item

    def __getattr__(self, name):          # act_norm, to_absolute, to_command, action_mode, ... come from the base dataset
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    @property
    def act_norm(self):
        return self.base.act_norm

    @property
    def state_norm(self):
        return self.base.state_norm


class FiLMResNet(nn.Module):
    """ResNet-18 stages 1-2 (GroupNorm) FiLM-modulated by a conditioning vector (RT-1 / BC-Z style early fusion),
    followed by spatial-softmax keypoints (so 'where' survives pooling)."""

    def __init__(self, cond_dim, num_kp=32, out_dim=256):
        super().__init__()
        import torchvision
        from .models import SpatialSoftmax, group_norm
        r = group_norm(torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1))
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.stages = nn.ModuleList([r.layer1, r.layer2])           # 16x16 map at 128 px (same recipe as ImageEncoder)
        self.films = nn.ModuleList([nn.Linear(cond_dim, 2 * c) for c in (64, 128)])
        for f in self.films:
            nn.init.zeros_(f.weight); nn.init.zeros_(f.bias)
        self.kp, self.proj = SpatialSoftmax(128, num_kp), nn.Sequential(nn.Linear(2 * num_kp, out_dim), nn.ReLU())
        self.out_dim = out_dim

    def forward(self, x, cond):
        x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
        x = self.stem(x)
        for stage, film in zip(self.stages, self.films):
            g, b = film(cond).chunk(2, -1)
            x = stage(x) * (1 + g[..., None, None]) + b[..., None, None]
        return self.proj(self.kp(x))


class LangChunkRegression(nn.Module):
    """Chunked-regression pixel policy conditioned on language by 'concat' (late fusion) or 'film' (early fusion)."""

    def __init__(self, lang_dim=512, horizon=16, act_dim=7, cameras=("front", "wrist"), fusion="film"):
        super().__init__()
        self.H, self.A, self.fusion = horizon, act_dim, fusion
        self.cameras = cameras
        if fusion == "film":
            self.backbone = FiLMResNet(lang_dim)
        else:
            from .models import ImageEncoder
            self.backbone = ImageEncoder(pretrained=True)
        d = self.backbone.out_dim * len(cameras) + 8 + lang_dim
        self.head = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, horizon * act_dim))

    def features(self, b, augment):
        feats = []
        for c in self.cameras:
            x = random_shift(b[c]) if augment else b[c]
            feats.append(self.backbone(x, b["lang"]) if self.fusion == "film" else self.backbone(x))
        return torch.cat(feats + [b["state"], b["lang"]], -1)

    def loss(self, b):
        return F.l1_loss(self.head(self.features(b, True)).view(-1, self.H, self.A), b["action"])

    @torch.no_grad()
    def sample(self, b, nfe=None):
        return self.head(self.features(b, False)).view(-1, self.H, self.A)


class LangChunkPolicy(ChunkPolicy):
    """ChunkPolicy that feeds an instruction embedding instead of the task one-hot. `instruction_fn(task)` picks the text."""

    def __init__(self, net, ds, env, embedder, instruction_fn=None, **kw):
        self.embedder = embedder
        self.instruction_fn = instruction_fn or (lambda task: task["instruction"])
        super().__init__(net, ds, env, **kw)

    def _batch(self, obs):
        b = {"state": torch.from_numpy(self.ds.state_norm.norm(obs["state"]).astype(np.float32))[None],
             "lang": self.embedder([self.instruction_fn(self.env.task)])}
        for c in self.net.cameras:
            b[c] = torch.from_numpy(np.ascontiguousarray(obs["images"][c])).permute(2, 0, 1).float().div(255)[None]
        return {k: v.to(self.device) for k, v in b.items()}
