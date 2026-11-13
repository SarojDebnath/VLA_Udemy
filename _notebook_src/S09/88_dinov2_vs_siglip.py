# %% [markdown]
# # 88 · DINOv2 vs SigLIP features: spatial vs semantic (why OpenVLA fuses both)

# %%
import time, numpy as np, torch, matplotlib.pyplot as plt
from transformers import AutoModel, AutoImageProcessor, SiglipModel, SiglipProcessor
from sklearn.linear_model import RidgeCV, LogisticRegression
from vlacourse import TableTop

device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def features(images, kind):
    out_cls, out_patch = [], []
    for i in range(0, len(images), 32):
        batch = list(images[i:i + 32])
        if kind == "dinov2":
            x = DINO_P(images=batch, return_tensors="pt").to(device); h = DINO(**x).last_hidden_state
            out_cls.append(h[:, 0].cpu()); out_patch.append(h[:, 1:].mean(1).cpu())
        else:
            x = SIG_P(images=batch, return_tensors="pt").to(device); h = SIG.vision_model(**x)
            out_cls.append(h.pooler_output.cpu()); out_patch.append(h.last_hidden_state.mean(1).cpu())
    return torch.cat(out_cls).numpy(), torch.cat(out_patch).numpy()

def dataset(n, seed):
    env = TableTop(cameras=("front",), image_size=(224, 224), seed=seed, n_cubes=3, n_trays=1)
    imgs, pos, which = [], [], []
    rng = np.random.default_rng(seed)
    for i in range(n):
        env.reset(seed=seed * 10 ** 5 + i)
        imgs.append(env.render("front")); pos.append(env.cube_pos("red")[:2])
        d = {c: np.linalg.norm(env.cube_pos(c)[:2] - env.tray_pos("yellow")[:2]) for c in env.cube_colors}
        which.append(env.cube_colors.index(min(d, key=d.get)))
    return np.array(imgs), np.array(pos), np.array(which)

# %%
DINO_P = AutoImageProcessor.from_pretrained("facebook/dinov2-small"); DINO = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
SIG_P = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224"); SIG = SiglipModel.from_pretrained("google/siglip-base-patch16-224").to(device).eval()
t0 = time.time()
tr_img, tr_pos, tr_w = dataset(800, 0); te_img, te_pos, te_w = dataset(200, 1)
print(f"rendered 1000 scenes in {time.time() - t0:.0f}s")

# %%
rows = []
for kind in ["dinov2", "siglip"]:
    (tr_c, tr_p), (te_c, te_p) = features(tr_img, kind), features(te_img, kind)
    for name, Xtr, Xte in [("global/CLS token", tr_c, te_c), ("mean of patch tokens", tr_p, te_p)]:
        reg = RidgeCV(alphas=np.logspace(-2, 4, 12)).fit(Xtr, tr_pos)
        err = np.linalg.norm(reg.predict(Xte) - te_pos, axis=1).mean() * 100
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, tr_w)
        rows.append((kind, name, err, clf.score(Xte, te_w)))
print(f"{'encoder':8s} {'feature':22s} {'red-cube xy error [cm]':>24s} {'which cube is nearest the tray':>32s}")
for r in rows:
    print(f"{r[0]:8s} {r[1]:22s} {r[2]:24.2f} {r[3]:32.2f}")
print("(chance for the 3-way relation probe: 0.33; predicting the mean position gives", f"{np.linalg.norm(te_pos - tr_pos.mean(0), axis=1).mean() * 100:.1f} cm)")

# %%
with torch.no_grad():
    texts = ["a red block", "a green block", "a blue block", "a yellow tray"]
    x = SIG_P(text=texts, images=list(te_img[:1]), return_tensors="pt", padding="max_length").to(device)
    out = SIG(**x)
print("SigLIP image-text sigmoid scores (one scene):", dict(zip(texts, torch.sigmoid(out.logits_per_image[0]).cpu().numpy().round(3))))
print("DINOv2 has no text tower: great spatial features, no language alignment. SigLIP: language-aligned. OpenVLA concatenates both.")
