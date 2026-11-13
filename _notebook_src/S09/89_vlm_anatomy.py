# %% [markdown]
# # 89 · Anatomy of a small VLM (SmolVLM-256M): vision encoder → projector → language model, run on TableTop

# %%
import time, numpy as np, torch, matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from vlacourse import TableTop

device = "cuda" if torch.cuda.is_available() else "cpu"
MID = "HuggingFaceTB/SmolVLM-256M-Instruct"
proc = AutoProcessor.from_pretrained(MID)
vlm = AutoModelForImageTextToText.from_pretrained(MID, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()

def ask(img, question, max_new_tokens=60):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inp = proc(text=prompt, images=[Image.fromarray(img)], return_tensors="pt").to(device)
    with torch.no_grad():
        out = vlm.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip(), inp

# %%
def count(m):
    return sum(p.numel() for p in m.parameters()) / 1e6
parts = {name: count(mod) for name, mod in vlm.model.named_children()}
print("top-level modules (M params):", {k: round(v, 1) for k, v in parts.items()}, "| lm_head", round(count(vlm.lm_head), 1))
print(f"total {count(vlm):.0f}M parameters")

# %%
env = TableTop(cameras=("front",), image_size=(384, 384), seed=0)
img = env.reset(seed=11)["images"]["front"]
plt.figure(figsize=(4, 4)); plt.imshow(img); plt.axis("off"); plt.show()
t0 = time.time()
ans, inp = ask(img, "Describe the objects on the table.")
n_img_tokens = int((inp["input_ids"] == vlm.config.image_token_id).sum())
print(f"prompt tokens {inp['input_ids'].shape[1]}, of which image tokens {n_img_tokens} | {time.time() - t0:.1f}s")
print("VLM:", ans)

# %%
for q in ["How many cubes are on the table?", "What color is the tray?", "Which cube is closest to the robot gripper?", "Is the red cube inside the yellow tray?"]:
    print(f"Q: {q}\nA: {ask(img, q, 30)[0]}\n")
print("Small VLMs describe scenes well and stumble on precise spatial questions: the gap a VLA's action head must close.")
