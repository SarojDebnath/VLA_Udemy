import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def _in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None and "IPKernelApp" in get_ipython().config
    except Exception:
        return False


def _slug(text):
    return "".join(ch if ch.isalnum() else "_" for ch in text.lower()).strip("_")[:60] or "video"


def show_video(frames, fps=30, title="video", loop=1, inline=True, save_dir="videos", width=640):
    """Show RGB frames.
    In Jupyter: saves videos/<title>.mp4 (H.264) and embeds a player in the cell output, so the video is stored
    inside the .ipynb (⋮ menu / right-click -> download). Outside Jupyter (or in addition, when not headless):
    plays in an OpenCV window (press q to stop). The OpenCV window is skipped when VLA_HEADLESS=1."""
    if len(frames) == 0:
        return
    if inline and _in_notebook():
        try:
            from IPython.display import Video, display
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, _slug(title) + ".mp4")
            save_video(frames, path, fps=fps)
            display(Video(path, embed=True, width=min(width, np.asarray(frames[0]).shape[1] * 2),
                          html_attributes="controls loop autoplay muted"))
        except Exception as e:                       # never let a video break a lesson
            print(f"(inline video skipped: {e})")
    if os.environ.get("VLA_HEADLESS") == "1":
        return
    delay = max(1, int(1000 / fps))
    for _ in range(loop):
        for f in frames:
            cv2.imshow(title, cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2BGR))
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                cv2.destroyWindow(title)
                return
    cv2.waitKey(500)
    cv2.destroyWindow(title)


def save_video(frames, path, fps=30):
    """H.264 + yuv420p MP4 (plays in browsers, VS Code, PowerPoint, LinkedIn). Frames are cropped to even sizes."""
    import imageio
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fr = [np.asarray(f)[: np.asarray(f).shape[0] // 2 * 2, : np.asarray(f).shape[1] // 2 * 2] for f in frames]
    if fr and fr[0].ndim == 2:
        fr = [np.repeat(f[..., None], 3, -1) for f in fr]
    imageio.mimsave(path, [f.astype(np.uint8) for f in fr], fps=fps, codec="libx264", quality=7, macro_block_size=2,
                    pixelformat="yuv420p", ffmpeg_params=["-movflags", "+faststart"])
    return path


def show_images(images, titles=None, cols=None, size=3.0):
    n = len(images)
    cols = cols or n
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            img = images[i]
            ax.imshow(img, cmap=None if np.ndim(img) == 3 else "viridis")
            if titles:
                ax.set_title(titles[i], fontsize=9)
    plt.tight_layout()
    plt.show()
