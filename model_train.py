
import os
import math
import random
from typing import List, Tuple, Dict
import glob
import numpy as np
from PIL import Image
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision import transforms
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from transformers import AutoModel


# ================== CFG ==================
class CFG:
    # ----- dataset root -----
    ROOT = r"FF++"
    FAKE_METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "FaceShifter", "NeuralTextures")
    IMG_EXT = "*.jpg"

    # ----- random video-level split -----
    # The split is regenerated from video IDs at every run using SEED.
    # All frames from the same source video and all forged videos generated
    # from that source are kept in the same subset to avoid data leakage.
    N_TRAIN_REAL_VID: int = 700
    N_VAL_REAL_VID: int = 100
    N_TEST_REAL_VID: int = 200

    # ----- train -----
    BATCH_SIZE: int = 32
    EPOCHS: int = 20
    LR: float = 1e-4
    NUM_WORKERS: int = 6
    SEED: int = 42

    # ----- image -----
    IMAGE_SIZE: int = 224
    PATCH_SIZE: int = 16
    POOL: str = "mean"

    # ----- model -----
    VIT_NAME: str = "google/vit-base-patch16-224-in21k"
    SAVE_PATH: str = "checkpoint.pth"

    # ----- injection -----
    HP_BIAS_LAYERS = (0, 3, 6)   # ViT 第1/4/7个block
    FUSION_TARGET_GRID: int = 14 # 对齐到 ViT patch 网格 14x14

    # ----- SAM -----
    SAM_DIM: int = 256
    DROP_RATE_SAM: float = 0.0

    # ----- tqdm print -----
    TQDM_UPDATE_EVERY: int = 50
    PRINT_CM: bool = True


# ================== Utils ==================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def list_frames(vid_dir: str):
    return sorted(glob.glob(os.path.join(vid_dir, CFG.IMG_EXT)))


def load_image(path: str, tfm):
    img = Image.open(path).convert("RGB")
    return tfm(img)


# ================== Transform ==================
train_transform = transforms.Compose([
    transforms.Resize((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ================== Random video-level split ==================
def split_real_videos_by_original(root: str, seed: int) -> Dict[str, List[str]]:
    original_root = os.path.join(root, "original")
    if not os.path.isdir(original_root):
        raise FileNotFoundError(f"找不到 original 目录: {original_root}")

    real_vids = sorted([
        d for d in os.listdir(original_root)
        if os.path.isdir(os.path.join(original_root, d))
    ])
    real_vids = [vid for vid in real_vids if len(list_frames(os.path.join(original_root, vid))) > 0]

    need = CFG.N_TRAIN_REAL_VID + CFG.N_VAL_REAL_VID + CFG.N_TEST_REAL_VID
    if len(real_vids) < need:
        raise RuntimeError(f"original 视频数不足：{len(real_vids)} < {need}")

    rng = random.Random(seed)
    rng.shuffle(real_vids)

    train_vids = real_vids[:CFG.N_TRAIN_REAL_VID]
    val_vids = real_vids[CFG.N_TRAIN_REAL_VID: CFG.N_TRAIN_REAL_VID + CFG.N_VAL_REAL_VID]
    test_vids = real_vids[
        CFG.N_TRAIN_REAL_VID + CFG.N_VAL_REAL_VID:
        CFG.N_TRAIN_REAL_VID + CFG.N_VAL_REAL_VID + CFG.N_TEST_REAL_VID
    ]

    print(
        f"[split-real] original total={len(real_vids)} | "
        f"train={len(train_vids)}, val={len(val_vids)}, test={len(test_vids)}"
    )
    return {"train": train_vids, "val": val_vids, "test": test_vids}


def build_split_image_list_by_real(root: str, split_real_vids, split_name: str) -> List[Tuple[str, int]]:
    split_real_vids = set(split_real_vids)
    original_root = os.path.join(root, "original")
    img_label_list: List[Tuple[str, int]] = []

    real_cnt = 0
    for vid in sorted(split_real_vids):
        frame_dir = os.path.join(original_root, vid)
        if not os.path.isdir(frame_dir):
            continue
        frames = list_frames(frame_dir)
        for fp in frames:
            img_label_list.append((fp, 0))
        real_cnt += len(frames)

    fake_cnt = 0
    for method in CFG.FAKE_METHODS:
        method_dir = os.path.join(root, method)
        if not os.path.isdir(method_dir):
            continue

        for fake_vid in sorted(os.listdir(method_dir)):
            fake_vid_dir = os.path.join(method_dir, fake_vid)
            if not os.path.isdir(fake_vid_dir):
                continue

            real_prefix = fake_vid[:3]
            if real_prefix not in split_real_vids:
                continue

            frames = list_frames(fake_vid_dir)
            if not frames:
                continue

            for fp in frames:
                img_label_list.append((fp, 1))
            fake_cnt += len(frames)

    print(f"[build_{split_name}] total={len(img_label_list)} | #real={real_cnt}, #fake={fake_cnt}")
    return img_label_list



def make_random_video_split(root: str, seed: int):
    """
    Randomly split the dataset at the source-video level.

    For each source real video, its real frames and all forged videos generated
    from that source are assigned to the same subset. This prevents identity-
    and video-level leakage across training, validation, and test sets.
    """
    split_real = split_real_videos_by_original(root, seed)

    train_list = build_split_image_list_by_real(root, split_real["train"], "train")
    val_list = build_split_image_list_by_real(root, split_real["val"], "val")
    test_list = build_split_image_list_by_real(root, split_real["test"], "test")

    train_ids = set(split_real["train"])
    val_ids = set(split_real["val"])
    test_ids = set(split_real["test"])

    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("视频级划分出现重叠，请检查划分逻辑。")

    print(f"[random-split] seed={seed}")
    print("[random-split] 已按源视频进行随机划分；同源真实与伪造视频保持在同一子集。")
    print("[random-split] train/val/test 之间不存在源视频重叠。")

    return train_list, val_list, test_list, split_real


# ================== Dataset ==================
class FFPPImageDataset(Dataset):
    def __init__(self, img_label_list: List[Tuple[str, int]], transform=None):
        self.data = img_label_list
        self.transform = transform if transform is not None else val_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        img = load_image(path, self.transform)
        return img, torch.tensor(label, dtype=torch.long)


def build_class_balanced_sampler(img_label_list: List[Tuple[str, int]]) -> WeightedRandomSampler:
    labels = [y for _, y in img_label_list]
    cnt_real = sum(1 for y in labels if y == 0)
    cnt_fake = sum(1 for y in labels if y == 1)
    if cnt_real == 0 or cnt_fake == 0:
        raise RuntimeError("训练集某一类为 0，请检查训练集构造结果。")

    w_real = 1.0 / cnt_real
    w_fake = 1.0 / cnt_fake
    weights = [w_fake if y == 1 else w_real for y in labels]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ================== Metrics ==================
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(np.int64)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc = (y_pred == y_true).mean() if len(y_true) else 0.0

    return {
        "acc": float(acc),
        "auc": float(auc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "cm": cm,
    }



class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, dilation,
            groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.conv1(x))


class SpatialCNN(nn.Module):
    def __init__(self, in_channels, embed_dim=768, level=1, stride=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.level = level
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels * 2),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(in_channels * 4),
            nn.GELU(),
            nn.Conv2d(in_channels * 4, in_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels * 2),
            nn.GELU(),
        )
        size = 8 // (2 ** level)
        self.fc = nn.Conv2d(in_channels * 2, embed_dim, kernel_size=size, stride=size, padding=0)

    def forward(self, x):
        b = x.shape[0]
        x = self.block(x)
        x_patch = self.fc(x).view(b, self.embed_dim, -1).permute(0, 2, 1)  # [B, N, C]
        return x, x_patch


class TMF(nn.Module):
    def __init__(self, dim: int, target_grid: int = 14, init_stage: int = None):
        super().__init__()
        self.dim = dim
        self.target_grid = target_grid


        self.weight_mlp = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, 3)
        )


        nn.init.zeros_(self.weight_mlp[-1].weight)
        nn.init.zeros_(self.weight_mlp[-1].bias)

        self.norm = nn.LayerNorm(dim)

    def _align_one(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        g = int(n ** 0.5)
        if g * g != n:
            raise RuntimeError(f"TMF expects square token grid, got N={n}")
        x2d = x.transpose(1, 2).reshape(b, c, g, g)
        x2d = F.interpolate(
            x2d,
            size=(self.target_grid, self.target_grid),
            mode="bilinear",
            align_corners=False,
        )
        return x2d.flatten(2).transpose(1, 2)  # [B, N, C]

    def forward(self, stage_tokens: List[torch.Tensor]) -> torch.Tensor:
        # 对齐三个尺度
        aligned = [self._align_one(x) for x in stage_tokens]
        x1, x2, x3 = aligned  # [B, N, C]


        concat = torch.cat([x1, x2, x3], dim=-1)  # [B, N, 3C]


        weights = self.weight_mlp(concat)  # [B, N, 3]
        weights = torch.softmax(weights, dim=-1)


        fused = (
            weights[..., 0:1] * x1 +
            weights[..., 1:2] * x2 +
            weights[..., 2:3] * x3
        )

        return self.norm(fused)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        b, n, c = x.shape
        _, m, _ = context.shape
        q = self.q(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(context).reshape(b, m, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(context).reshape(b, m, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class Injector(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=12,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        attn_drop=0.0,
        proj_drop=0.0,
        init_values=0.0,
    ):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.norm = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=False, attn_drop=attn_drop, proj_drop=proj_drop)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, spa_prior):
        attn_spa_prior = self.attn(self.query_norm(query), self.feat_norm(spa_prior))
        return query + self.gamma * self.norm(attn_spa_prior)



class SAM(nn.Module):
    
    def __init__(self, dim, num_heads=12, SAM_dim=256, drop_rate_SAM=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = SAM_dim

        self.adaptor_a = nn.Linear(dim, SAM_dim, bias=False)
        nn.init.kaiming_uniform_(self.adaptor_a.weight, a=math.sqrt(5))

        self.adaptor_b = nn.Linear(SAM_dim, dim * 3, bias=False)
        nn.init.zeros_(self.adaptor_b.weight)

        self.adaptor_conv = nn.Conv2d(SAM_dim, SAM_dim, 3, 1, 1)
        self.adaptor_drop = nn.Dropout(p=drop_rate_SAM)

        self.afi_q = nn.Linear(dim, dim, bias=False)
        self.afi_k = nn.Linear(dim, dim, bias=False)
        self.afi_v = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.afi_q.weight)
        nn.init.zeros_(self.afi_k.weight)
        nn.init.zeros_(self.afi_v.weight)

        self.gate = nn.Parameter(torch.zeros(1))

    def high_pass_fft(self, x_2d, radius_ratio=0.25):
        b, c, h, w = x_2d.shape
        xf = torch.fft.fft2(x_2d, dim=(-2, -1))
        xf = torch.fft.fftshift(xf, dim=(-2, -1))

        yy = torch.arange(h, device=x_2d.device) - h // 2
        xx = torch.arange(w, device=x_2d.device) - w // 2
        y, x = torch.meshgrid(yy, xx, indexing="ij")
        dist = torch.sqrt(x ** 2 + y ** 2)

        r = radius_ratio * min(h, w)
        mask = (dist >= r).float().unsqueeze(0).unsqueeze(0)

        xf = xf * mask
        xf = torch.fft.ifftshift(xf, dim=(-2, -1))
        return torch.fft.ifft2(xf, dim=(-2, -1)).real

    def forward(self, x):
        b, n, c = x.shape

        # ---- spatial SAM branch ----
        qkv_delta_sam = self.adaptor_a(self.adaptor_drop(x))
        g = int((n - 1) ** 0.5)
        if (g * g) != (n - 1):
            raise RuntimeError(f"SAM expects square patch grid, got N={n}")

        x_patch = qkv_delta_sam[:, 1:].reshape(b, g, g, self.dim).permute(0, 3, 1, 2)
        x_patch = self.adaptor_conv(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(b, g * g, self.dim)

        x_cls = qkv_delta_sam[:, :1].reshape(b, 1, 1, self.dim).permute(0, 3, 1, 2)
        x_cls = self.adaptor_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(b, 1, self.dim)

        qkv_delta_sam = torch.cat([x_cls, x_patch], dim=1)
        qkv_delta_sam = qkv_delta_sam.reshape(b, n, self.dim)
        qkv_delta_sam = self.adaptor_b(qkv_delta_sam).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)

        # ---- AFI branch ----
        x_patch_src = x[:, 1:].reshape(b, g, g, c).permute(0, 3, 1, 2)
        x_hp = self.high_pass_fft(x_patch_src)
        x_hp = x_hp.permute(0, 2, 3, 1).reshape(b, g * g, c)
        x_hp = torch.cat([torch.zeros_like(x[:, :1]), x_hp], dim=1)

        dq = self.afi_q(x_hp)
        dk = self.afi_k(x_hp)
        dv = self.afi_v(x_hp)

        qkv_delta_afi = torch.stack([dq, dk, dv], dim=2).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        alpha = torch.tanh(self.gate)
        return qkv_delta_sam + alpha * qkv_delta_afi



def pick_attr(obj, candidates):
    for n in candidates:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


def require_attr(obj, candidates, err):
    x = pick_attr(obj, candidates)
    if x is None:
        raise AttributeError(err + f" | tried: {candidates} | got type={type(obj).__name__}")
    return x


class ModelFull(nn.Module):
    def __init__(self, vit_name: str, img_size: int, patch_size: int, pool: str, num_classes: int = 2, bias_init: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(vit_name)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        self.pool = pool
        dim = self.backbone.config.hidden_size
        self.num_heads = self.backbone.config.num_attention_heads
        self.head_dim = dim // self.num_heads
        self.inject_layers = set(CFG.HP_BIAS_LAYERS)

        self.spa_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            SeparableConv2d(32, 32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        module = []
        for i in range(len(CFG.HP_BIAS_LAYERS)):
            if i == 0:
                module.append(SpatialCNN(32 * (2 ** i), embed_dim=dim, level=i + 1, stride=1))
            else:
                module.append(SpatialCNN(32 * (2 ** i), embed_dim=dim, level=i + 1))
        self.spm_blocks = nn.ModuleList(module)

        self.fusion_blocks = nn.ModuleList([
            TMF(dim=dim, target_grid=CFG.FUSION_TARGET_GRID, init_stage=i)
            for i in range(len(CFG.HP_BIAS_LAYERS))
        ])
        self.inject_blocks = nn.ModuleList([
            Injector(dim, num_heads=self.num_heads)
            for _ in range(len(CFG.HP_BIAS_LAYERS))
        ])

        encoder = require_attr(self.backbone, ["encoder"], "Cannot find backbone.encoder")
        layers = require_attr(encoder, ["layer", "layers"], "Cannot find encoder.layer/layers")
        depth = len(layers)
        self.sams = nn.ModuleDict({
            str(i): SAM(dim=dim, num_heads=self.num_heads, SAM_dim=CFG.SAM_DIM, drop_rate_SAM=CFG.DROP_RATE_SAM)
            for i in range(depth)
        })

        self.classifier = nn.Linear(dim, num_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)

        # ---- precompute all 3 CNN stages once ----
        s = self.spa_stem(x)
        stage_tokens = []
        for spm in self.spm_blocks:
            s, s_patch = spm(s)
            stage_tokens.append(s_patch)

        with torch.no_grad():
            hidden = self.backbone.embeddings(pixel_values=x)

        encoder = self.backbone.encoder
        layers = require_attr(encoder, ["layer", "layers"], "Cannot find encoder layers")
        inject_index = 0

        for i, layer in enumerate(layers):
            if i in self.inject_layers:
                fused_prior = self.fusion_blocks[inject_index](stage_tokens)
                hidden = self.inject_blocks[inject_index](hidden, fused_prior)
                inject_index += 1

            norm1 = require_attr(layer, ["layernorm_before", "norm1"], "Cannot find norm1")
            norm2 = require_attr(layer, ["layernorm_after", "norm2"], "Cannot find norm2")

            attn_wrap = require_attr(layer, ["attention", "attn"], "Cannot find attention wrapper")
            attn_core = require_attr(attn_wrap, ["attention", "self"], "Cannot find attention core")

            q_proj = require_attr(attn_core, ["query", "q"], "Cannot find Q projection")
            k_proj = require_attr(attn_core, ["key", "k"], "Cannot find K projection")
            v_proj = require_attr(attn_core, ["value", "v"], "Cannot find V projection")

            out_mod = require_attr(attn_wrap, ["output"], "Cannot find attention output")
            out_dense = require_attr(out_mod, ["dense"], "Cannot find attention output dense")
            out_drop = pick_attr(out_mod, ["dropout"])

            mlp = require_attr(layer, ["intermediate"], "Cannot find intermediate MLP")
            out_mlp = require_attr(layer, ["output"], "Cannot find output MLP")
            fc1 = require_attr(mlp, ["dense"], "Cannot find fc1")
            fc2 = require_attr(out_mlp, ["dense"], "Cannot find fc2")
            mlp_drop = pick_attr(out_mlp, ["dropout"])
            act = pick_attr(mlp, ["intermediate_act_fn"]) or F.gelu

            residual = hidden
            h = norm1(hidden)

            Q = q_proj(h)
            K = k_proj(h)
            V = v_proj(h)

            s_len = Q.size(1)
            Q = Q.view(b, s_len, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(b, s_len, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(b, s_len, self.num_heads, self.head_dim).transpose(1, 2)

            qkv_delta = self.sams[str(i)](h)
            q_delta, k_delta, v_delta = qkv_delta.unbind(0)
            Q = Q + q_delta
            K = K + k_delta
            V = V + v_delta

            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_probs = torch.softmax(attn_scores, dim=-1)
            context = torch.matmul(attn_probs, V)
            context = context.transpose(1, 2).contiguous().view(b, s_len, self.num_heads * self.head_dim)

            attn_out_x = out_dense(context)
            if out_drop is not None:
                attn_out_x = out_drop(attn_out_x)
            hidden = residual + attn_out_x

            residual = hidden
            h = norm2(hidden)
            x_ffn = fc1(h)
            x_ffn = act(x_ffn) if callable(act) else act(x_ffn)
            if mlp_drop is not None:
                x_ffn = mlp_drop(x_ffn)
            x_ffn = fc2(x_ffn)
            if mlp_drop is not None:
                x_ffn = mlp_drop(x_ffn)
            hidden = residual + x_ffn

        ln = pick_attr(self.backbone, ["layernorm", "ln_f", "norm"])
        if ln is not None:
            hidden = ln(hidden)

        feat = hidden[:, 0, :] if self.pool == "cls" else hidden[:, 1:, :].mean(dim=1)
        return self.classifier(feat)


# ================== train / eval ==================
def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch_idx: int,
    split_name: str,
):
    is_train = split_name.lower().startswith("train")
    model.train() if is_train else model.eval()

    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    all_probs = []
    all_labels = []

    pbar = tqdm(loader, desc=f"{split_name} Epoch {epoch_idx}", dynamic_ncols=True, leave=True)

    for step, (imgs, labels) in enumerate(pbar, start=1):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(imgs)
            loss = ce(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()

        probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        all_probs.extend(list(probs))
        all_labels.extend(list(labels.detach().cpu().numpy()))

        if step % CFG.TQDM_UPDATE_EVERY == 0:
            avg_loss = total_loss / max(1, total_n)
            acc = total_correct / max(1, total_n)

            y_true_run = np.asarray(all_labels, dtype=np.int64)
            y_prob_run = np.asarray(all_probs, dtype=np.float32)
            if len(y_true_run) > 0 and (y_true_run.min() != y_true_run.max()):
                try:
                    auc_run = roc_auc_score(y_true_run, y_prob_run)
                except Exception:
                    auc_run = float("nan")
            else:
                auc_run = float("nan")

            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "acc": f"{acc:.4f}",
                "auc": f"{auc_run:.4f}" if auc_run == auc_run else "nan"
            })

    y_true = np.asarray(all_labels, dtype=np.int64)
    y_prob = np.asarray(all_probs, dtype=np.float32)

    metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    avg_loss = total_loss / max(1, total_n)

    print(
        f"[{split_name}][Epoch {epoch_idx}] "
        f"loss={avg_loss:.4f} acc={metrics['acc']:.4f} auc={metrics['auc']:.4f} "
        f"p={metrics['precision']:.4f} r={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
    )
    if CFG.PRINT_CM and (not is_train):
        print(f"ConfusionMatrix [[TN,FP],[FN,TP]] = {metrics['cm'].tolist()}")

    return avg_loss, metrics


# ================== main ==================
def main():
    set_seed(CFG.SEED)
    device = get_device()
    print("使用设备:", device)

    train_list, val_list, test_list, _ = make_random_video_split(
        CFG.ROOT, CFG.SEED
    )

    train_dataset = FFPPImageDataset(train_list, transform=train_transform)
    train_sampler = build_class_balanced_sampler(train_list)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        FFPPImageDataset(val_list, transform=val_transform),
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        FFPPImageDataset(test_list, transform=val_transform),
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
    )

    model = ModelFull(
        vit_name=CFG.VIT_NAME,
        img_size=CFG.IMAGE_SIZE,
        patch_size=CFG.PATCH_SIZE,
        pool=CFG.POOL,
        num_classes=2,
        bias_init=0.1,
    ).to(device)

    total_p, trainable_p = count_params(model)
    print(f"[params] total={total_p:,} | trainable={trainable_p:,} (trainable: spa_stem+spm_blocks+TMF+injectors+SAM+head)")

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=CFG.LR)

    best_val_auc = -1.0
    best_state = None

    for epoch in range(1, CFG.EPOCHS + 1):
        run_one_epoch(model, train_loader, optimizer, device, epoch, "train")
        _, val_metrics = run_one_epoch(model, val_loader, optimizer, device, epoch, "val")

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, CFG.SAVE_PATH)
            print(f"[save] {CFG.SAVE_PATH} (val_auc={best_val_auc:.4f})")

    print(f"[done] best val_auc={best_val_auc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    print("[test] evaluating.")
    run_one_epoch(model, test_loader, optimizer, device, 0, "test")


if __name__ == "__main__":
    main()
