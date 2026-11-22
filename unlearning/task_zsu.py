import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms


class ZeroShotRetainGenerator(nn.Module):
    """Lightweight 224x224 generator adapted from GKT (Chundawat et al.) for retain anchoring."""
    def __init__(self, z_dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(z_dim, 256 * 7 * 7)
        self.net = nn.Sequential(
            nn.BatchNorm2d(256),
            # 7 -> 14 -> 28 -> 56 -> 112 -> 224
            self._block(256, 256),
            self._block(256, 128),
            self._block(128, 64),
            self._block(64, 32),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh(),  # matches [-1, 1] range of preprocess()
        )

    @staticmethod
    def _block(in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 256, 7, 7)
        return self.net(h)


def kl_at_temp(logits_p: torch.Tensor, logits_q: torch.Tensor, temp: float = 2.0) -> torch.Tensor:
    """KL(softmax(logits_p / T) || softmax(logits_q / T)), matching server evaluation."""
    p = F.softmax(logits_p / temp, dim=-1)
    log_p = F.log_softmax(logits_p / temp, dim=-1)
    log_q = F.log_softmax(logits_q / temp, dim=-1)
    return (p * (log_p - log_q)).sum(dim=-1).mean()


model = unlearn(
    model,
    forget_images,
    epochs=8,
    batch_size=64,
    lr_student=1e-4,
    lr_gen=1e-3,
    npo_beta=0.25,
    lambda_forget=1.0,
    lambda_retain_kl=2.0,
    lambda_weight_l2=0.1,
    unfreeze_last_n_blocks=6,
) -> torch.nn.Module:
    device = next(model.parameters()).device

    # 1. Freeze original model as teacher
    teacher = copy.deepcopy(model).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # 2. Compute pseudo-labels and teacher log-probs on exact forget images
    pseudo_labels = []
    with torch.no_grad():
        for i in range(0, len(forget_images), 128):
            x = preprocess(forget_images[i : i + 128])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits = teacher(x)
            pseudo_labels.append(logits.argmax(dim=1).cpu())
    pseudo_labels = torch.cat(pseudo_labels)

    # 3. Optionally freeze early ViT blocks to protect general features & utility gate
    student = model.to(device)
    for p in student.parameters():
        p.requires_grad_(False)
    for blk in student.blocks[-unfreeze_last_n_blocks:]:
        for p in blk.parameters():
            p.requires_grad_(True)
    for p in student.norm.parameters():
        p.requires_grad_(True)
    for p in student.head.parameters():
        p.requires_grad_(True)

    # Snapshot trainable weights for L2 anchor
    orig_params = {
        name: p.detach().clone()
        for name, p in student.named_parameters()
        if p.requires_grad
    }

    # 4. Zero-shot retain generator (adapted GKT)
    z_dim = 128
    generator = ZeroShotRetainGenerator(z_dim=z_dim).to(device)

    opt_s = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=lr_student,
        weight_decay=0.0,
    )
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr_gen)

    # Augmentations to generalize unlearning to hidden altered copies
    aug = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    ])

    loader = DataLoader(
        TensorDataset(forget_images, pseudo_labels),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    for epoch in range(epochs):
        student.train()
        generator.train()
        running_npo, running_rkl = 0.0, 0.0

        for imgs_u8, y_hat in loader:
            y_hat = y_hat.to(device)

            # Preprocess to [-1, 1], apply augmentations in [0, 1] then re-normalize
            x_clean = imgs_u8.to(device).permute(0, 3, 1, 2).float() / 255.0
            x_aug = (aug(x_clean) - 0.5) / 0.5
            x_exact = (x_clean - 0.5) / 0.5

            # Mix exact and altered views 50/50 (matches equal weighting in scoring)
            x_f = torch.cat([x_exact[: batch_size // 2], x_aug[batch_size // 2 :]], dim=0)

            # --- Step A: Generator maximizes T=2 KL(teacher || student) ---
            z = torch.randn(batch_size, z_dim, device=device)
            opt_g.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                x_syn = generator(z)
                with torch.no_grad():
                    t_syn = teacher(x_syn)
                s_syn = student(x_syn)
                loss_g = -kl_at_temp(t_syn, s_syn, temp=2.0)
            loss_g.backward()
            opt_g.step()

            # --- Step B: Student forgets via NPO + anchors retain via GKT KL + L2 ---
            opt_s.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                with torch.no_grad():
                    t_f = teacher(x_f)
                    t_logp_y = F.log_softmax(t_f, dim=-1).gather(1, y_hat.unsqueeze(1)).squeeze(1)

                s_f = student(x_f)
                s_logp_y = F.log_softmax(s_f, dim=-1).gather(1, y_hat.unsqueeze(1)).squeeze(1)

                # Bounded NPO loss (pushes down memorized confidence smoothly)
                log_ratio = s_logp_y - t_logp_y
                loss_npo = -(2.0 / npo_beta) * F.logsigmoid(-npo_beta * log_ratio).mean()

                # Zero-shot retain KL on fresh generator batch
                with torch.no_grad():
                    x_syn_det = generator(torch.randn(batch_size, z_dim, device=device)).detach()
                    t_syn_det = teacher(x_syn_det)
                s_syn_det = student(x_syn_det)
                loss_retain_kl = kl_at_temp(t_syn_det, s_syn_det, temp=2.0)

                # Weight-space L2 anchor
                loss_l2 = sum(
                    (p - orig_params[name]).pow(2).sum()
                    for name, p in student.named_parameters()
                    if p.requires_grad
                ) / len(orig_params)

                loss_s = (
                    lambda_forget * loss_npo
                    + lambda_retain_kl * loss_retain_kl
                    + lambda_weight_l2 * loss_l2
                )

            loss_s.backward()
            torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
            opt_s.step()

            running_npo += loss_npo.item()
            running_rkl += loss_retain_kl.item()

        print(
            f"[Epoch {epoch + 1}/{epochs}] "
            f"NPO={running_npo / len(loader):.4f} | RetainKL(T=2)={running_rkl / len(loader):.5f}"
        )

    # Restore requires_grad and eval mode cleanly before saving state_dict
    for p in student.parameters():
        p.requires_grad_(True)
    return student.eval()
if __name__ == "__main__":
    import task_template as tt
    preprocess = tt.preprocess  # expose preprocess() used inside unlearn()

    print("Loading model and forget images...")
    model = tt.load_model()
    forget_images = tt.load_forget_images()
    print(f"Loaded {len(forget_images)} forget images on {tt.DEVICE}.")

    model = unlearn(model, forget_images)

    out_path = "submission_zsu.pt"
    state_dict = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}
    torch.save(state_dict, out_path)
    print(f"Saved unlearned model to {out_path}.")
