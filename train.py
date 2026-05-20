import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.backbone import DetectionBranch, NonMaxSuppression, R2U_Net
from models.matching import OptimalMatching
from utils.dataset import CrowdAI
from utils.loss import compute_l_angle_loss, cross_entropy_loss, iou_loss_function
from utils.utils import (
    permutations_to_polygons,
    scores_to_permutations,
    soft_winding_number,
    sort_sync_nsm_points,
)


def collate_fn(batch):
    return tuple(zip(*batch))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", default="data/train/images/")
    p.add_argument("--annotations", default="data/train/annotation.json")
    p.add_argument("--weights-dir", default="trained_weights")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--window-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lambda-winding", type=float, default=1000.0)
    p.add_argument("--seg-loss-weight", type=float, default=10.0)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_models(weights_dir: Path, device: str):
    backbone = R2U_Net().to(device).train()
    backbone.load_state_dict(torch.load(weights_dir / "polyworld_backbone", map_location=device))

    head_ver = DetectionBranch().to(device).train()
    head_ver.load_state_dict(torch.load(weights_dir / "polyworld_seg_head", map_location=device))

    suppression = NonMaxSuppression().to(device)
    matching = OptimalMatching().to(device).train()

    # Freeze pre-trained components; only train the matching network.
    for param in backbone.parameters():
        param.requires_grad = False
    for param in head_ver.parameters():
        param.requires_grad = False

    return backbone, head_ver, suppression, matching


def train_one_epoch(
    epoch: int,
    dataloader: DataLoader,
    backbone,
    head_ver,
    suppression,
    matching,
    optimizer,
    device: str,
    window_size: int,
    lambda_winding: float,
    seg_loss_weight: float,
):
    last_losses = {}
    for batch in tqdm(dataloader, desc=f"epoch {epoch}"):
        image, gt_vertex_mask, gt_seg_mask, gt_perm, gt_polys, gt_graph, gt_vertices, gt_dic = batch

        image = torch.stack(image).float().permute(0, 3, 1, 2).to(device)
        gt_seg_mask = torch.stack(gt_seg_mask).permute(0, 2, 1).to(device)
        gt_perm = torch.stack(gt_perm).to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            features = backbone(image)
            vertex_logits = head_ver(features)
            _, graph = suppression(vertex_logits)

        graph = sort_sync_nsm_points(graph, gt_vertices, gt_dic).to(device)

        polys, perm_mat, _, sinkhorn_scores, graph = matching.predict(image, features, graph)

        try:
            pred_mask = soft_winding_number(
                polys, lam=lambda_winding, img_size=window_size, device=device
            )
        except Exception as e:
            print(f"soft_winding_number failed: {e}; skipping batch")
            continue

        segmentation_loss = iou_loss_function(pred_mask, gt_seg_mask)
        matching_loss = cross_entropy_loss(sinkhorn_scores, gt_perm)
        pred_angle, gt_angle = compute_l_angle_loss(
            gt_perm.double(), gt_vertices, perm_mat, graph, device=device
        )
        angle_loss = torch.mean(1 - torch.exp(-10 * torch.abs(pred_angle - gt_angle)))

        loss = matching_loss + segmentation_loss * seg_loss_weight + angle_loss

        loss.backward()
        optimizer.step()

        last_losses = {
            "matching": matching_loss.item(),
            "segmentation": segmentation_loss.item(),
            "angle": angle_loss.item(),
            "total": loss.item(),
        }

    return last_losses


def main():
    args = parse_args()
    device = args.device

    weights_dir = Path(args.weights_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = CrowdAI(
        images_directory=args.images_dir,
        annotations_path=args.annotations,
        window_size=args.window_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    backbone, head_ver, suppression, matching = build_models(weights_dir, device)

    optimizer = torch.optim.Adam(matching.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        losses = train_one_epoch(
            epoch,
            dataloader,
            backbone,
            head_ver,
            suppression,
            matching,
            optimizer,
            device,
            args.window_size,
            args.lambda_winding,
            args.seg_loss_weight,
        )

        if losses:
            print(
                f"epoch {epoch} | matching={losses['matching']:.4f} "
                f"seg={losses['segmentation']:.4f} angle={losses['angle']:.4f} "
                f"total={losses['total']:.4f}"
            )

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = checkpoint_dir / f"matching_epoch{epoch:04d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "matching_state_dict": matching.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "losses": losses,
                },
                ckpt_path,
            )


if __name__ == "__main__":
    main()
