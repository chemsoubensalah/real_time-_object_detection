"""Tiled inference (the idea behind SAHI): help the model see SMALL objects without retraining.

A small drone covers only a few pixels of the model input. If we cut the frame into overlapping tiles and
run the model on each tile (every tile is enlarged to the model size), the drone looks bigger to the model.
We also run the full frame (for big objects), then merge all boxes with NMS.

Cost: (grid*grid + 1) model runs per frame instead of one, so it is slower.

    from tiled_inference import detect_tiled
    det = detect_tiled(model, frame_bgr, size=512, grid=2)   # model: object with .predict(bgr, size) -> Nx6 tensor
"""
import torch
from torchvision.ops import batched_nms


def tile_origins(length, grid, overlap):
    """Start positions and tile length along one axis, tiles overlapping by `overlap` of their length."""
    if grid <= 1:
        return [0], length
    tile = int(round(length / (grid - (grid - 1) * overlap)))
    stride = (length - tile) / (grid - 1)
    return [int(round(i * stride)) for i in range(grid)], tile


def detect_tiled(model, frame, size, grid=2, overlap=0.2, iou=0.5, include_full=True):
    h, w = frame.shape[:2]
    xs, tw = tile_origins(w, grid, overlap)
    ys, th = tile_origins(h, grid, overlap)
    parts = [model.predict(frame, size)] if include_full else []
    for y0 in ys:
        for x0 in xs:
            det = model.predict(frame[y0:y0 + th, x0:x0 + tw], size).clone()
            det[:, [0, 2]] += x0  # back to full-frame coordinates
            det[:, [1, 3]] += y0
            parts.append(det)
    det = torch.cat(parts) if parts else torch.zeros((0, 6))
    if len(det) == 0:
        return det
    keep = batched_nms(det[:, :4], det[:, 4], det[:, 5].long(), iou)
    return det[keep]
