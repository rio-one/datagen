import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(os.environ.get("DATAGEN_ROOT", Path(__file__).resolve().parents[2]))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def new_stage(work, stage):
    path = Path(work) / stage
    if path.exists():
        raise ValueError(f"Stage already exists: {path}. Keep completed stages; use a NEW work directory to rerun upstream stages.")
    path.mkdir(parents=True)
    return path


def extract(args):
    source = Path(args.video).resolve()
    if not source.is_file():
        raise ValueError(f"Video not found: {source}")
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps <= 0:
        raise ValueError("Cannot decode video or read its frame rate")
    out = new_stage(args.work, "frames")
    records, index, next_time = [], 0, args.start
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = index / fps
            source_index = index
            index += 1
            if args.end is not None and timestamp > args.end:
                break
            if timestamp + 0.5 / fps < next_time:
                continue
            blur = float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            if blur < args.min_sharpness:
                continue
            name = f"{len(records):06d}.jpg"
            if not cv2.imwrite(str(out / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise IOError(f"Could not write {name}")
            records.append(dict(name=name, source_frame=source_index, timestamp_sec=timestamp, sharpness=blur))
            next_time = timestamp + 1.0 / args.fps
            if len(records) >= args.max_frames:
                break
    finally:
        cap.release()
    if len(records) < 3:
        raise ValueError("Fewer than 3 frames extracted; inspect video/interval/sharpness settings")
    write_json(Path(args.work) / "extraction.json", dict(video=str(source), video_fps=fps, timing="nominal CFR index/fps; transcode variable-frame-rate video first", frames=records))
    print(f"Extracted {len(records)} frames: {out}")


def sfm(args):
    import pycolmap
    if args.threads < 1:
        raise ValueError("threads must be positive")
    if hasattr(os, "sched_getaffinity"):
        # Limit only this process; includes COLMAP routines with internal thread pools.
        allowed_cpus = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, allowed_cpus[:args.threads])
    work = Path(args.work).resolve()
    if not (work / "extraction.json").exists():
        raise ValueError("Run extract first")
    out = new_stage(work, "sfm")
    database = out / "database.db"
    sift = pycolmap.SiftExtractionOptions()
    sift.num_threads = args.threads
    sift.max_num_features = args.features
    pycolmap.extract_features(database_path=str(database), image_path=str(work / "frames"), camera_mode=pycolmap.CameraMode.SINGLE, camera_model="SIMPLE_RADIAL", sift_options=sift, device=pycolmap.Device.cpu)
    matching = pycolmap.SiftMatchingOptions()
    matching.num_threads = args.threads
    if args.matching == "exhaustive":
        pycolmap.match_exhaustive(database_path=str(database), sift_options=matching, device=pycolmap.Device.cpu)
    else:
        opts = pycolmap.SequentialMatchingOptions()
        opts.overlap = args.overlap
        opts.loop_detection = False
        pycolmap.match_sequential(database_path=str(database), matching_options=opts, sift_options=matching, device=pycolmap.Device.cpu)
    mapper = pycolmap.IncrementalPipelineOptions()
    mapper.num_threads = args.threads
    models = pycolmap.incremental_mapping(database_path=str(database), image_path=str(work / "frames"), output_path=str(out / "sparse"), options=mapper)
    if not models:
        raise ValueError("COLMAP registered no model. Increase overlap, improve texture/parallax, or use exhaustive matching in a NEW work directory.")
    best_id, model = max(models.items(), key=lambda item: item[1].num_reg_images())
    best_path = out / "sparse" / str(best_id)
    pycolmap.undistort_images(output_path=str(out / "undistorted"), input_path=str(best_path), image_path=str(work / "frames"), output_type="COLMAP")
    write_json(out / "result.json", dict(best_model=int(best_id), registered_images=model.num_reg_images(), pycolmap=pycolmap.__version__, device="cpu"))
    print(f"Registered {model.num_reg_images()} frames. Next: prepare.")


def rotation_between(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    if np.linalg.norm(a) < 1e-9 or np.linalg.norm(b) < 1e-9:
        raise ValueError("Up direction cannot be zero")
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if c < -1 + 1e-8:
        axis = np.cross(a, [1, 0, 0] if abs(a[0]) < 0.9 else [0, 1, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + skew + skew @ skew / (1 + c)


def crop_intrinsics(K, width, height, size):
    edge = min(width, height)
    left, top = (width - edge) // 2, (height - edge) // 2
    ratio = size / edge
    A = np.array([[ratio, 0, -ratio * left], [0, ratio, -ratio * top], [0, 0, 1]])
    return A @ K, (left, top, left + edge, top + edge), A


def cam_matrix(image):
    pose = image.cam_from_world
    pose = pose() if callable(pose) else pose
    return np.asarray(pose.matrix(), dtype=float)


def prepare(args):
    import pycolmap
    work = Path(args.work).resolve()
    base = work / "sfm" / "undistorted"
    model_path = base / "sparse"
    if not (model_path / "cameras.bin").exists() and (model_path / "0" / "cameras.bin").exists():
        model_path /= "0"
    rec = pycolmap.Reconstruction(str(model_path))
    images = sorted(rec.images.values(), key=lambda im: im.name)
    xyz = np.array([p.xyz for p in rec.points3D.values()], dtype=float)
    if len(xyz) < 20:
        raise ValueError("Too few sparse points for a scene normalization")
    up = np.asarray(args.up if args.up else np.mean([-cam_matrix(im)[:3, :3].T[:, 1] for im in images], axis=0))
    Q = rotation_between(up, [0, 0, 1])
    aligned = xyz @ Q.T
    center = np.median(aligned, axis=0)
    radius = float(np.percentile(np.linalg.norm(aligned - center, axis=1), 98) / 0.9)
    if radius < 1e-6:
        raise ValueError("Degenerate scene extent")
    S = np.eye(4)
    S[:3, :3] *= radius
    S[:3, 3] = center
    out = new_stage(work, "prepared")
    (out / "rgb").mkdir()
    source = {r["name"]: r for r in read_json(work / "extraction.json")["frames"]}
    records, matrices, projections = [], {}, []
    for index, im in enumerate(images):
        camera = rec.cameras[im.camera_id]
        image_path = base / "images" / im.name
        image = Image.open(image_path).convert("RGB")
        K, box, A = crop_intrinsics(np.array(camera.calibration_matrix()), image.width, image.height, args.size)
        stem = f"{index:06d}"
        image.crop(box).resize((args.size, args.size), Image.Resampling.BILINEAR).save(out / "rgb" / f"{stem}.jpg", quality=95)
        rt = cam_matrix(im).copy()
        rt[:, :3] = rt[:, :3] @ Q.T
        W = np.eye(4)
        W[:3, :4] = K @ rt
        matrices[f"world_mat_{index}"] = W.astype(np.float32)
        matrices[f"scale_mat_{index}"] = S.astype(np.float32)
        rec_item = dict(index=index, stem=stem, source=source[im.name], K=K.tolist(), world_to_camera_aligned=rt.tolist(), crop=box, pixel_transform=A.tolist())
        records.append(rec_item)
        # Reprojection previews use the observed COLMAP points, not invented targets.
        if index < 6:
            canvas = cv2.cvtColor(np.array(image.crop(box).resize((args.size, args.size))), cv2.COLOR_RGB2BGR)
            errors = []
            for point in im.points2D:
                if not point.has_point3D() or point.point3D_id not in rec.points3D:
                    continue
                original_xyz = rec.points3D[point.point3D_id].xyz
                p = K @ (rt[:, :3] @ (Q @ original_xyz) + rt[:, 3])
                if p[2] <= 0:
                    continue
                prediction = p[:2] / p[2]
                observation = (A @ np.r_[point.xy, 1])[:2]
                errors.append(float(np.linalg.norm(prediction - observation)))
                if np.all(prediction >= 0) and np.all(prediction < args.size):
                    cv2.circle(canvas, tuple(np.rint(prediction).astype(int)), 1, (0, 255, 0), -1)
            cv2.imwrite(str(out / f"reprojection_{stem}.jpg"), canvas)
            projections.append(dict(frame=index, points=len(errors), median_error_px=float(np.median(errors)) if errors else None))
    np.savez(out / "cameras.npz", **matrices)
    rec.export_PLY(str(out / "sparse_original_coordinates.ply"))
    write_json(out / "manifest.json", dict(size=args.size, up_source="explicit" if args.up else "estimated_from_camera_vertical_VERIFY_VISUALLY", original_world_up=up.tolist(), alignment_rotation=Q.tolist(), normalization_scale_mat=S.tolist(), metric_scale_known=False, frames=records, reprojected_previews=projections))
    print(f"Prepared {len(records)} registered frames. Inspect reprojection previews and verify +Z up before export.")


def annotations(path):
    data = read_json(path)
    objects = data["objects"]
    if not objects:
        raise ValueError("At least one foreground object is required")
    ids = [obj["id"] for obj in objects]
    if ids != list(range(1, len(objects) + 1)) or len(objects) > 254:
        raise ValueError("Object IDs must be ordered 1..N, N<=254, background=0")
    if len({obj["name"] for obj in objects}) != len(objects):
        raise ValueError("Object names must be unique")
    return objects


def segment(args):
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    work = Path(args.work).resolve()
    objects = annotations(args.annotations)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise ValueError(f"Missing SAM2 checkpoint: {checkpoint}. See MANUAL_TODO.md")
    manifest = read_json(work / "prepared" / "manifest.json")
    size, count = manifest["size"], len(manifest["frames"])
    training_anchors = set(read_json(work / "split.json")["train"]) if (work / "split.json").exists() else None
    for obj in objects:
        if not obj.get("prompts"):
            raise ValueError(f"Object {obj['id']} needs points or box prompts")
        for prompt in obj["prompts"]:
            if not 0 <= prompt["frame"] < count:
                raise ValueError("Prompt frame outside prepared sequence")
            if training_anchors is not None and prompt["frame"] not in training_anchors:
                raise ValueError("Use training frames for SAM prompts; split.json marks this prompt frame as held out/nontraining")
            if "points" in prompt:
                pts = np.asarray(prompt["points"])
                if pts.ndim != 2 or pts.shape[1] != 2 or np.any(pts < 0) or np.any(pts >= size):
                    raise ValueError("Points must be [x,y] coordinates in PREPARED images")
                if len(prompt.get("labels", [])) != len(pts):
                    raise ValueError("Each point needs a foreground=1/background=0 label")
    out = new_stage(work, "labels")
    predictor = build_sam2_video_predictor(args.sam_config, str(checkpoint), device=args.device, apply_postprocessing=False)
    device_type = torch.device(args.device).type
    from contextlib import nullcontext
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
    with torch.inference_mode(), autocast:
        state = predictor.init_state(video_path=str(work / "prepared" / "rgb"), offload_video_to_cpu=True)
        for obj in objects:
            for prompt in obj["prompts"]:
                predictor.add_new_points_or_box(inference_state=state, frame_idx=prompt["frame"], obj_id=obj["id"], points=np.asarray(prompt["points"], dtype=np.float32) if "points" in prompt else None, labels=np.asarray(prompt["labels"], dtype=np.int32) if "points" in prompt else None, box=np.asarray(prompt["box"], dtype=np.float32) if "box" in prompt else None)
        for frame_idx, object_ids, logits in predictor.propagate_in_video(state, start_frame_idx=0):
            scores = logits[:, 0].float().cpu().numpy()
            best = np.argmax(scores, axis=0)
            label = np.asarray(object_ids, dtype=np.uint8)[best]
            label[np.max(scores, axis=0) <= 0] = 0
            Image.fromarray(label).save(out / f"{frame_idx:06d}.png")
    shutil.copy2(args.annotations, work / "annotations.used.json")
    print("Instance tracking completed. Inspect with: datagen preview --work ...")


def priors(args):
    import sys
    import torch
    import timm
    work = Path(args.work).resolve()
    files = sorted((work / "prepared" / "rgb").glob("*.jpg"))
    if not files:
        raise ValueError("Run prepare first")
    models = Path(args.models)
    for task in ("depth", "normal"):
        if not (models / f"omnidata_dpt_{task}_v2.ckpt").is_file():
            raise ValueError(f"Missing Omnidata {task} checkpoint; see MANUAL_TODO.md")
    out = work / "priors"
    out.mkdir(exist_ok=True)
    sys.path.insert(0, str(Path(args.omnidata) / "omnidata_tools" / "torch"))
    # The full Omnidata checkpoint contains the backbone. Do not fetch unrelated ImageNet weights.
    original_create = timm.create_model
    def offline_create(*a, **kw):
        kw["pretrained"] = False
        return original_create(*a, **kw)
    timm.create_model = offline_create
    from modules.midas.dpt_depth import DPTDepthModel
    for task in ("depth", "normal"):
        missing = [f for f in files if not (out / f"{f.stem}_{task}.npy").exists()]
        if not missing:
            continue
        model = DPTDepthModel(backbone="vitb_rn50_384", **({"num_channels": 3} if task == "normal" else {}))
        # Only tensor weights and a known legacy callback metadata key are permitted.
        from .checkpoint_compat import load_omnidata_checkpoint
        ckpt = load_omnidata_checkpoint(models / f"omnidata_dpt_{task}_v2.ckpt")
        state = ckpt.get("state_dict", ckpt)
        state = {k.removeprefix("model."): v for k, v in state.items()}
        model.load_state_dict(state)
        model.to(args.device).eval()
        with torch.inference_mode():
            for path in missing:
                rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
                if rgb.shape[:2] != (384, 384):
                    raise ValueError("This Omnidata adapter requires prepare --size 384; no hidden second crop is applied")
                tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0).to(args.device)
                if task == "depth":
                    tensor = (tensor - 0.5) / 0.5
                prediction = model(tensor).clamp(0, 1)[0].cpu().numpy().astype(np.float32)
                np.save(out / f"{path.stem}_{task}.npy", prediction)
        del model, ckpt, state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for path in files:
        depth = np.load(out / f"{path.stem}_depth.npy")
        normal = np.load(out / f"{path.stem}_normal.npy")
        if depth.shape != (384, 384) or normal.shape != (3, 384, 384):
            raise ValueError(f"Unexpected prior shape at {path.stem}: {depth.shape}, {normal.shape}")
        mask = np.isfinite(depth) & np.isfinite(normal).all(axis=0) & (np.linalg.norm(normal * 2 - 1, axis=0) > 0.1)
        np.save(out / f"{path.stem}_mask.npy", mask.astype(np.float32))
    write_json(out / "provenance.json", dict(depth="omnidata_dpt_depth_v2", normal="omnidata_dpt_normal_v2", normal_layout="CHW", normal_encoding="[0,1] -> camera normal via 2*x-1; follows ObjectSDF++ adapter", units="relative depth; no metric claim", preprocessing="prepared RGB; no additional crop"))
    print("Depth, normal and validity masks completed")


def evenly(values, n):
    if n < 0 or n > len(values):
        raise ValueError(f"Requested {n} distinct frames from {len(values)} available")
    if n == 0:
        return []
    return [values[i] for i in np.linspace(0, len(values) - 1, n).round().astype(int)]


def choose_split(n, training_count, test_count):
    if training_count < 1 or test_count < 1 or training_count + test_count > n:
        raise ValueError("Need enough distinct registered frames for nonempty train and test splits")
    # Reserve interior test frames, leaving the first frame available as a training prompt anchor.
    test = evenly(list(range(1, n - 1)), test_count)
    train = evenly([i for i in range(n) if i not in test], training_count)
    return dict(train=train, test=test)


def split_frames(args):
    work = Path(args.work)
    destination = work / "split.json"
    if destination.exists():
        raise ValueError(f"Split already exists: {destination}; edit explicitly if needed, before making annotations")
    n = len(read_json(work / "prepared" / "manifest.json")["frames"])
    result = choose_split(n, args.train_views, args.test_views)
    write_json(destination, result)
    print(json.dumps(result))
    print(f"Split saved: {destination}. Place annotation prompts on TRAINING frames only.")


def preview(args):
    work = Path(args.work)
    files = sorted((work / "prepared" / "rgb").glob("*.jpg"))
    if not files:
        raise ValueError("Run prepare first")
    out = work / "preview"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    palette = rng.integers(40, 255, size=(256, 3), dtype=np.uint8)
    palette[0] = 0
    for path in evenly(files, min(args.count, len(files))):
        rgb = np.asarray(Image.open(path).convert("RGB"))
        label_path = work / "labels" / f"{path.stem}.png"
        if label_path.exists():
            label = np.asarray(Image.open(label_path))
            rgb = np.where((label > 0)[..., None], (0.55 * rgb + 0.45 * palette[label]).astype(np.uint8), rgb)
        canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for coordinate in range(0, canvas.shape[0], 64):
            cv2.line(canvas, (0, coordinate), (canvas.shape[1] - 1, coordinate), (180, 180, 180), 1)
            cv2.line(canvas, (coordinate, 0), (coordinate, canvas.shape[0] - 1), (180, 180, 180), 1)
            cv2.putText(canvas, str(coordinate), (2, max(12, coordinate)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            cv2.putText(canvas, str(coordinate), (coordinate, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.imwrite(str(out / f"{path.stem}.jpg"), canvas)
    print(f"Preview images (x horizontal, y vertical, grid every 64 pixels): {out}")


def export(args):
    work, output = Path(args.work).resolve(), Path(args.output).resolve()
    manifest = read_json(work / "prepared" / "manifest.json")
    objects = annotations(args.annotations)
    background = read_json(args.annotations).get("background", {})
    if not background.get("geometry_prompt") or not background.get("texture_prompt"):
        raise ValueError("Annotations need background.geometry_prompt and background.texture_prompt for DP channel 0")
    for obj in objects:
        if not obj.get("geometry_prompt") or not obj.get("texture_prompt") or not isinstance(obj.get("floor_supported"), bool):
            raise ValueError("Each object needs nonempty geometry_prompt/texture_prompt and boolean floor_supported")
    split_path = Path(args.split) if args.split else work / "split.json"
    if split_path.exists():
        split = read_json(split_path)
        train, test = split["train"], split["test"]
    else:
        if args.split:
            raise FileNotFoundError(split_path)
        split = choose_split(len(manifest["frames"]), args.train_views, args.test_views)
        train, test = split["train"], split["test"]
    n = len(manifest["frames"])
    if not train or not test or len(set(train + test)) != len(train + test) or any(not isinstance(i, int) or i < 0 or i >= n for i in train + test):
        raise ValueError("Train/test must be distinct, nonempty valid prepared frame indices")
    if args.render_count not in (0, 100):
        raise ValueError("Current DP-Recon expects exactly 100 render cameras; use 0 for training-only export")
    render = []
    if args.render_count:
        if len(test) > 100 or n < 100:
            raise ValueError("Full DP-Recon export requires >=100 registered frames and <=100 test frames; use --render-count 0 for training-only export")
        render = sorted(test + evenly([i for i in range(n) if i not in test], 100 - len(test)))
        Ks = np.array([manifest["frames"][i]["K"] for i in render + train])
        if not np.allclose(Ks, Ks[0], atol=1e-3):
            raise ValueError("Current DP-Recon auxiliary renderer assumes one common intrinsic matrix")
    # Check only files actually needed by the requested export; never scan model caches.
    source_cameras = np.load(work / "prepared" / "cameras.npz")
    checked = set(train + test)
    visible = set()
    for i in checked:
        stem = f"{i:06d}"
        label = np.asarray(Image.open(work / "labels" / f"{stem}.png"))
        if label.shape != (manifest["size"], manifest["size"]) or np.any(label > len(objects)):
            raise ValueError(f"Invalid instance map: {stem}")
        if i in train:
            visible.update(np.unique(label).tolist())
            for kind, shape in (("depth", label.shape), ("normal", (3,) + label.shape), ("mask", label.shape)):
                prior = np.load(work / "priors" / f"{stem}_{kind}.npy")
                if prior.shape != shape or not np.isfinite(prior).all():
                    raise ValueError(f"Invalid {kind} array: {stem}")
        for key in ("world_mat", "scale_mat"):
            if source_cameras[f"{key}_{i}"].shape != (4, 4):
                raise ValueError("Invalid camera matrix shape")
    if not set(range(1, len(objects) + 1)).issubset(visible):
        raise ValueError("Some target objects never appear in the selected TRAINING views; revise split")
    if output.exists():
        raise ValueError(f"Output already exists; choose a new output path: {output}")
    output.mkdir(parents=True)
    (output / "instance_mask").mkdir()
    (output / "segs").mkdir()

    def camera_subset(indices, destination):
        arrays = {f"{kind}_{j}": source_cameras[f"{kind}_{i}"] for j, i in enumerate(indices) for kind in ("world_mat", "scale_mat")}
        np.savez(destination, **arrays)

    for j, i in enumerate(train):
        stem, source_stem = f"{j:06d}", f"{i:06d}"
        Image.open(work / "prepared" / "rgb" / f"{source_stem}.jpg").save(output / f"{stem}_rgb.png")
        for kind in ("depth", "normal", "mask"):
            shutil.copy2(work / "priors" / f"{source_stem}_{kind}.npy", output / f"{stem}_{kind}.npy")
        shutil.copy2(work / "labels" / f"{source_stem}.png", output / "instance_mask" / f"{stem}.png")
        shutil.copy2(work / "labels" / f"{source_stem}.png", output / "segs" / f"{stem}_segs.png")
    camera_subset(train, output / "cameras.npz")
    write_json(output / "instance_id.json", {obj["name"]: obj["id"] for obj in objects})
    (output / "instance_mapping.txt").write_text("\n".join(f"{i},0,{i}" for i in range(len(objects) + 1)) + "\n")
    prefix = f"{args.data_dir}_scan{args.scan_id}"
    write_json(output / f"{prefix}_prompt_geometry.json", {"0": background["geometry_prompt"], **{str(obj["id"]): obj["geometry_prompt"] for obj in objects}})
    write_json(output / f"{prefix}_prompt_texture.json", {"0": background["texture_prompt"], **{str(obj["id"]): obj["texture_prompt"] for obj in objects}})
    write_json(output / f"{prefix}_location_modify.json", {str(obj["id"]): int(obj["floor_supported"]) for obj in objects})
    evaluation = output / "eval-novel-views"
    (evaluation / "rgb").mkdir(parents=True)
    (evaluation / "instance_mask").mkdir()
    camera_subset(render if render else test, evaluation / "cameras.npz")
    test_ids = [render.index(i) if render else j for j, i in enumerate(test)]
    write_json(evaluation / "eval-10views.json", test_ids)
    for camera_id, frame_id in zip(test_ids, test):
        Image.open(work / "prepared" / "rgb" / f"{frame_id:06d}.jpg").save(evaluation / "rgb" / f"{camera_id:06d}_rgb.png")
        shutil.copy2(work / "labels" / f"{frame_id:06d}.png", evaluation / "instance_mask" / f"{camera_id:06d}.png")
    provenance = dict(train=train, test=test, render=render, evaluation_camera_ids=test_ids, frames=manifest["frames"], size=manifest["size"], object_count=len(objects), up_source=manifest["up_source"], metric_scale_known=False, dp_recon_full_camera_layout=bool(render), heldout_policy="Test RGB/masks are excluded from training export. SfM uses all candidate images for poses; disclose this protocol.")
    write_json(output / "datagen_manifest.json", provenance)
    shutil.copy2(args.annotations, output / "annotations.used.json")
    text = f"""# Dataset configuration notes
Generated from one common source for both baselines. No baseline code was modified.

Both: data_dir={args.data_dir}, scan_id={args.scan_id}, img_res=[{manifest['size']},{manifest['size']}], center_crop_type=no_crop, num_views=-1.
ObjectSDF++: model.implicit_network.d_out={len(objects)+1}; use_mask=True.
ObjectSDF++ resolves ../data/<data_dir>/scan<id> relative to its code working directory. Arrange a symlink manually later or use an isolated run/code directory with ../data linked to this dataset's root; this tool does not write to the baseline.
DP-Recon: data_root_dir must be parent of '{args.data_dir}'; final path must end in {args.data_dir}/scan{args.scan_id}.
DP auxiliary renderer cameras: {len(render)}. Full current 100-camera flow: {'YES' if render else 'NO: training-only; adapt renderer before full pipeline'}.
The supplied JSONs contain background channel 0, foreground instance prompts, and foreground floor flags.
This exporter prepares data, not a guarantee that every scene-specific assumption in the full DP-Recon pipeline is removed.
Evaluate test RGB/masks separately; the ObjectSDF++ default train/test visualization split is not an independent test.
Monocular video has no independent complete 3D ground truth or known metric scale.
"""
    (output / "CONFIG_NOTES.md").write_text(text, encoding="utf-8")
    print(f"Exported {len(train)} training / {len(test)} held-out images: {output}")


def main():
    parser = argparse.ArgumentParser(description="Video -> shared DP-Recon / ObjectSDF++ data. Independent, restartable stages; no baseline writes.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--video", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--fps", type=float, default=2)
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--min-sharpness", type=float, default=0)
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--end", type=float)
    p.set_defaults(func=extract)
    p = sub.add_parser("sfm")
    p.add_argument("--work", required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--features", type=int, default=8192)
    p.add_argument("--matching", choices=["sequential", "exhaustive"], default="sequential")
    p.add_argument("--overlap", type=int, default=10)
    p.set_defaults(func=sfm)
    p = sub.add_parser("prepare")
    p.add_argument("--work", required=True)
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--up", type=float, nargs=3, help="World-up vector in original COLMAP coordinates; omission estimates from camera vertical, needs visual verification")
    p.set_defaults(func=prepare)
    p = sub.add_parser("split")
    p.add_argument("--work", required=True)
    p.add_argument("--train-views", type=int, default=10)
    p.add_argument("--test-views", type=int, default=10)
    p.set_defaults(func=split_frames)
    p = sub.add_parser("segment")
    p.add_argument("--work", required=True)
    p.add_argument("--annotations", required=True)
    p.add_argument("--checkpoint", default=str(ROOT / "models" / "sam2.1_hiera_small.pt"))
    p.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    p.add_argument("--device", default="cuda:0")
    p.set_defaults(func=segment)
    p = sub.add_parser("priors")
    p.add_argument("--work", required=True)
    p.add_argument("--models", default=str(ROOT / "models"))
    p.add_argument("--omnidata", default=str(ROOT / "repos" / "omnidata"))
    p.add_argument("--device", default="cuda:0")
    p.set_defaults(func=priors)
    p = sub.add_parser("preview")
    p.add_argument("--work", required=True)
    p.add_argument("--count", type=int, default=12)
    p.set_defaults(func=preview)
    p = sub.add_parser("export")
    p.add_argument("--work", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--annotations", required=True)
    p.add_argument("--train-views", type=int, default=10)
    p.add_argument("--test-views", type=int, default=10)
    p.add_argument("--split", help="Optional JSON containing explicit prepared frame index lists train/test")
    p.add_argument("--render-count", type=int, default=100)
    p.add_argument("--data-dir", default="custom")
    p.add_argument("--scan-id", type=int, default=1)
    p.set_defaults(func=export)
    args = parser.parse_args()
    if args.command == "extract" and (args.fps <= 0 or args.max_frames < 3 or args.start < 0 or (args.end is not None and args.end <= args.start)):
        parser.error("Require fps>0, max_frames>=3 and 0<=start<end")
    if args.command in ("preview", "prepare") and getattr(args, "count", getattr(args, "size", 1)) <= 0:
        parser.error("size/count must be positive")
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
