"""Geometry/export tests with explicitly synthetic data; no model or network needed."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from datagen.cli import crop_intrinsics, rotation_between, evenly, export, write_json, extract


def test_camera_projection_crop_and_similarity():
    K = np.array([[600., 0, 320], [0, 600, 240], [0, 0, 1]])
    new_K, box, A = crop_intrinsics(K, 640, 480, 384)
    point = np.array([0.3, 0.2, 2.0, 1.0])
    R = rotation_between([0, -1, 0], [0, 0, 1])
    rt = np.c_[np.eye(3), [0.1, 0.2, 2.0]]
    aligned_rt = rt.copy()
    aligned_rt[:, :3] = rt[:, :3] @ R.T
    aligned_point = np.r_[R @ point[:3], 1]
    S = np.eye(4)
    S[:3, :3] *= 3
    S[:3, 3] = [0.2, 0.5, 0.7]
    normalized = np.linalg.inv(S) @ aligned_point
    expected = A @ K @ rt @ point
    actual = new_K @ aligned_rt @ S @ normalized
    np.testing.assert_allclose(actual / actual[2], expected / expected[2], atol=1e-8)
    np.testing.assert_allclose(R @ np.array([0, -1, 0]), [0, 0, 1], atol=1e-8)
    np.testing.assert_allclose(rotation_between([0, 0, -1], [0, 0, 1]) @ [0, 0, -1], [0, 0, 1], atol=1e-8)


def make_fixture(base, n=100):
    work = base / 'work'
    for directory in ('prepared/rgb', 'labels', 'priors'):
        (work / directory).mkdir(parents=True)
    size, frames, mats = 16, [], {}
    K = np.array([[20.,0,8],[0,20,8],[0,0,1]])
    for i in range(n):
        stem = f'{i:06d}'
        Image.fromarray(np.full((size,size,3), i, np.uint8)).save(work/'prepared/rgb'/f'{stem}.jpg')
        label = np.zeros((size,size), np.uint8)
        label[2:14,2:14] = 1
        Image.fromarray(label).save(work/'labels'/f'{stem}.png')
        for kind, arr in [('depth', np.ones((size,size),np.float32)), ('normal', np.full((3,size,size),0.5,np.float32)), ('mask', np.ones((size,size),np.float32))]:
            np.save(work/'priors'/f'{stem}_{kind}.npy', arr)
        W = np.eye(4)
        W[:3,:4] = K @ np.c_[np.eye(3), [i*0.01,0,2]]
        mats[f'world_mat_{i}'] = W
        mats[f'scale_mat_{i}'] = np.eye(4)
        frames.append(dict(index=i,stem=stem,K=K.tolist(),source=dict(source_frame=i,timestamp_sec=i/2)))
    write_json(work/'prepared/manifest.json', dict(size=size, frames=frames,up_source='explicit'))
    np.savez(work/'prepared/cameras.npz', **mats)
    annotation = base/'annotations.json'
    write_json(annotation, dict(background=dict(geometry_prompt='An empty room',texture_prompt='White walls'),objects=[dict(id=1,name='object_1', geometry_prompt='A cube',texture_prompt='A red cube',floor_supported=True)]))
    return work, annotation


def test_export_and_existing_loader_compatible_layout(tmp_path):
    work, annotation = make_fixture(tmp_path)
    output = tmp_path/'datasets/custom/scan1'
    export(argparse.Namespace(work=work,output=output,annotations=annotation,split=None,train_views=10,test_views=10,render_count=100,data_dir='custom',scan_id=1))
    result = json.loads((output/'datagen_manifest.json').read_text())
    assert set(result['train']).isdisjoint(result['test'])
    assert len(list(output.glob('*_rgb.png'))) == 10
    assert len(np.load(output/'eval-novel-views/cameras.npz').files) == 200
    for train_id, source_id in enumerate(result['train']):
        rgb = np.asarray(Image.open(output/f'{train_id:06d}_rgb.png'))
        assert int(rgb[0,0,0]) == source_id
        np.testing.assert_array_equal(np.asarray(Image.open(output/'instance_mask'/f'{train_id:06d}.png')),np.asarray(Image.open(output/'segs'/f'{train_id:06d}_segs.png')))
        np.testing.assert_array_equal(np.load(output/'cameras.npz')[f'world_mat_{train_id}'],np.load(work/'prepared/cameras.npz')[f'world_mat_{source_id}'])
    with pytest.raises(ValueError, match='already exists'):
        export(argparse.Namespace(work=work,output=output,annotations=annotation,split=None,train_views=10,test_views=10,render_count=100,data_dir='custom',scan_id=1))


def test_even_sampling_no_duplicates():
    for count in (1, 10, 99, 100):
        assert len(set(evenly(list(range(100)), count))) == count


def test_video_extract_preserves_frame_ids(tmp_path):
    source = tmp_path/'test.avi'
    writer = cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),10,(64,48))
    for i in range(20):
        writer.write(np.full((48,64,3),i*10,np.uint8))
    writer.release()
    extract(argparse.Namespace(video=source,work=tmp_path/'extract',start=0,end=None,fps=2,max_frames=20,min_sharpness=0))
    data = json.loads((tmp_path/'extract/extraction.json').read_text())
    assert [r['source_frame'] for r in data['frames']] == [0,5,10,15]
