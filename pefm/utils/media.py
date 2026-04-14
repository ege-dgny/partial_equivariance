"""
Utilities for loading, saving, and manipulating videos and images.
"""

import os
import sys
import numpy as np
import cv2
import skvideo
# Point skvideo at the conda env's ffmpeg if not on PATH
_env_bin = os.path.join(os.path.dirname(sys.executable))
if os.path.exists(os.path.join(_env_bin, "ffmpeg")):
    skvideo.setFFmpegPath(_env_bin)
import skvideo.io
import imageio


def _make_dir(filename):
    folder = os.path.dirname(filename)
    os.makedirs(folder, exist_ok=True)


def save_image(image, filename):
    _make_dir(filename)
    if np.max(image) < 2:
        image = np.array(image * 255)
    image = image.astype(np.uint8)
    cv2.imwrite(filename, image[..., ::-1])


def save_gif(video_array, file_path, fps=10):
    try:
        video_array = (255 * (1.0 - video_array)).astype("uint8")
        imageio.mimsave(file_path, video_array, duration=len(video_array) / fps, loop=1)
        print(f"Saved GIF to {file_path}")
    except Exception as e:
        print(f"Error saving GIF: {e}")


def save_video(video_frames, filename, fps=10, video_format="mp4"):
    if len(video_frames) == 0:
        return False

    assert fps == int(fps), fps
    _make_dir(filename)

    skvideo.io.vwrite(
        filename,
        video_frames,
        inputdict={
            "-r": str(int(fps)),
        },
        outputdict={"-f": video_format, "-pix_fmt": "yuv420p"},
    )

    return True


def read_video(filename):
    return skvideo.io.vread(filename)


def get_video_framerate(filename):
    videometadata = skvideo.io.ffprobe(filename)
    frame_rate = videometadata["video"]["@avg_frame_rate"]
    return eval(frame_rate)


def combine_videos(images, num_cols=5):
    if len(images) == 1:
        return np.array(images[0])
    max_frames = np.max([len(im) for im in images])
    images = [
        np.concatenate([im[:-1], np.array([im[-1]] * (max_frames - len(im) + 1))])
        for im in images
    ]
    images = np.array(images)
    B = images.shape[0]
    if B % num_cols != 0:
        images = np.concatenate(
            [images, np.zeros((num_cols - (B % num_cols),) + tuple(images.shape[1:]))]
        )
    B, T, H, W, C = images.shape
    images = images.reshape(B // num_cols, num_cols, T, H, W, C).transpose(
        2, 0, 3, 1, 4, 5
    )
    images = images.reshape(T, B // num_cols * H, num_cols * W, C)
    return images
