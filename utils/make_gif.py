"""
Script used to make a gif from the generated images.

Author: Shubham Singhal
Github: shubham21197

Usage: python make_gif.py max_sampels suffix
max_samples: Number of images to insert into the GIF (defaul: 3010)
suffix: Suffix to add to the folder name (default: '')
"""

from PIL import Image
import sys

num_frames = 3010
suffix = ''

if len(sys.argv) > 1:
    num_frames = int(sys.argv[1])

    if len(sys.argv) > 2:
        suffix = sys.argv[2]

frames = []
for i in range(0, num_frames, 10):
    frames.append(Image.open('../data/genned' + suffix + '/image_at_epoch_{:04d}.png'.format(i)))

frame_one = frames[0]
_frames = []

for i in range(1, 11):
    _frames = _frames + [frames[i]] * (11 - i)

frames = _frames + frames[10:]

for _ in range(20):
    frames.append(frames[-1])

frame_one.save("training.gif", format="GIF", append_images=frames[1:],
            save_all=True, duration=120, loop=0)
    