import os
from random import choice, triangular, randint

from PIL import Image

__dir__ = os.path.dirname(os.path.abspath(__file__))

CONFIG_DIR = os.path.join(__dir__, 'config')
CLASS_LIST_FILE = os.path.join(__dir__, 'config', 'class_list.txt')
TRAIN_DIR = os.path.join(__dir__, 'data')

bg_image = Image.open('%s/trainUtils/BJ.jpg' % (__dir__)).convert('RGBA')


def square_image(image):
    big_size, small_size = (max(image.size), min(image.size))

    square_bg_image = Image.new('RGBA', (big_size, big_size), (0, 0, 0))

    x1, y1, x2, y2 = image.getbbox()

    square_bg_image.paste(image, image.getbbox(), image)
    square_bg_image.paste(image, (x1, y1 + small_size, x2, y2 + small_size), image)

    return square_bg_image


def bbox(center, image):
    cx, cy = center
    w, h = image.size

    x1 = int(cx - 0.5 * w)
    y1 = int(cy - 0.5 * h)

    return (x1, y1, x1 + w, y1 + h)


square_bg_image = square_image(bg_image)

image_size = square_bg_image.size[0]

class_list = [fn.strip() for fn in open(CLASS_LIST_FILE).readlines()]
print(class_list)

class_list_dict = dict((fn, index) for (index, fn) in enumerate(class_list))

class_images = [Image.open(os.path.join(CONFIG_DIR, fn)).convert('RGBA') for fn in class_list]

for id in range(2000):
    labels = []

    n_objects = randint(1, 12)
    canvas = square_bg_image.copy()

    for n in range(n_objects):
        picked_class = class_list_dict[choice(class_list)]
        picked_image = class_images[picked_class]

        scale = triangular(0.5, 0.8)

        orig_size = picked_image.size

        picked_image = picked_image.copy().resize((int(v * scale) for v in orig_size))

        w, h = [s * 1.0 / image_size for s in picked_image.size]

        cx, cy = (triangular(0.5 * w, 1 - 0.5 * w), triangular(0.5 * h, 1 - 0.5 * h))

        label = (picked_class, cx, cy, w, h)

        labels.append(label)

        canvas.paste(picked_image, bbox((cx * image_size, cy * image_size), picked_image), picked_image)

    canvas.resize((512, 512)).save(os.path.join(TRAIN_DIR, '%05d.png' % (id + 1)))

    f = open(os.path.join(TRAIN_DIR, '%05d.txt' % (id + 1)), 'w')
    with f:
        f.write('\n'.join(' '.join(str(v) for v in l) for l in labels))
