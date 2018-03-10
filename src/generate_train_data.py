from random import choice, random, triangular
from uuid import uuid4

from PIL import Image, ImageDraw

bg_image = Image.open('./trainUtils/BJ.jpg').convert('RGBA')


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

class_list = map(lambda file_name: file_name.strip(), open('./config/class_list.txt').readlines())

class_list_dict = dict((fn, index) for (index, fn) in enumerate(class_list))

class_images = map(lambda fn: Image.open('./config/%s' % (fn)).convert('RGBA'), class_list)

for id in xrange(1, 3):
    picked_class = class_list_dict[choice(class_list)]
    picked_image = class_images[picked_class]

    w, h = [s * 1.0 / image_size for s in picked_image.size]

    cx, cy = (triangular(0.5 * w, 1 - 0.5 * w), triangular(0.5 * h, 1 - 0.5 * h))

    label = (picked_class, cx, cy, w, h)

    canvas = square_bg_image.copy()

    canvas.paste(picked_image, bbox((cx * image_size, cy * image_size), picked_image), picked_image)

    canvas.save('./data/%05d.png' % (id))

    f = open('./data/%05d.txt' % (id), 'wa')

    with f:
        f.write(' '.join(str(v) for v in label))
