import torchvision.transforms as T
import torch

from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from PIL import Image

image = Image.open('./BJ.jpg')


def put_grid_on(image):
    image_draw = ImageDraw.Draw(image)

    n_x_grid = 16
    n_y_grid = 12

    x_grid_size = 800 / n_x_grid
    y_grid_size = 600 / n_y_grid

    for i in range(n_x_grid):
        x = (i + 1) * x_grid_size
        image_draw.line((x, 0, x, 600), fill=0, width=2)

    for i in range(n_y_grid):
        y = (i + 1) * y_grid_size
        image_draw.line((0, y, 800, y), fill=0, width=2)

    return image


scale_to = T.Compose(
    [
        T.Scale((800, 600)),
        # T.ToTensor(),
        put_grid_on,
        # T.ToPILImage(),
    ]
)

grid_color = [0, 0, 0]

image = scale_to(image)

image.save('./BJ.grid.jpg')

plt.imshow(image)
plt.show()
#
# print(image.data)
#
# print("end? yoo")
