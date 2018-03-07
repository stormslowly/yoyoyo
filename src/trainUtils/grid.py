import torchvision.transforms as T
from PIL import Image
from PIL import ImageDraw
from matplotlib import pyplot as plt

image = Image.open('./BJ.jpg').convert("RGBA")
btn_image = Image.open('./login_youke.png').convert("RGBA")


def put_grid_on(image):
    image_draw = ImageDraw.Draw(image)

    x_grid_size = 40
    y_grid_size = 40

    w, h = image.size

    n_x_grid = w / x_grid_size
    n_y_grid = h / y_grid_size

    for i in range(n_x_grid):
        x = (i + 1) * x_grid_size
        image_draw.line((x, 0, x, h), fill=0, width=4)

    for i in range(n_y_grid):
        y = (i + 1) * y_grid_size
        image_draw.line((0, y, w, y), fill=0, width=4)

    return image


scale_to = T.Compose(
    [
        # T.Scale((800, 600)),
        # T.ToTensor(),
        put_grid_on,
        # T.ToPILImage(),
    ]
)


def paste_to(to_paste_img, pos, convas):
    image_w, image_h = to_paste_img.size
    x, y = pos
    target_position = (x - image_w / 2, y - image_h / 2)
    convas.paste(to_paste_img, target_position, to_paste_img)

    return convas


def draw_rectangle(image):
    image_draw = ImageDraw.Draw(image)

    im_w, im_h = image.size

    center_x = im_w / 2
    center_y = im_h / 2

    image_draw.rectangle(xy=[(1, 1), (im_w - 2, im_h - 2)], outline=(255, 50, 0))
    image_draw.ellipse(xy=[center_x - 4, center_y - 4, center_x + 4, center_y + 4],
                       fill=(255, 0, 0))

    return image


paste_to(draw_rectangle(btn_image), (200, 200), image)

image = scale_to(image)
image.save('./BJ.grid.png')

plt.imshow(image)
plt.show()
#
# print(image.data)
#
# print("end? yoo")
