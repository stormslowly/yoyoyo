import torchvision.transforms as T
import torch

from  matplotlib import pyplot as plt

from PIL import Image

image = Image.open('./BJ.jpg')


def put_grid_on(torsor):
    np = torsor.numpy()
    print(np.shape)

    dx = 800 / 40

    np[0:, ::dx, :] = 0
    np[1:, ::dx, :] = 0
    np[2:, ::dx, :] = 0

    dy = int(600 / 27)

    np[0:, :, ::dy] = 0
    np[1:, :, ::dy] = 0
    np[2:, :, ::dy] = 0

    return torch.from_numpy(np)


scaleto = T.Compose(
    [
        T.Scale((800, 600)),
        T.ToTensor(),
        put_grid_on,
        T.ToPILImage(),
    ]
)

grid_color = [0, 0, 0]

image = scaleto(image)

print(image)
gridSize = (40, 27)

plt.imshow(image)
plt.show()

print("end? yoo")
