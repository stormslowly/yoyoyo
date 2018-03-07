import torch
from PIL.ImageDraw import ImageDraw
from torchvision import transforms
from matplotlib import pyplot as plt

import dataset

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

init_width, init_height = (160, 160)

test_loader = torch.utils.data.DataLoader(
    dataset.listDataset('../voc_train.txt', shape=None,
                        shuffle=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                        ]), train=True),
    batch_size=1, shuffle=False, )

iter = enumerate(test_loader)

(i, [targets, labels]) = iter.next()

to_image = transforms.ToPILImage()

target_image = to_image(targets[0])
label = labels[0]

print(label.size())

viewd = label.view(-1, 5)

(class_index, x, y, w, h) = viewd[0]

draw = ImageDraw(target_image)

for i in range(viewd.size(0)):
    (class_index, x, y, w, h) = viewd[i]
    if class_index == 0 and x == 0:
        break
    draw.rectangle(((x - 0.5 * w) * 416, (y - 0.5 * h) * 416, (x + 0.5 * w) * 416, (y + 0.5 * h) * 416),
                   outline=3)
    print(classes[int(class_index)])

print(targets[0].size())

plt.imshow(target_image)
plt.show()
