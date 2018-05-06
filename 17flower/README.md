# Tranfer Learning for 17 Category Flower Dataset

### Environment

- PyTorch 0.4.0
- matplotlib 2.2.2

### Dataset

17 Category Flower Dataset ([view](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/)) consists of 1360 images of flowers with 80 in each category. The directory of data should look like ./data/[mode (train/val/test)]/[class id (0~16)]/[name].jpg for Python script running.

### Model

Pretrained ResNet-18 available in torchvision, finetune using a new fc-layer.

### Result

#### Loss

![loss](./loss.png)

#### Accuracy

![acc](./acc.png)