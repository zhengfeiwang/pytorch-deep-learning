import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from tensorboardX import SummaryWriter

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


def train(model, dataloader, dataset_size, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    writer = SummaryWriter()

    for epoch in range(1, args.num_epoch + 1):
        epoch_loss = {'train': 0.0, 'val': 0.0}
        epoch_accuracy = {'train': 0.0, 'val': 0.0}

        for mode in ['train', 'val']:
            if mode is 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader[mode]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(mode == 'train'):
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)
                    if mode is 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predictions == labels.data)

            epoch_loss[mode] = running_loss / dataset_size[mode]
            epoch_accuracy[mode] = running_corrects.double() / dataset_size[mode]
            print('Epoch {} / {}: {} Loss: {:.4f} Acc: {:.4f}'.format(
                epoch, args.num_epoch, mode, epoch_loss[mode], epoch_accuracy[mode])
            )

        print()

        # write into TensorBoard
        writer.add_scalars('loss', {'train': epoch_loss['train'], 'val': epoch_loss['val']}, epoch)
        writer.add_scalars('accuracy', {'train': epoch_accuracy['train'], 'val': epoch_accuracy['val']}, epoch)

    writer.close()

    return model


def pretrained_model(num_class):
    model = models.resnet50(pretrained=True)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_class)

    model = model.to(device)

    return model


def get_dataloader(args):
    modes = ['train', 'val']

    transformer = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(args.dim),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.RandomResizedCrop(args.dim_test),
            transforms.CenterCrop(args.dim),
            transforms.ToTensor()
        ])
    }

    dataset = {mode: datasets.ImageFolder(os.path.join(args.data_dir, mode), transformer[mode]) for mode in modes}
    dataset_size = {mode: len(dataset[mode]) for mode in modes}

    dataloader = {mode: torch.utils.data.DataLoader(
        dataset[mode], batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker
    ) for mode in modes}

    return dataloader, dataset_size


def main(args):
    if args.seed > 0:
        torch.manual_seed(args.seed)    # set random seed

    dataloader, dataset_size = get_dataloader(args)
    model = pretrained_model(args.num_class)
    model = train(model, dataloader, dataset_size, args)
    torch.save(model.state_dict(), args.weights_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('[PyTorch] Kaggle Competition: Plant Seedlings Classification')

    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--weights_file', default='./weights.pt', type=str)

    parser.add_argument('--num_class', default=12, type=int)
    parser.add_argument('--dim', default=224, type=int, help='width and height of image input')
    parser.add_argument('--dim_test', default=256, type=int, help='width and height of test image input')

    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='minibatch size')
    parser.add_argument('--num_epoch', default=20, type=int, help='number of epoch')

    parser.add_argument('--num_worker', default=16, type=int, help='multiprocess worker')
    parser.add_argument('--seed', default=-1, type=int, help='random seed')

    args = parser.parse_args()

    main(args)
