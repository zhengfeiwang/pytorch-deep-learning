#!/usr/bin/env python3
import os
import copy
import argparse
import matplotlib
matplotlib.use('Agg') # must before import pyplot
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from tensorboardX import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()


def train_model(dataloader, model, criterion, optimizer, n_epochs, dataset_size):
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch + 1, n_epochs))
        print('-' * 10)
        epoch_loss = {'train': 0.0, 'val': 0.0}
        epoch_accuracy = {'train': 0.0, 'val': 0.0}

        for phase in ['train', 'val']:
            running_loss = 0.0
            running_corrects = 0
            if phase is 'train':
                model.train()
            else:
                model.eval()

            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase is 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss[phase] = running_loss / dataset_size[phase]
            epoch_accuracy[phase] = running_corrects.double() / dataset_size[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss[phase], epoch_accuracy[phase]))

            if phase is 'val' and epoch_accuracy[phase] > best_acc:
                best_acc = epoch_accuracy[phase]
                best_model_weights = copy.deepcopy(model.state_dict())

        writer.add_scalars('loss', {'train': epoch_loss['train'], 'val': epoch_loss['val']}, epoch)
        writer.add_scalars('accuracy', {'train': epoch_accuracy['train'], 'val': epoch_accuracy['val']}, epoch)
        print()
        
    print('Best val Acc: {:.4f}'.format(best_acc))
    model.load_state_dict(best_model_weights)
    return model


def evaluate_model(dataloader, model, dataset_size):
    model.eval()
    corrects = 0
    for inputs, labels in dataloader['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        corrects += torch.sum(preds == labels.data)
    acc = corrects.double() / dataset_size['test']
    return acc.item()


def model_finetune(number_class):
    model = models.resnet18(pretrained=True)
    number_features = model.fc.in_features
    model.fc = nn.Linear(number_features, number_class)
    model = model.to(device)
    return model


def main(args):
    torch.manual_seed(args.seed)    # set random seed
    if os.path.exists(args.weights_file) is False:
        args.evaluation = False
    dataloader, dataset_size = load_data(args.evaluation, args.data_dir, args.batch_size, args.number_worker)
    if args.evaluation is False:
        model = model_finetune(args.number_class)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        model = train_model(dataloader, model, criterion, optimizer, args.number_epoch, dataset_size)
        torch.save(model.state_dict(), args.weights_file)
    else:
        model = model_finetune(args.number_class)
        model.load_state_dict(torch.load(args.weights_file))
        test_acc = evaluate_model(dataloader, model, dataset_size)
        print('finetuned model performance (accuracy) on test set: {:.4f}'.format(test_acc))
    writer.close()


def load_data(evaluation, data_dir, batch_size, number_worker):
    if evaluation is False:
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x), data_transforms[x])
                        for x in ['train', 'val']}
        dataset_size = {x: len(image_datasets[x]) for x in ['train', 'val']}
        dataloader = {x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=number_worker) for x in ['train', 'val']}
    else:
        data_transforms = {
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        image_datasets = {'test': datasets.ImageFolder(os.path.join(args.data_dir, 'test'), data_transforms['test'])}
        dataset_size = {'test': len(image_datasets['test'])}
        dataloader = {'test': torch.utils.data.DataLoader(
            image_datasets['test'], batch_size=batch_size, shuffle=True, num_workers=number_worker)}
    return dataloader, dataset_size


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch transfer learning for 17 Category Flower Dataset')
    parser.add_argument('--evaluation', default=False, action='store_true', help='whether evaluate the model (default: False)')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory (default: data/)')
    parser.add_argument('--number_class', type=int, default=17, help='number of flower class (default: 17)')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size (default: 8)')
    parser.add_argument('--number_epoch', type=int, default=50, help='number of epoch for training (default: 50)')
    parser.add_argument('--number_worker', type=int, default=16, help='number of multiprocess worker (default: 16)')
    parser.add_argument('--seed', type=int, default=730, help='random seed (default: 730)')
    parser.add_argument('--weights_file', type=str, default='weights.pt', help='model weights file path (default: weights.pt)')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
