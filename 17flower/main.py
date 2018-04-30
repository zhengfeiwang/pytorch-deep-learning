import os
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

N_CLASSES = 17
N_EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
NUM_WORKERS = 16

epoch_summary = []
train_loss_summary = []
train_acc_summary = []
val_loss_summary = []
val_acc_summary = []

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
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
data_dir = 'data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val', 'test']}
dataloader = {x: torch.utils.data.DataLoader(
    image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    for x in ['train', 'val', 'test']}
class_name = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, n_epochs=N_EPOCHS):
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch, n_epochs - 1))
        print('-' * 10)
        epoch_summary.append(epoch)
        for phase in ['train', 'val']:
            if phase is 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

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

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase is 'train':
                train_loss_summary.append(epoch_loss)
                train_acc_summary.append(epoch_acc)
            else:
                val_loss_summary.append(epoch_loss)
                val_acc_summary.append(epoch_acc)

            if phase is 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
        print()

    print('Best val Acc: {:.4f}'.format(best_acc))
    model.load_state_dict(best_model_weights)

    return model


# test model
def test_model(model):
    model.eval()
    corrects = 0
    for inputs, labels in dataloader['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        corrects += torch.sum(preds == labels.data)
    
    acc = corrects.double() / len(image_datasets['test'])
    return acc.item()


if __name__ == "__main__":
    model_finetune = models.resnet18(pretrained=True)
    num_features = model_finetune.fc.in_features
    model_finetune.fc = nn.Linear(num_features, N_CLASSES)
    model_finetune = model_finetune.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_finetune = optim.Adam(model_finetune.parameters(), lr=LEARNING_RATE)

    model_finetune = train_model(model_finetune, criterion, optimizer_finetune, n_epochs=N_EPOCHS)
    test_acc = test_model(model_finetune)
    print("Finetuned ResNet's accuracy on test set: ", test_acc)

    plt.figure()
    plt.title("loss - Finetuning ResNet on 17 Category Flower Dataset")
    plt.plot(epoch_summary, train_loss_summary, label='train')
    plt.plot(epoch_summary, val_loss_summary, label='val')
    plt.legend()
    plt.grid()
    plt.savefig("loss.png", dpi=300)

    plt.figure()
    plt.title("accuracy - Finetuning ResNet on 17 Category Flower Dataset")
    plt.plot(epoch_summary, train_acc_summary, label='train')
    plt.plot(epoch_summary, val_acc_summary, label='val')
    plt.legend()
    plt.grid()
    plt.savefig("accuracy.png", dpi=300)
