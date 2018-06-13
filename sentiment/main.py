import argparse
import random
from collections import Counter
import torch
import torch.nn as nn
from torchtext import vocab
from model import RNN
from utils import *
from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()


def evaluate(data, model, batch_size, word2vec):
    model.eval()
    corrects = 0

    hidden = model.init_hidden(batch_size).to(device)

    for i in range(len(data) // batch_size):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        inputs, labels = get_batch(data, start_idx, end_idx, word2vec, args.sequence_length)
        inputs = inputs.cuda()
        labels = labels.to(device)

        hidden = hidden.detach()

        output, hidden = model(inputs, hidden)
        _, predicted = torch.max(output.data, 1)
        corrects += torch.sum(predicted == labels.data)

    acc = corrects.double() / len(data)
    print('acc:', acc.item())
    return acc.item()


def train(data, model, criterion, optimizer, batch_size, nb_epoch, word2vec):
    model.train()
    for epoch in range(1, nb_epoch + 1):
        hidden = model.init_hidden(batch_size).to(device)

        epoch_loss = 0.0

        for i in range(len(data) // batch_size):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            inputs, labels = get_batch(data, start_idx, end_idx, word2vec, args.sequence_length)
            inputs = inputs.cuda()
            labels = labels.to(device)

            hidden = hidden.detach()

            optimizer.zero_grad()
            output, hidden = model(inputs, hidden)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        print('Epoch {}/{}, Loss: {:.4f}'.format(epoch, nb_epoch, epoch_loss))
        writer.add_scalars('loss', {'train': epoch_loss}, epoch)


def main(args):
    # prepare data
    train_texts, train_labels = read_data(os.path.join(args.data_dir, 'train'))
    test_texts, test_labels = read_data(os.path.join(args.data_dir, 'test'))
    training_set = list(zip(train_texts, train_labels))
    test_set = list(zip(test_texts, test_labels))
    random.shuffle(training_set)
    random.shuffle(test_set)

    vocab_counter = Counter(flatten([get_words(text) for text in train_texts]))
    word2vec = vocab.Vocab(vocab_counter, max_size=20000, min_freq=3, vectors='glove.6B.100d')

    model = RNN(args.input_size, args.hidden_size, args.nb_class)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train(training_set, model, criterion, optimizer, args.batch_size, args.nb_epoch, word2vec)
    evaluate(test_set, model, args.batch_size, word2vec)

    torch.save(model.state_dict(), args.weights_file)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Sentiment Classification for Large Movie Review Dataset')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--nb_epoch', default=50, type=int, help='number of epoch')

    parser.add_argument('--input_size', default=100, type=int, help='RNN input size')
    parser.add_argument('--nb_class', default=2, type=int, help='batch size')
    parser.add_argument('--hidden_size', default=50, type=int, help='RNN hidden size')
    parser.add_argument('--sequence_length', default=500, type=int, help='sequence length')

    parser.add_argument('--data_dir', default='aclImdb', type=str)
    parser.add_argument('--weights_file', type=str, default='weights.pt', help='model weights file path')

    args = parser.parse_args()
    main(args)
