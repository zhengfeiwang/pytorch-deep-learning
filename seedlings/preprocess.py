import os
import shutil
import random
import matplotlib.pyplot as plt


def process_dir(src_dir, dst_dir, split_size):
    src_train = os.path.join(src_dir, 'train')
    src_test = os.path.join(src_dir, 'test')

    categories = os.listdir(src_train)

    # check dst dir exist
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    dst_train = os.path.join(dst_dir, 'train')
    if not os.path.exists(dst_train):
        os.mkdir(dst_train)
    dst_val = os.path.join(dst_dir, 'val')
    if not os.path.exists(dst_val):
        os.mkdir(dst_val)
    dst_test = os.path.join(dst_dir, 'test')
    if not os.path.exists(dst_test):
        os.mkdir(dst_test)

    # split and copy train set
    for category in categories:
        tmp_dir = os.path.join(src_train, category)
        cur_train = os.path.join(dst_train, category)
        cur_val = os.path.join(dst_val, category)
        os.mkdir(cur_train)
        os.mkdir(cur_val)

        filenames = os.listdir(tmp_dir)
        random.shuffle(filenames)
        train_files = filenames[int(len(filenames) * split_size):]
        val_files = filenames[:int(len(filenames) * split_size)]

        for filename in train_files:
            shutil.copyfile(os.path.join(tmp_dir, filename), os.path.join(cur_train, filename))
        for filename in val_files:
            shutil.copyfile(os.path.join(tmp_dir, filename), os.path.join(cur_val, filename))

    # copy test set
    for filename in os.listdir(src_test):
        shutil.copyfile(os.path.join(src_test, filename), os.path.join(dst_test, filename))


def plot_distribution(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    # get class labels and amount of train set
    categories = os.listdir(train_dir)
    print('categories:', categories)
    amount = [len(os.listdir(os.path.join(train_dir, category))) for category in categories]
    print('amount:', amount)

    # plot the distribution bar
    plt.figure(figsize=(25, 10))
    plt.bar(range(len(amount)), amount, tick_label=categories)
    plt.title('distribution')
    plt.xlabel('categories')
    plt.ylabel('amount')
    plt.savefig('distribution.png', dpi=300)

    # explore test set
    nb_test = len(os.listdir(test_dir))
    print('test set size:', nb_test)


if __name__ == "__main__":
    plot_distribution('./download')
    process_dir('./download', './data', 0.1)
