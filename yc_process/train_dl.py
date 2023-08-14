from util.visualization.visualize_loss import loss_visualize
from prefetch_generator import BackgroundGenerator
from dataloaders.NinaPro_dataset import *
from util.parse_config import parse_config
from torch.utils.data import DataLoader
from losses.loss_function import FocalLossV2, FocalLoss
from networks.NetFactory import NetFactory
import torchvision.transforms as tt
import torch.optim as optim
import torch.nn as nn
import torch
import time
import dataloader_yc


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class AccuracyMetrics:
    def __init__(self, class_num=None):
        self.shot = np.zeros(class_num)
        self.total = np.zeros(class_num) + 1e-5  # avoid divide zero

    def get_data(self, prediction, label):
        _, predicted_label = torch.max(prediction, dim=1)
        for i in range(len(predicted_label)):
            id = label[i].cpu()
            self.total[id] += 1
            if predicted_label[i] == id:
                self.shot[id] += 1

    def evaluate(self):
        class_accuracy = self.shot / self.total
        mean_accuracy = np.mean(class_accuracy)
        return mean_accuracy, class_accuracy

    def clean(self):
        self.shot *= 0
        self.total *= 0
        self.total += 1e-5


def plot_acc(train_class_accuracy, valid_class_accuracy, net_name, batch_size):
    length = train_class_accuracy.shape[0]
    width = 0.35
    x = np.arange(length)
    tick_label = x
    plt.bar(x, train_class_accuracy, width=width, label='train class accuracy')
    plt.bar(x + width, valid_class_accuracy, width=width, label='valid class accuracy')
    plt.title('Net type: {} Batch size: {}'.format(net_name, batch_size))
    plt.gca().set(xlabel='label id', ylabel='accuracy')
    plt.legend()
    plt.xticks(x + width / 2, tick_label)
    plt.show()


def train(config):
    # load data config
    config_data = config['data']
    cutoff_frequency = config_data['cutoff_frequency']
    sampling_frequency = config_data['sampling_frequency']
    # wn = 2 * cutoff_frequency / sampling_frequency
    wn = None
    window_length = config_data['window_length']
    batch_size = config_data['batch_size']
    iter_num = config_data['iter_num']
    root = config_data['data_root']
    overlap = config_data['overlap']

    # load net config
    config_net = config['network']
    base_feature_num = config_net['base_feature_number']
    net_name = config_net['net_type']

    # load train config
    config_train = config['training']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = config_train['maximal_epoch']
    print_step = config_train['print_step']
    best_acc_iter = 0
    cur_accuracy = 0
    best_acc = config_train['best_accuracy']

    # initiate dataset
    # change to my own dataloader
    # set class number = 49
    class_nums = 49
    '''
    train_dataset = NinaProDataset(root=root,
                                   split='train',
                                   #butterWn=wn,
                                   window_length=window_length,
                                   random_sample=iter_num,
                                   overlap=overlap,
                                   transform=tt.Compose([Normalize(), ToTensor()]))
    valid_dataset = NinaProDataset(root=root,
                                   split='valid',
                                   #butterWn=wn,
                                   window_length=window_length,
                                   transform=tt.Compose([Normalize(), ToTensor()]))
    trainLoader = DataLoaderX(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    validLoader = DataLoaderX(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)
    '''

    trainLoader, validLoader = dataloader_yc.main()
    # initiate net
    net_class = NetFactory.create(net_name)
    net = net_class(base_features=base_feature_num, class_num=class_nums)
    net = net.to(device)
    if config_train['load_weight']:
        weight = torch.load(config_train['model_path'], map_location=lambda storage, loc: storage)
        net.load_state_dict(weight)

    # initiate metrics and loss func
    evaluator = AccuracyMetrics(class_nums)
    loss_func = nn.CrossEntropyLoss()
    # loss_func = FocalLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    show_loss = loss_visualize()

    # train begin
    for epoch in range(epochs):
        train_batch_loss = 0
        evaluator.clean()
        net.train()
        # print('1')
        for i, sample in enumerate(trainLoader):
            
            data, label = sample[0].to(device), sample[1].to(device)
            prediction = net(data)
            evaluator.get_data(prediction, label.squeeze(1).squeeze(1))
            train_loss = loss_func(prediction, label.squeeze(1).squeeze(1))
            train_batch_loss += train_loss
            # train_loss = torch.abs(train_loss - 0.4) + 0.4  # trick: flood loss
            optimizer.zero_grad()  # 梯度归零
            train_loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            if i % print_step == 0:
                print('\tbatch id: {} train_loss: {}'.format(i, train_loss.cpu().detach().numpy()))
        train_batch_loss = train_batch_loss.cpu().detach().numpy() / (i + 1)
        train_accuracy, train_class_accuracy = evaluator.evaluate()
        print('{} epoch: {} train batch loss: {} train accuracy: {}'.
              format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                     epoch, train_batch_loss, train_accuracy))

        # valid
        with torch.no_grad():
            net.eval()
            valid_batch_loss = 0
            evaluator.clean()
            for i, sample in enumerate(validLoader):
                data, label = sample['data'].to(device), sample['label'].to(device)
                prediction = net(data)
                evaluator.get_data(prediction, label.squeeze(1).squeeze(1))
                valid_loss = loss_func(prediction, label.squeeze(1).squeeze(1))
                valid_batch_loss += valid_loss
        valid_batch_loss = valid_batch_loss.cpu().detach().numpy() / (i + 1)
        valid_accuracy, valid_class_accuracy = evaluator.evaluate()
        print('{} epoch: {} valid batch loss: {} valid accuracy: {}'.
              format(time.strftime("%Y-%m-%d %H:%M:%S"), epoch, valid_batch_loss, valid_accuracy))

        # plot on visdom
        epoch_metrics = {'train': train_batch_loss, 'valid': valid_batch_loss,
                         'train_accuracy': train_accuracy, 'valid_accuracy': valid_accuracy}
        show_loss.plot_loss(epoch, epoch_metrics)

        # plot final class accuracy
        if epoch == epochs - 1:
            plot_acc(train_class_accuracy, valid_class_accuracy, net_name, batch_size)

        # 模型保存
        '当前批次模型储存'
        if os.path.exists(config_train['model_save_prefix'] + "_cur_{0:}.pkl".format(cur_accuracy)):
            os.remove(config_train['model_save_prefix'] + "_cur_{0:}.pkl".format(cur_accuracy))
        cur_accuracy = valid_accuracy
        torch.save(net.state_dict(), config_train['model_save_prefix'] + "_cur_{0:}.pkl".format(cur_accuracy))
        '判断是否高于之前最优dice'
        if valid_accuracy > best_acc:
            if best_acc_iter > 0:
                os.remove(config_train['model_save_prefix'] + "_{0:}.pkl".format(best_acc))
            best_acc = valid_accuracy
            torch.save(net.state_dict(), config_train['model_save_prefix'] + "_{0:}.pkl".format(best_acc))
            best_acc_iter += 1


if __name__ == '__main__':
    config_file = 'config/train_dl.txt'
    cfg = parse_config(config_file)
    train(config=cfg)
