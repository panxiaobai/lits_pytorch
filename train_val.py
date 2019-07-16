import LITS_DataSet
import torch
import argparse
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import LITS_reader
import metrics
from torch.utils.data import Dataset, DataLoader
from Unet import UNet,ResBlock,RecombinationBlock
import logger
import init_util


def val(model, val_loader, device, epoch, val_dict, logger):
    model.eval()
    val_loss = 0
    val_dice0 = 0
    val_dice1 = 0
    val_dice2 = 0
    with torch.no_grad():
        for data, target in val_loader:
            data = torch.squeeze(data, dim=0)
            target = torch.squeeze(target, dim=0)
            data, target = data.float(), target.float()
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss = metrics.DiceMeanLoss()(output, target)
            dice0 = metrics.dice(output, target, 0)
            dice1 = metrics.dice(output, target, 1)
            dice2 = metrics.dice(output, target, 2)

            val_loss += float(loss)
            val_dice0 += float(dice0)
            val_dice1 += float(dice1)
            val_dice2 += float(dice2)

    val_loss /= len(val_loader)
    val_dice0 /= len(val_loader)
    val_dice1 /= len(val_loader)
    val_dice2 /= len(val_loader)

    val_dict['loss'].append(float(val_loss))
    val_dict['dice0'].append(float(val_dice0))
    val_dict['dice1'].append(float(val_dice1))
    val_dict['dice2'].append(float(val_dice2))
    logger.scalar_summary('val_loss', val_loss, epoch)
    logger.scalar_summary('val_dice0', val_dice0, epoch)
    logger.scalar_summary('val_dice1', val_dice1, epoch)
    logger.scalar_summary('val_dice2', val_dice2, epoch)
    print('\nVal set: Average loss: {:.6f}, dice0: {:.6f}\tdice1: {:.6f}\tdice2: {:.6f}\t\n'.format(
        val_loss, val_dice0, val_dice1, val_dice2))


def train(model, train_loader, device, optimizer, epoch, train_dict, logger):
    model.train()
    train_loss = 0
    train_dice0 = 0
    train_dice1 = 0
    train_dice2 = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = torch.squeeze(data, dim=0)
        target = torch.squeeze(target, dim=0)
        data, target = data.float(), target.float()
        data, target = data.to(device), target.to(device)
        output = model(data)

        optimizer.zero_grad()

        # loss = nn.CrossEntropyLoss()(output,target)
        # loss=metrics.SoftDiceLoss()(output,target)
        # loss=nn.MSELoss()(output,target)
        loss = metrics.DiceMeanLoss()(output, target)
        # loss=metrics.WeightDiceLoss()(output,target)
        # loss=metrics.CrossEntropy()(output,target)
        loss.backward()
        optimizer.step()

        train_loss = loss
        train_dice0 = metrics.dice(output, target, 0)
        train_dice1 = metrics.dice(output, target, 1)
        train_dice2 = metrics.dice(output, target, 2)
        print(
            'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tdice0: {:.6f}\tdice1: {:.6f}\tdice2: {:.6f}\tT: {:.6f}\tP: {:.6f}\tTP: {:.6f}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), loss.item(),
                train_dice0, train_dice1, train_dice2,
                metrics.T(output, target), metrics.P(output, target), metrics.TP(output, target)))

    train_dict['loss'].append(float(train_loss))
    train_dict['dice0'].append(float(train_dice0))
    train_dict['dice1'].append(float(train_dice1))
    train_dict['dice2'].append(float(train_dice2))

    logger.scalar_summary('train_loss', train_loss, epoch)
    logger.scalar_summary('train_dice0', train_dice0, epoch)
    logger.scalar_summary('train_dice1', train_dice1, epoch)
    logger.scalar_summary('train_dice2', train_dice2, epoch)


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    # torch.cuda.set_device(3)
    parser = argparse.ArgumentParser(description='PyTorch LIST')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    model = UNet(1, [32, 48, 64, 96, 128], 3, net_mode='3d',conv_block=RecombinationBlock).to(device)
    init_util.print_network(model)

    model = nn.DataParallel(model, device_ids=[0, 1])  # multi-GPU

    reader = LITS_reader.LITS_reader(data_fix=False)
    train_set = LITS_DataSet.Lits_DataSet([16, 96, 96], 12, reader,0.5)
    val_set = LITS_DataSet.Lits_DataSet_val([16, 96, 96], 12, reader,0.5)
    train_loader=DataLoader(dataset=train_set,shuffle=True)
    val_loader=DataLoader(dataset=val_set,shuffle=True)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    train_dict = {'loss': [], 'dice0': [], 'dice1': [], 'dice2': []}
    val_dict = {'loss': [], 'dice0': [], 'dice1': [], 'dice2': []}

    logger = logger.Logger('./log')
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch, args)
        train(model, train_loader, device, optimizer, epoch, train_dict, logger)
        val(model, val_loader, device, epoch, val_dict, logger)
        torch.save(model, 'model')

