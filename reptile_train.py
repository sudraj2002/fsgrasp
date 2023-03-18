#!/usr/bin/env python3

import random
import copy

import argparse
import logging
from pathlib import Path

import wandb
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data_loading import BasicDataset
from utils.dice_score import MainLoss
from utils import metrics
from utils.utils import split_train_test_tasks
from unet import UNet
from variables import *
import os
import warnings
from utils.utils import save_images


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--iterations', '-e', metavar='E', type=int, default=50000, help='Number of iterations')
    parser.add_argument('--load', '-f', type=str, default=None, help='Load model from a .pth file')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of output channels')
    parser.add_argument('--dir_checkpoint', '-d', type=str, default=None, help='Checkpoint saving directory')
    parser.add_argument('--test', '-t', type=bool, default=False, help='Whether to test the network')
    parser.add_argument('--depth', '-z', type=bool, default=False, help='Whether to train with depth maps')
    parser.add_argument('--aug', '-a', type=bool, default=False, help='Whether to use augmentation')
    parser.add_argument('--image_size', '-i', type=int, default=182, help='Image size')
    parser.add_argument('--save_tr', type=bool, default=False, help='Whether to save train inputs')
    parser.add_argument('--loss', type=str, default='Dice_BCE', help='Loss function')
    parser.add_argument('--loss_weight', type=float, default=None, help='Loss weight')
    parser.add_argument('--data_dir', type=str, default="../FSSD", help='Train dataset')
    parser.add_argument('--ways', type=int, default=1, help='Number of ways (few-shot)')
    parser.add_argument('--train_shots', type=int, default=5, help='Number of train shots')
    parser.add_argument('--test_shots', type=int, default=5, help='Number of test shots')
    parser.add_argument('--meta_lr_initial', type=int, default=1.0, help='Initial meta learning rate')
    parser.add_argument('--meta_lr_final', type=int, default=0.001, help='Final meta learning rate')
    parser.add_argument('--fast_lr', type=int, default=3e-4, help='Adaptation learning rate')
    parser.add_argument('--meta_train_bsz', type=int, default=5, help='Meta batch size for training')
    parser.add_argument('--meta_test_bsz', type=int, default=1, help='Meta batch size for testing')
    parser.add_argument('--train_inner_bsz', type=int, default=5, help='Adaptation batch size for train')
    parser.add_argument('--test_inner_bsz', type=int, default=5, help='Adaptation batch size for test')
    parser.add_argument('--train_steps', type=int, default=5, help='Number of train steps')
    parser.add_argument('--test_steps', type=int, default=12, help='Number of test steps')
    parser.add_argument('--test_interval', type=int, default=1000, help='Test interval')
    parser.add_argument('--cuda', type=int, default=1, help='Use cuda')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument('--num_test_tasks', type=int, default=50, help='Number of test tasks')
    parser.add_argument('--num_val_tasks', type=int, default=50, help='Number of validation tasks')
    parser.add_argument('--save_interval', type=int, default=1000, help='Checkpoint save interval')
    parser.add_argument('--save_checkpoint', type=bool, default=True, help='Whether to save checkpoints')
    parser.add_argument('--img_scale', type=float, default=1.0, help='Image resize scale')
    return parser.parse_args()


def fast_adapt(batch, learner, adapt_opt, loss, adaptation_steps, shots, ways, batch_size, device,
               loss_type, iteration=0, st=False, return_preds=False):
    data, labels = batch["taskimage"], batch["taskmask"]
    data, labels = torch.squeeze(data, axis=0), torch.squeeze(labels, axis=0)
    if loss_type == 'Dice_BCE' or loss_type == 'BCE' or loss_type == 'Dice':
        data, labels = data.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
    elif loss_type == 'Dice_CE' or loss_type == 'CE':
        data, labels = data.to(device, dtype=torch.float32), labels.to(device, dtype=torch.long)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots * ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    if adaptation_data.size(0) < batch_size:
        warnings.warn('Batch size higher than available tasks {}'.format(adaptation_data.size(0)))
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
    # Adapt the model
    for step in range(adaptation_steps):
        idx = torch.randint(
            adaptation_data.size(0),
            size=(batch_size,)
        )
        adapt_X = adaptation_data[idx]
        adapt_y = adaptation_labels[idx]
        adapt_opt.zero_grad()
        prediction_train = learner(adapt_X)
        error = loss(prediction_train, adapt_y)
        error.backward()
        adapt_opt.step()
    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    if st is True:
        save_images(evaluation_data, evaluation_labels, predictions, mode=loss_type, extra=iteration, st=st)
    valid_error = loss(predictions, evaluation_labels)
    valid_error /= len(evaluation_data)
    mIoU = metrics.compute_iou(predictions, evaluation_labels, mode=loss_type)
    valid_accuracy = mIoU
    if return_preds:
        return valid_error, valid_accuracy, [evaluation_data, evaluation_labels, predictions]
    return valid_error, valid_accuracy


def main():
    dir_checkpoint = args.dir_checkpoint
    test = args.test
    depth = args.depth
    aug = args.aug
    im_size = args.image_size
    loss_type = args.loss
    loss_weight = args.loss_weight
    st = args.save_tr
    data_dir = args.data_dir
    iterations = args.iterations
    ways = args.ways
    train_shots = args.train_shots
    test_shots = args.test_shots
    meta_lr_initial = args.meta_lr_initial
    meta_lr_final = args.meta_lr_final
    fast_lr = args.fast_lr
    meta_train_bsz = args.meta_train_bsz
    train_inner_bsz = args.train_inner_bsz
    test_inner_bsz = args.test_inner_bsz
    train_steps = args.train_steps
    test_steps = args.test_steps
    test_interval = args.test_interval
    cuda = args.cuda
    seed = args.seed
    num_test_tasks = args.num_test_tasks
    num_val_tasks = args.num_val_tasks
    save_interval = args.save_interval
    save_checkpoint = args.save_checkpoint
    amp = args.amp
    img_scale = args.img_scale

    if loss_weight is not None:
        if loss_type == 'Dice_BCE' or loss_type == 'BCE':
            loss_weight = torch.tensor([loss_weight], dtype=torch.float32).cuda()
        elif loss_type == 'Dice_CE' or loss_type == 'CE':
            loss_weight = torch.tensor([loss_weight, 1], dtype=torch.float32).cuda()

    cuda = bool(cuda)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')
    if args.test:
        assert args.load is not None

    # Create model
    if depth:
        num_channels = 1
    else:
        num_channels = 3
    model = UNet(n_channels=num_channels, n_classes=args.classes, bilinear=args.bilinear)
    model.to(device)

    opt = optim.SGD(model.parameters(), meta_lr_initial, weight_decay=1e-7)
    adapt_opt = torch.optim.Adam(model.parameters(), lr=fast_lr, betas=(0, 0.999), weight_decay=1e-7)

    contd = 0
    best_acc = 0.0

    if args.load:
        info = torch.load(args.load, map_location=device)
        m_dict = info["model_state_dict"]
        opt_dict = info["opt_state_dict"]
        adapt_opt_dict = info["adapt_opt_state_dict"]
        contd = info["iteration"]
        best_acc = info["acc"]
        t_names = info["t_names"]
        v_names = info["v_names"]
        model.load_state_dict(m_dict)
        adapt_opt.load_state_dict(adapt_opt_dict)
        opt.load_state_dict(opt_dict)
        logging.info(f'Model loaded from {args.load}')
        print(f"Model loaded from {args.load}")
        print("Iteration: {}".format(contd))
        print("Model accuracy: {}".format(best_acc))
        del info
        del m_dict
        del opt_dict
        del adapt_opt_dict

    adapt_opt_state = adapt_opt.state_dict()
    loss = MainLoss(loss_type=loss_type, loss_weight=loss_weight)

    # Split the dataset into train, val and test tasks
    tasks = []
    for task in os.listdir(data_dir):
        tasks.append(task)

    if test:
        train_names, test_names = split_train_test_tasks(tasks, num_test_tasks)
        test_dataset = BasicDataset(test_names, data_dir=data_dir, image_size=im_size, scale=1.0, use_depth=depth,
                                    use_aug=False, mode=loss_type)
        n_test = int(len(test_dataset))
    else:
        train_names, val_names = split_train_test_tasks(tasks, num_val_tasks)
        if args.load:
            train_names, val_names = t_names, v_names
        if aug:
            train_dataset = BasicDataset(train_names, data_dir=data_dir, image_size=im_size, scale=1.0, use_depth=depth,
                                         use_aug=True, mode=loss_type)
        else:
            train_dataset = BasicDataset(train_names, data_dir=data_dir, image_size=im_size, scale=1.0, use_depth=depth,
                                         use_aug=False, mode=loss_type)
        val_dataset = BasicDataset(val_names, data_dir=data_dir, image_size=im_size, scale=1.0, use_depth=depth,
                                   use_aug=False, mode=loss_type)
        n_test = 0

    n_train = len(train_dataset)
    n_val = len(val_dataset)
    val_percent = n_val * 100 / (n_test + n_train + n_val)

    # 3. Create data loaders
    train_loader_args = dict(batch_size=1, num_workers=4, pin_memory=True)
    test_loader_args = dict(batch_size=1, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **train_loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **test_loader_args)
    if test:
        test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **test_loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow')
    experiment.config.update(dict(epochs=iterations, batch_size=meta_train_bsz, learning_rate=meta_lr_initial,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Iterations:      {iterations}
        Batch size:      {meta_train_bsz}
        Learning rate (adapt):   {fast_lr}
        Step size:   {meta_lr_initial}
        Training size:   {n_train}
        Evaluation size: {n_test}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    train_inner_errors = []
    train_inner_accuracies = []
    valid_inner_errors = []
    valid_inner_accuracies = []
    global_step = 0

    tr_loader = iter(train_loader)
    v_loader = iter(val_loader)
    with tqdm(total=iterations, leave=True, position=0, desc=f"Training Progress", initial=contd,
              ascii=True) as pbar:
        for iteration in range(contd, iterations):
            opt.zero_grad()
            meta_train_error = 0.0
            meta_train_accuracy = 0.0
            meta_valid_error = 0.0
            meta_valid_accuracy = 0.0

            if 8000 < iteration <= 15000:
                for pg in opt.param_groups:
                    pg['lr'] = meta_lr_initial / 10
            elif 15000 < iteration <= 25000:
                for pg in opt.param_groups:
                    pg['lr'] = meta_lr_initial / 100
            elif iteration > 25000:
                for pg in opt.param_groups:
                    pg['lr'] = meta_lr_final

            # zero-grad the parameters
            for p in model.parameters():
                p.grad = torch.zeros_like(p.data)

            for task in range(meta_train_bsz):
                # Compute meta-training loss
                learner = copy.deepcopy(model)
                adapt_opt = torch.optim.Adam(
                    learner.parameters(),
                    lr=fast_lr,
                    betas=(0, 0.999)
                )

                adapt_opt.load_state_dict(adapt_opt_state)
                try:
                    batch = next(tr_loader)
                except StopIteration:
                    tr_loader = iter(train_loader)
                    batch = next(tr_loader)
                evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                                   learner,
                                                                   adapt_opt,
                                                                   loss,
                                                                   train_steps,
                                                                   train_shots,
                                                                   ways,
                                                                   train_inner_bsz,
                                                                   device, st=st, loss_type=loss_type,
                                                                   iteration=iteration)
                adapt_opt_state = adapt_opt.state_dict()
                for p, l in zip(model.parameters(), learner.parameters()):
                    p.grad.data.add_(-1.0, l.data)

                meta_train_error += evaluation_error.item()
                meta_train_accuracy += evaluation_accuracy.item()
            if iteration % test_interval == 0:
                # Compute meta-validation loss
                for k in range(len(val_loader)):
                    learner = copy.deepcopy(model)
                    adapt_opt = torch.optim.Adam(
                        learner.parameters(),
                        lr=fast_lr,
                        betas=(0, 0.999)
                    )
                    adapt_opt.load_state_dict(adapt_opt_state)
                    try:
                        batch = next(v_loader)
                    except StopIteration:
                        v_loader = iter(val_loader)
                        batch = next(v_loader)
                    evaluation_error, evaluation_accuracy, prediction_op = fast_adapt(batch,
                                                                                      learner,
                                                                                      adapt_opt,
                                                                                      loss,
                                                                                      test_steps,
                                                                                      test_shots,
                                                                                      ways,
                                                                                      test_inner_bsz,
                                                                                      device, st=st,
                                                                                      loss_type=loss_type,
                                                                                      return_preds=True)
                    meta_valid_error += evaluation_error.item()
                    meta_valid_accuracy += evaluation_accuracy.item()

                    ims = prediction_op[0]
                    labels = prediction_op[1]
                    predicts = prediction_op[2]
                    save_images(ims, labels, predicts, mode=loss_type, extra=k)

                    if loss_type == 'Dice_BCE' or loss_type == 'BCE' or loss_type == 'Dice':
                        predicts[predicts > 0.5] = 1
                        predicts[predicts <= 0.5] = 0
                        labels = labels.unsqueeze(1)
                    elif loss_type == 'Dice_CE' or loss_type == 'CE':
                        predicts = predicts.argmax(dim=1).unsqueeze(dim=1)
                        labels = labels.unsqueeze(1)

            # Print some metrics
            pbar.update(1)
            new_lr = opt.param_groups[0]['lr']
            if iteration % test_interval == 0:
                pbar.set_postfix(**{'Train Error': meta_train_error / meta_train_bsz,
                                    'Train Acc': meta_train_accuracy / meta_train_bsz,
                                    'Meta LR': new_lr, 'Valid Error': meta_valid_error / n_val,
                                    'Valid Acc': meta_valid_accuracy / n_val})
            else:
                pbar.set_postfix(**{'Train Error': meta_train_error / meta_train_bsz,
                                    'Train Acc': meta_train_accuracy / meta_train_bsz,
                                    'Meta LR': new_lr})
            global_step += 1

            # Track quantities
            train_inner_errors.append(meta_train_error / meta_train_bsz)
            train_inner_accuracies.append(meta_train_accuracy / meta_train_bsz)
            if iteration % test_interval == 0:
                valid_inner_errors.append(meta_valid_error / n_val)
                valid_inner_accuracies.append(meta_valid_accuracy / n_val)

            # Average the accumulated gradients and optimize
            for p in model.parameters():
                p.grad.data.mul_(1.0 / meta_train_bsz).add_(p.data)
            opt.step()

            histograms = {}
            for tag, value in model.named_parameters():
                tag = tag.replace('/', '.')
                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

            val_score = meta_valid_accuracy / n_val
            logging.info('Validation Accuracy: {}'.format(val_score))
            if iteration % test_interval == 0:
                experiment.log({
                    'Train loss': meta_train_error / meta_train_bsz,
                    'Meta learning rate': opt.param_groups[0]['lr'],
                    'Adaptation learning rate': adapt_opt.param_groups[0]['lr'],
                    'Validation accuracy': val_score,
                    'Images': wandb.Image(ims),
                    'Masks': {
                        'true': wandb.Image(labels.type(torch.float)),
                        'pred': wandb.Image(predicts.type(torch.float)),
                    },
                    'Iteration': iteration,
                    **histograms
                })
            else:
                experiment.log({
                    'Train loss': meta_train_error / meta_train_bsz,
                    'Meta learning rate': opt.param_groups[0]['lr'],
                    'Adaptation learning rate': adapt_opt.param_groups[0]['lr'],
                    'Validation accuracy': val_score,
                    'Iteration': iteration,
                    **histograms})

            if iteration % save_interval == 0:
                if save_checkpoint:
                    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                    torch.save({
                        'iteration': iteration,
                        'model_state_dict': model.state_dict(),
                        'opt_state_dict': opt.state_dict(),
                        'adapt_opt_state_dict': adapt_opt.state_dict(),
                        'acc': val_score,
                        't_names': train_names,
                        'v_names': val_names
                    }, str(dir_checkpoint + 'checkpoint_epoch{}.pth'.format(iteration)))
                    logging.info(f'Checkpoint {iteration} saved!')

            if save_checkpoint:
                if val_score > best_acc:
                    best_acc = val_score
                    torch.save({
                        'iteration': iteration,
                        'model_state_dict': model.state_dict(),
                        'opt_state_dict': opt.state_dict(),
                        'adapt_opt_state_dict': adapt_opt.state_dict(),
                        'acc': val_score,
                        't_names': train_names,
                        'v_names': val_names
                    }, str(dir_checkpoint + 'best_checkpoint_epoch.pth'))
        pbar.close()


if __name__ == '__main__':
    args = get_args()
    main()
