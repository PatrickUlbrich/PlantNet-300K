import os
from tqdm import tqdm
import pickle
import argparse
import time
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

from utils import set_seed, load_model, save, get_model, update_optimizer, get_data
from epoch import train_epoch, val_epoch, test_epoch
from cli import add_all_parsers


def train(args):
    set_seed(args, use_gpu=torch.cuda.is_available())
    
    # start of my additional code
    # load number of images per class. This is used in utils.py get_data for the added weighted sampling option
    import pandas as pd

    class_counts_df = pd.read_csv('./results/class_counts.csv')
    class_counts = dict(zip(class_counts_df['class'], class_counts_df['total']))
    # end of my additional code
    
    train_loader, val_loader, test_loader, dataset_attributes = get_data(args.root, args.image_size, args.crop_size,
                                                                         args.batch_size, args.num_workers, args.pretrained,
                                                                         args.threshold, args.weighted_sampler) # my additional code
    
    # start of my additional code
    # load Plantnet models as checkpoint if parameter option is used
    if args.continue_training:
        from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, inception_v3, mobilenet_v2, densenet121, \
        densenet161, densenet169, densenet201, alexnet, squeezenet1_0, shufflenet_v2_x1_0, wide_resnet50_2, wide_resnet101_2,\
        vgg11, mobilenet_v3_large, mobilenet_v3_small

        pytorch_models = {'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50, 'resnet101': resnet101,
                          'resnet152': resnet152, 'densenet121': densenet121, 'densenet161': densenet161,
                          'densenet169': densenet169, 'densenet201': densenet201, 'mobilenet_v2': mobilenet_v2,
                          'inception_v3': inception_v3, 'alexnet': alexnet, 'squeezenet': squeezenet1_0,
                          'shufflenet': shufflenet_v2_x1_0, 'wide_resnet50_2': wide_resnet50_2,
                          'wide_resnet101_2': wide_resnet101_2, 'vgg11': vgg11, 'mobilenet_v3_large': mobilenet_v3_large,
                          'mobilenet_v3_small': mobilenet_v3_small
                          }
        
        checkpoint_path = f"./pre-trained-models/{args.model}_weights_best_acc.tar"
        checkpoint = torch.load(checkpoint_path)

        model = pytorch_models[args.model](pretrained=False, num_classes=1081)
        model.load_state_dict(checkpoint['model'])     
        
    else:
        model = get_model(args, n_classes=dataset_attributes['n_classes'])

    # load other loss function if parameter option is used 
    if args.weighted_crossentropy:
        class_to_idx = dataset_attributes['class_to_idx']
        
        # convert the keys in class_counts to string, to match with class_to_idx
        class_counts = {str(name): count for name, count in class_counts.items()}
        
        weights = [1.0 / class_counts[cls] for cls in sorted(class_to_idx, key=class_to_idx.get)]
        weights_tensor = torch.tensor(weights, dtype=torch.float)
        
        if args.use_gpu:
            weights_tensor = weights_tensor.cuda()
        
        criteria = CrossEntropyLoss(weight=weights_tensor)
    else:
        criteria = CrossEntropyLoss()
    
    val_accuracies = []
    # end of my additional code
    
    if args.use_gpu:
        print('USING GPU')
        torch.cuda.set_device(0)
        model.cuda()
        criteria.cuda()

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.mu, nesterov=True)

    # containers for storing metrics over epochs
    loss_train, acc_train, topk_acc_train = [], [], []
    loss_val, acc_val, topk_acc_val, avgk_acc_val, class_acc_val = [], [], [], [], []

    save_name = args.save_name_xp.strip()
    save_dir = os.path.join(os.getcwd(), 'results', save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print('args.k : ', args.k)

    lmbda_best_acc = None
    best_val_acc = float('-inf')

    
    for epoch in tqdm(range(args.n_epochs), desc='epoch', position=0):
        t = time.time()
        optimizer = update_optimizer(optimizer, lr_schedule=args.epoch_decay, epoch=epoch)

        loss_epoch_train, acc_epoch_train, topk_acc_epoch_train = train_epoch(model, optimizer, train_loader,
                                                                              criteria, loss_train, acc_train,
                                                                              topk_acc_train, args.k,
                                                                              dataset_attributes['n_train'],
                                                                              args.use_gpu)

        loss_epoch_val, acc_epoch_val, topk_acc_epoch_val, \
        avgk_acc_epoch_val, lmbda_val = val_epoch(model, val_loader, criteria,
                                                  loss_val, acc_val, topk_acc_val, avgk_acc_val,
                                                  class_acc_val, args.k, dataset_attributes, args.use_gpu)
        
        # start of my additional code
        # save epoch results in a csv for later visualization
        import csv
        val_accuracies.append((epoch, loss_epoch_train, acc_epoch_train, loss_epoch_val, acc_epoch_val))
        with open('./results/epoch_acc_train_val.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'loss_train', 'acc_train', 'loss_val', 'acc_val']) # header row
            writer.writerows(val_accuracies)
        # end of my additional code
        
        
        # save model at every epoch
        save(model, optimizer, epoch, os.path.join(save_dir, save_name + '_weights.tar'))

        # save model with best val accuracy
        if acc_epoch_val > best_val_acc:
            best_val_acc = acc_epoch_val
            lmbda_best_acc = lmbda_val
            save(model, optimizer, epoch, os.path.join(save_dir, save_name + '_weights_best_acc.tar'))

        print()
        print(f'epoch {epoch} took {time.time()-t:.2f}')
        print(f'loss_train : {loss_epoch_train}')
        print(f'loss_val : {loss_epoch_val}')
        print(f'acc_train : {acc_epoch_train} / topk_acc_train : {topk_acc_epoch_train}')
        print(f'acc_val : {acc_epoch_val} / topk_acc_val : {topk_acc_epoch_val} / '
              f'avgk_acc_val : {avgk_acc_epoch_val}')
        
    # load weights corresponding to best val accuracy and evaluate on test
    load_model(model, os.path.join(save_dir, save_name + '_weights_best_acc.tar'), args.use_gpu)
    loss_test_ba, acc_test_ba, topk_acc_test_ba, \
    avgk_acc_test_ba, class_acc_test = test_epoch(model, test_loader, criteria, args.k,
                                                  lmbda_best_acc, args.use_gpu,
                                                  dataset_attributes)

    # Save the results as a dictionary and save it as a pickle file in desired location
    results = {'loss_train': loss_train, 'acc_train': acc_train, 'topk_acc_train': topk_acc_train,
               'loss_val': loss_val, 'acc_val': acc_val, 'topk_acc_val': topk_acc_val, 'class_acc_val': class_acc_val,
               'avgk_acc_val': avgk_acc_val,
               'test_results': {'loss': loss_test_ba,
                                'accuracy': acc_test_ba,
                                'topk_accuracy': topk_acc_test_ba,
                                'avgk_accuracy': avgk_acc_test_ba,
                                'class_acc_dict': class_acc_test},
               'params': args.__dict__}

    with open(os.path.join(save_dir, save_name + '.pkl'), 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_all_parsers(parser)
    args = parser.parse_args()
    train(args)