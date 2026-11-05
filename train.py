import torch.backends.cudnn as cudnn
import torch.optim as optim
import csv
from glob import glob
import sys,time
from os.path import join
import torch
from lib.losses.loss import *
from lib.common import *
from config import parse_args
from lib.logger import Logger, Print_Logger
import models

from function import get_dataloader, train, val, get_dataloaderV2


def main():
    run_started = time.perf_counter()
    setpu_seed(2021)
    args = parse_args()
    save_path = join(args.outf, args.save)
    save_args(args,save_path)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    cudnn.benchmark = True
    
    log = Logger(save_path)
    sys.stdout = Print_Logger(os.path.join(save_path,'train_log.txt'))
    print('The computing device used is: ','GPU' if device.type=='cuda' else 'CPU')
    
    net = models.UNetFamily.U_Net(1, 2).to(device)
    print("Total number of parameters: " + str(count_parameters(net)))

    log.save_graph(net,torch.randn((1,1,48,48)).to(device))  # Save the model structure to the tensorboard file
    # torch.nn.init.kaiming_normal(net, mode='fan_out')      # Modify default initialization method
    # net.apply(weight_init)

    criterion = CrossEntropyLoss2d() # Initialize loss function
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # The training speed of this task is fast, so pre training is not recommended
    if args.pre_trained is not None:
        checkpoint_path = join(args.outf, args.pre_trained, 'latest_model.pth')
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch']+1

    # criterion = LossMulti(jaccard_weight=0,class_weights=np.array([0.5,0.5]))
    # create a list of learning rate with epochs
    # lr_schedule = make_lr_schedule(np.array([50, args.N_epochs]),np.array([0.001, 0.0001]))
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)
    if args.start_epoch > 1:
        for param_group in optimizer.param_groups:
            param_group.setdefault('initial_lr', args.lr)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.N_epochs, eta_min=0, last_epoch=args.start_epoch - 2)
    else:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.N_epochs, eta_min=0)
    
    train_loader, val_loader = get_dataloaderV2(args) # create dataloader
    # train_loader, val_loader = get_dataloader(args)
    
    best = {'epoch':0,'AUC_roc':0.5} # Initialize the best epoch and performance(AUC of ROC)
    if args.pre_trained is not None:
        previous_rows = []
        for log_path in glob(join(save_path, 'log*.csv')):
            with open(log_path, newline='') as log_file:
                previous_rows.extend(csv.DictReader(log_file))
        if previous_rows:
            best_row = max(previous_rows, key=lambda row: float(row['val_auc_roc']))
            best = {'epoch':int(best_row['epoch']), 'AUC_roc':float(best_row['val_auc_roc'])}
    trigger = 0  # Early stop Counter
    for epoch in range(args.start_epoch,args.N_epochs+1):
        print('\nEPOCH: %d/%d --(learn_rate:%.6f) | Time: %s' % \
            (epoch, args.N_epochs,optimizer.state_dict()['param_groups'][0]['lr'], time.asctime()))
        
        # train stage
        train_log = train(train_loader,net,criterion, optimizer,device) 
        # val stage
        val_log = val(val_loader,net,criterion,device)

        log.update(epoch,train_log,val_log) # Add log information
        lr_scheduler.step()

        # Save checkpoint of latest and best model.
        state = {'net': net.state_dict(),'optimizer':optimizer.state_dict(),'epoch': epoch}
        torch.save(state, join(save_path, 'latest_model.pth'))
        trigger += 1
        if val_log['val_auc_roc'] > best['AUC_roc']:
            print('\033[0;33mSaving best model!\033[0m')
            torch.save(state, join(save_path, 'best_model.pth'))
            best['epoch'] = epoch
            best['AUC_roc'] = val_log['val_auc_roc']
            trigger = 0
        print('Best performance at Epoch: {} | AUC_roc: {}'.format(best['epoch'],best['AUC_roc']))
        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break
        torch.cuda.empty_cache()
    elapsed = time.perf_counter() - run_started
    print('Total training time: {:.2f} seconds'.format(elapsed))
    if device.type == 'cuda':
        peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        print('Peak GPU memory: {:.2f} GiB allocated, {:.2f} GiB reserved'.format(
            peak_allocated, peak_reserved))
    print('Checkpoints and logs saved to: {}'.format(save_path))
if __name__ == '__main__':
    main()
