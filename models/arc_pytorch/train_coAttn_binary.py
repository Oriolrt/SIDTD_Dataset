import os
import argparse
import pandas as pd
import torch
from torch.autograd import Variable
from datetime import datetime, timedelta
from sklearn import metrics
from sklearn.utils import shuffle
import batcher_kfold_binary as batcher
from batcher_kfold_binary import Batcher
from utils import *

import models_binary
from models_binary import ArcBinaryClassifier, CustomResNet50, CoAttn


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument("--b_low", default = 0, type=int, help="lowest k fold iteration limit")
parser.add_argument("--b_high", default = 10, type=int, help="highest k fold iteration limit")
parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to ARC')
parser.add_argument("--save_model_path", default = os.getcwd() + '/trained_models', type=str, help="Path where you wish to store the trained models")
parser.add_argument("--npy_dataset_path", default = os.getcwd() + '/omniglot/', type=str, help="Path where are located the image arrays for each label and partition")
parser.add_argument("--csv_dataset_path", default = os.getcwd() + '/split_kfold/', type=str, help="Path where are located the image paths for each partition")
parser.add_argument("--dataset", default = 'dataset_raw', type=str, help='Name of the dataset to use. Must be the exact same name as the dataset directory name')
parser.add_argument("--save_results", default = True, type=bool, help="Save results performance in csv or not.")
parser.add_argument('--glimpseSize', type=int, default=8, help='the height / width of glimpse seen by ARC')
parser.add_argument('--numStates', type=int, default=128, help='number of hidden states in ARC controller')
parser.add_argument('--numGlimpses', type=int, default=6, help='the number glimpses of each image in pair seen by ARC')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--cuda', default='True', help='enables cuda')
parser.add_argument('--name', default='coatten_training', help='Custom name for this configuration. Needed for saving'
                                                 ' model score in a separate folder.')
parser.add_argument('--load', default=None, help='the model to load from. Start fresh if not specified.')

parser.add_argument('--n_its', type=int, default=5000, help='number of iterations for training')
parser.add_argument('--apply_fcn', type=bool, default=True, help='apply a resnet to input prior to ARC')
parser.add_argument('--use_coAttn', type=bool, default=True, help='apply the coattention mechanism before discrimination')
opt = parser.parse_args()

def seed_torch(seed=777):
    """
    Seeds the random variables of the different libraries.

    Parameters
    ----------
    seed : int, optional
        Random seed. The default is 777.

    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 


def get_pct_accuracy(pred: Variable, target) -> int:
    hard_pred = (pred > 0.5).int()
    correct = (hard_pred == target).sum().item()
    accuracy = float(correct) / target.size()[0]
    accuracy = int(accuracy * 100)
    return accuracy

def one_shot_eval(pred, truth): 
    pred = pred.round()
    corrects = (pred == truth).sum().item()
    return corrects 


def train(iteration):

    #Define the ResNet50 NN
    if opt.apply_fcn:
        print('Use ResNet50')
        resNet = CustomResNet50()
        
    #Define the CoAttn
    if opt.use_coAttn:
        print('Use Co Attention Model')
        coAtten = CoAttn()
             
    # initialise the model
    discriminator = ArcBinaryClassifier(num_glimpses=opt.numGlimpses,
                                        glimpse_h=opt.glimpseSize,
                                        glimpse_w=opt.glimpseSize,
                                        channels = 1024,
                                        controller_out=opt.numStates)

    if opt.cuda:
        print('Use GPU')
        discriminator.cuda()
        if opt.apply_fcn:
            resNet.cuda()
        if opt.use_coAttn:
            coAtten.cuda()

    # load from a previous checkpoint, if specified.
    if opt.load is not None:
        discriminator.load_state_dict(torch.load(opt.save_model_path + '/{}_{}_best_accuracy.pth'.format(opt.dataset, opt.name)))

    # set up the optimizer.
    bce = torch.nn.BCELoss()
    if opt.cuda:
        bce = bce.cuda()
    #ce_loss = torch.nn.CrossEntropyLoss()

    optim_params = []
    optim_params.append(list(discriminator.parameters()))
    
    if opt.apply_fcn:
        optim_params.append(list(resNet.parameters()))
    if opt.use_coAttn:
        optim_params.append(list(coAtten.parameters()))
        
    flat_params = [item for sublist in optim_params for item in sublist]
    
    optimizer = torch.optim.Adam(params=flat_params, lr=opt.lr)

    # load the dataset in memory.
    paths_splits = {'train':{}, 'val' :{}, 'test': {}}
    n_val = 0
    for d_set in ['train', 'val','test']:
        for key in ['reals','fakes']:
            path = opt.npy_dataset_path +  opt.dataset + '/'+ d_set +'_split_' + key  + '_it_' + str(iteration) + '.npy'
            data = np.load(path)
            if d_set == 'val':
                n_val = n_val + len(data)
            paths_splits[d_set][key] = list(data)
    loader = Batcher(paths_splits= paths_splits, batch_size=opt.batchSize, image_size=opt.imageSize)

    # ready to train ...
    best_validation_loss = None
    best_validation_auc = None
    best_validation_acc = None
    
    saving_threshold = 1.02
    
    training_iteration_list = []
    validation_loss_list = []
    training_loss_list = []
    
    validation_acc_list = []
    training_acc_list = []
    window = opt.batchSize
    
    for training_iteration in range(0,opt.n_its):
        
        discriminator.train(mode=True)
        

        X, Y = loader.fetch_batch("train", batch_size=opt.batchSize)
        if opt.cuda:
            X = X.cuda()
            Y = Y.cuda()
        B,P,C,W,H=X.size()
        
        if opt.apply_fcn:
            resNet.train()
            X = resNet(X.view(B*P,C,W,H))
            _,C,W,H = X.size()
            X =X.view(B,P,C,W,H)
            
        # CoAttention Module 
        if opt.use_coAttn:
            coAtten.train()
            X = coAtten(X)
            
        pred = discriminator(X)
        loss = bce(pred, Y.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if training_iteration % 50 == 0:
        
            discriminator.eval()
            if opt.apply_fcn:
                resNet.eval()
            if opt.use_coAttn:
                coAtten.eval()
            
            # validate your model
            acc_val = 0
            loss_val = 0
            auc_val = 0
            nloop = n_val // window
            for i in range(nloop):
                X_val, Y_val = loader.fetch_batch(part = "val", batch_size = opt.batchSize)
                if opt.cuda:
                    X_val = X_val.cuda()
                    Y_val = Y_val.cuda()
                
                B,P,C,W,H=X_val.size()
                
                # if we apply the FCN
                if opt.apply_fcn:
                    with torch.no_grad():
                        X_val = resNet(X_val.view(B*P,C,W,H))
                    _,C,W,H = X_val.size()
                    X_val = X_val.view(B,P,C,W,H)

                # CoAttention Module 
                if opt.use_coAttn:
                    with torch.no_grad():
                        X_val = coAtten(X_val)   
                

                with torch.no_grad():
                    pred_val = discriminator(X_val)
                
                pred_val = torch.reshape(pred_val, (-1,))
                Y_val = torch.reshape(Y_val, (-1,))
                
                acc = one_shot_eval(pred_val.cpu().detach().numpy(), Y_val.cpu().detach().numpy())
                auc = sklearn.metrics.roc_auc_score(Y_val.cpu().detach().numpy(), pred_val.cpu().detach().numpy())
                
                auc_val = auc_val + auc
                acc_val = acc_val + (acc/window)
                loss_val = loss_val + bce(pred_val, Y_val.float())
            
            acc_val = (acc_val / nloop)*100
            loss_val = loss_val / nloop
            validation_auc = auc_val / nloop


            training_loss = loss.item()
            validation_loss = loss_val.item()
            
            training_iteration_list = training_iteration_list + [training_iteration]
            validation_loss_list = validation_loss_list + [validation_loss]
            training_loss_list = training_loss_list + [training_loss]
                            
            training_acc = get_pct_accuracy(pred, Y)
            
            training_acc_list = training_acc_list + [training_acc]
            validation_acc_list = validation_acc_list + [acc_val]
            plot_loss(opt, training_loss_list, training_acc_list, validation_loss_list, validation_acc_list, training_iteration_list, iteration)
            print("kfold number {} Iteration: {} \t Train: Acc={}%, Loss={} \t\t Validation: Acc={}%, Loss={}, ROC AUC={}".format(
                iteration, training_iteration, training_acc, training_loss, acc_val, validation_loss, validation_auc))

            if best_validation_loss is None:
                best_validation_loss = validation_loss
                
            if best_validation_auc is None:
                best_validation_auc = validation_auc
            
            if best_validation_acc is None:
                best_validation_acc = acc_val

            if not os.path.exists(opt.save_model_path + '/{}/'.format(opt.dataset) ):
                os.makedirs(opt.save_model_path + '/{}/'.format(opt.dataset))

            if (best_validation_auc*saving_threshold) < validation_auc:
                print("Significantly improved validation ROC AUC from {} --> {}. Saving...".format(
                    best_validation_auc, validation_auc
                ))
                torch.save(discriminator.state_dict(),opt.save_model_path + '/{}/{}_best_auc_n{}.pth'.format(opt.dataset, opt.name,iteration))
                if opt.apply_fcn :
                    torch.save(resNet.state_dict(),opt.save_model_path + '/{}/{}_fcn_best_auc_n{}.pth'.format(opt.dataset, opt.name, iteration))
                if opt.use_coAttn :
                    torch.save(coAtten.state_dict(),opt.save_model_path + '/{}/{}_coatten_best_auc_n{}.pth'.format(opt.dataset, opt.name, iteration))
                best_validation_auc = validation_auc
                
            if best_validation_loss > (saving_threshold * validation_loss):
                print("Significantly improved validation loss from {} --> {}. Saving...".format(
                    best_validation_loss, validation_loss
                ))
                torch.save(discriminator.state_dict(),opt.save_model_path + '/{}/{}_best_loss_n{}.pth'.format(opt.dataset, opt.name, iteration))
                if opt.apply_fcn :
                    torch.save(resNet.state_dict(),opt.save_model_path + '/{}/{}_fcn_best_loss_n{}.pth'.format(opt.dataset, opt.name, iteration))
                if opt.use_coAttn :
                    torch.save(coAtten.state_dict(),opt.save_model_path + '/{}/{}_coatten_best_loss_n{}.pth'.format(opt.dataset, opt.name, iteration))
                best_validation_loss = validation_loss
                
            if acc_val > (saving_threshold * best_validation_acc):
                print("Significantly improved validation accuracy from {} --> {}. Saving...".format(
                    best_validation_acc, acc_val
                ))
                torch.save(discriminator.state_dict(),opt.save_model_path + '/{}/{}_best_accuracy_n{}.pth'.format(opt.dataset, opt.name, iteration))
                if opt.apply_fcn :
                    torch.save(resNet.state_dict(),opt.save_model_path + '/{}/{}_fcn_best_accuracy_n{}.pth'.format(opt.dataset, opt.name, iteration))
                if opt.use_coAttn :
                    torch.save(coAtten.state_dict(),opt.save_model_path + '/{}/{}_coatten_best_accuracy_n{}.pth'.format(opt.dataset, opt.name, iteration))
                best_validation_acc = acc_val


    # Test model
    discriminator.load_state_dict(torch.load(opt.save_model_path + '/{}/{}_best_accuracy_n{}.pth'.format(opt.dataset, opt.name, iteration)))
    discriminator.eval()
    if opt.apply_fcn:
        resNet.load_state_dict(torch.load(opt.save_model_path + '/{}/{}_fcn_best_accuracy_n{}.pth'.format(opt.dataset, opt.name, iteration)))
        resNet.eval()
    if opt.use_coAttn:
        coAtten.load_state_dict(torch.load(opt.save_model_path + '/{}/{}_coatten_best_accuracy_n{}.pth'.format(opt.dataset, opt.name, iteration)))
        coAtten.eval()
    path_test = opt.csv_dataset_path + opt.dataset + '/test_split_' +  opt.dataset + '_it_' + str(iteration) + '.csv'
    df_test = pd.read_csv(path_test)
    image_paths = df_test.image_path.values
    label_name = df_test.label_name.values
    acc_test = 0
    loss_test = 0
    auc_test = 0
    i = 0
    while window*(i+1) < len(df_test):
        row0 = label_name[window*i:window*(i+1)]
        row2 = image_paths[window*i:window*(i+1)]
        i = i + 1            
        X_test, Y_test = loader.fetch_batch(part = "test", labels = row0, image_paths = row2, batch_size = opt.batchSize)
        if opt.cuda:
            X_test = X_test.cuda()
            Y_test = Y_test.cuda()
        
        if len(X_test.size())<5:
            X_test = X_test.unsqueeze(2)
            
        if opt.apply_fcn:
            if X_test.size()[2] == 1:
                # since the omiglot data is grayscale we need to transform it to 3 channels in order to fit through resnet
                X_test = X_test.repeat(1,1,3,1,1)
            B,P,C,W,H = X_test.size()

            with torch.no_grad():
                X_test = resNet(X_test.view(B*P,C,W,H))
            _, C, W, H = X_test.size()
            X_test = X_test.view(B, P, C, W, H)
            
        if opt.use_coAttn:
            with torch.no_grad():
                X_test = coAtten(X_test)
        
        with torch.no_grad():
            pred_test = discriminator(X_test)
        
        pred_test = torch.reshape(pred_test, (-1,))
        Y_test = torch.reshape(Y_test, (-1,))
        
        acc = one_shot_eval(pred_test.cpu().detach().numpy(), Y_test.cpu().detach().numpy())
        auc = sklearn.metrics.roc_auc_score(Y_test.cpu().detach().numpy(), pred_test.cpu().detach().numpy())
        
        auc_test = auc_test + auc
        acc_test = acc_test + (acc/window)
        loss_test = loss_test + bce(pred_test, Y_test.float())
        
    X_test, Y_test = loader.fetch_batch(part = "test", labels = label_name[-window:], image_paths = image_paths[-window:], batch_size = opt.batchSize)
    if opt.cuda:
        X_test = X_test.cuda()
        Y_test = Y_test.cuda()
    
    if len(X_test.size())<5:
        X_test = X_test.unsqueeze(2)
        
    if opt.apply_fcn:
        if X_test.size()[2] == 1:
            # since the omiglot data is grayscale we need to transform it to 3 channels in order to fit through resnet
            X_test = X_test.repeat(1,1,3,1,1)
        B,P,C,W,H = X_test.size()

        with torch.no_grad():
            X_test = resNet(X_test.view(B*P,C,W,H))
        _, C, W, H = X_test.size()
        X_test = X_test.view(B, P, C, W, H)
        
    if opt.use_coAttn:
        with torch.no_grad():
            X_test = coAtten(X_test)
    
    with torch.no_grad():
        pred_test = discriminator(X_test)
    
    pred_test = torch.reshape(pred_test, (-1,))
    Y_test = torch.reshape(Y_test, (-1,))
    
    acc = one_shot_eval(pred_test.cpu().detach().numpy(), Y_test.cpu().detach().numpy())
    auc = sklearn.metrics.roc_auc_score(Y_test.cpu().detach().numpy(), pred_test.cpu().detach().numpy())
    
    auc_test = auc_test + auc
    acc_test = acc_test + (acc/window)
    loss_test = loss_test + bce(pred_test, Y_test.float())

    n_total = i + 1
    acc_test = (acc_test / n_total)*100
    loss_test = loss_test / n_total
    test_auc = auc_test / n_total
    test_loss = loss_test.item()
    
    print('****** TRAINING COMPLETED ******')
    print('Kfold number:',iteration)
    print('Final accuracy:', acc_test, 'test AUC:', test_auc, 'test loss:', test_loss)
    
    return best_validation_loss, best_validation_acc, best_validation_auc, test_loss, acc_test, test_auc

def main() -> None:
    
    if opt.cuda:
        batcher.use_cuda = True
        models_binary.use_cuda = True  
    
    #SEED = 777 
    #seed_torch(SEED)

    if opt.name is None:
        # if no name is given, we generate a name from the parameters.
        # only those parameters are taken, which if changed break torch.load compatibility.
        opt.name = "{}_{}_{}_{}".format(opt.numGlimpses, opt.glimpseSize, opt.numStates)

    print("Will start training {} with parameters:\n{}\n\n".format(opt.name, opt))


    
    #writers to write the results obtained for each split
    f_val, f_test, writer_val, writer_test = save_results_setup(opt)
    
    for iteration in range(opt.b_low, opt.b_high):
        
        best_validation_loss, best_validation_acc, best_validation_auc, test_loss, acc_test, test_auc = train(iteration)
        
        #save results on the output cvs file
        val_res = [iteration, best_validation_loss, best_validation_acc, best_validation_auc]
        writer_val.writerow(val_res)
        test_res = [iteration, test_loss, acc_test, test_auc]
        writer_test.writerow(test_res)

    if opt.save_results:
        f_val.close()
        f_test.close() 


if __name__ == "__main__":
    main()
