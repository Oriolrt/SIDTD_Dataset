import matplotlib
matplotlib.use('Agg')

import glob
import torch
from torch.utils.data import DataLoader, Dataset 
from torch.optim import Adam, SGD, AdamW 
import torch.nn as nn 
from torch.optim.lr_scheduler import CosineAnnealingLR

import os


from utils import *

def test(LOGGER, model, device, criterion, test_loader, N_CLASSES, BATCH_SIZE):
               
    #Evaluation
    model.eval()
    avg_val_loss = 0.
    preds = np.zeros((len(test_loader.dataset)))
    reals = np.zeros((len(test_loader.dataset)))
    p_preds = []
    
    for i, (images, labels) in enumerate(test_loader):
     
        images = images.to(device)
        labels = labels.to(device)
        reals[i * BATCH_SIZE: (i+1) * BATCH_SIZE] = labels.to('cpu').numpy()     
     
        with torch.no_grad():
            y_preds = model(images)
                      
        preds[i * BATCH_SIZE: (i+1) * BATCH_SIZE] = y_preds.argmax(1).to('cpu').numpy()

        p_pred = F.softmax(y_preds, dim=1)
        p_preds.extend(p_pred[:,1].to('cpu').numpy())

        
        loss = criterion(y_preds, labels)
        avg_val_loss += loss.item() / len(test_loader)
    
    score = f1_score(reals, preds, average='macro')
    accuracy = accuracy_score(reals, preds)
    try: 
        roc_auc_score = sklearn.metrics.roc_auc_score(reals, p_preds)
    except:
        roc_auc_score = -1

    
    LOGGER.debug(f'TESTING: avg_test_loss: {avg_val_loss:.4f} F1: {score:.6f}  Accuracy: {accuracy:.6f} roc_auc_score: {roc_auc_score:.6f}') 

    return avg_val_loss, accuracy, roc_auc_score

           
def test_baseline_models(args, LOGGER, iteration):
    
    if args.device=='cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(torch.cuda.is_available())
        print(device)
    else:
        device = 'cpu'
    print(device)     
    SEED = 777 
    seed_torch(SEED)

    if not os.path.exists('results_files/{}'.format(args.dataset)):
        os.makedirs('results_files/{}'.format(args.dataset))
    
    if args.save_results:
        print("Results file: ", 'results_files/{}/{}_test_results.csv'.format(args.dataset, args.name))
        if os.path.isfile('results_files/{}/{}_test_results.csv'.format(args.dataset, args.name)):
            f_test = open('results_files/{}/{}_test_results.csv'.format(args.dataset, args.name), 'a')
            writer_test = csv.writer(f_test)
        else:
            f_test = open('results_files/{}/{}_test_results.csv'.format(args.dataset, args.name), 'w')
            # create the csv writer
            writer_test = csv.writer(f_test)
            header_test = ['iteration', 'loss', 'accuracy', 'roc_auc_score']
            writer_test.writerow(header_test)

    # Adjust BATCH_SIZE and ACCUMULATION_STEPS to values that if multiplied results in 64 !
    BATCH_SIZE = args.batch_size
    WORKERS = args.workers
    lr = args.learning_rate
    
    if args.model in ['vit_large_patch16_224', 'pretrained_banknotes_vit_large_patch16_224']:
        WIDTH, HEIGHT = 224, 224
    else: 
        WIDTH, HEIGHT = 299, 299

    N_CLASSES = args.nclasses

    print("*****************************************")
    print("Model {} inference on dataset {}".format(args.model,args.dataset))
    print("*****************************************")


    # Iterate for the K partitions
    model = setup_model(args, N_CLASSES)
    mean, std = get_mean_std(args, model)
    print('fold number :', iteration)

    

    if not os.path.exists("split_kfold/"):
        os.makedirs("split_kfold/")
        
    if os.path.exists("split_kfold/test_split_{}_it_{}.csv".format(args.dataset, iteration)):
        test_metadata_split = pd.read_csv("split_kfold/test_split_{}_it_{}.csv".format(args.dataset, iteration))
        
    print('process split csv file from path :',"split_kfold/test_split_{}_it_{}.csv".format(args.dataset, iteration))
    test_paths = test_metadata_split['image_path'].values.tolist()
    test_ids = test_metadata_split['label'].values.tolist()

    # Custom datasets following pytorch guidelines
    test_dataset = TrainDataset(test_paths, test_ids, transform=get_transforms(WIDTH, HEIGHT, mean, std, data='valid'))
    #Dataloaders
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)
    
    
    #Evaluation
    #inside kfold loop because it has to reset to default before initialising a new training
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=1, verbose=True, eps=1e-6)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    
    save_model_path = args.save_model_path + args.model + "_trained_models/"
    PATH = save_model_path + '/{}_{}_best_accuracy_n{}.pth'.format(args.dataset, args.name, iteration)
    print("********      Creating csv stat result file      *********")
    model.load_state_dict(torch.load(PATH))
    model.eval()
    
    loss, accuracy, roc_auc_score = test(LOGGER, model, device, criterion, test_loader, N_CLASSES, BATCH_SIZE)

    if args.save_results:
        test_res = [iteration, loss, accuracy, roc_auc_score]
        
        writer_test.writerow(test_res)
        

    if args.save_results:
        f_test.close()
