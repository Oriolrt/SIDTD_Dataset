import matplotlib
matplotlib.use('Agg')
import sys
import os
import torch

from torch.utils.data import DataLoader 
from torch.optim import SGD 

from ._utils import *
from SIDTD.utils.batch_generator import *

import torch.nn as nn

def test(LOGGER, model, device, criterion, test_loader, N_CLASSES, BATCH_SIZE):
    """ Returns inference results on a chosen dataset and a chosen model """
               
    #Evaluation
    model.eval()
    avg_val_loss = 0.
    preds = np.zeros((len(test_loader.dataset)))
    reals = np.zeros((len(test_loader.dataset)))
    p_preds = []
    
    # Iterate over the whole test set batch by batch
    for i, (images, labels) in enumerate(test_loader):
     
        images = images.to(device)
        labels = labels.to(device)
        reals[i * BATCH_SIZE: (i+1) * BATCH_SIZE] = labels.to('cpu').numpy()     
     
        with torch.no_grad():
            y_preds = model(images)
                      
        preds[i * BATCH_SIZE: (i+1) * BATCH_SIZE] = y_preds.argmax(1).to('cpu').numpy()   # Label predicted by the model 

        p_pred = F.softmax(y_preds, dim=1)   # Probability predicted for each label
        p_preds.extend(p_pred[:,1].to('cpu').numpy())   # Probability predicted for fake label

        
        loss = criterion(y_preds, labels)
        avg_val_loss += loss.item() / len(test_loader)
    
    # Compute metrics: F1, accuracy, ROC AUC, FPR, FNR
    score = f1_score(reals, preds, average='macro')
    accuracy = accuracy_score(reals, preds)
    try: 
        roc_auc_score = sklearn.metrics.roc_auc_score(reals, p_preds)
    except:
        roc_auc_score = -1

    FPR, FNR = get_FPR_FNR(actual = reals, pred = preds)
    
    LOGGER.debug(f'TESTING: avg_test_loss: {avg_val_loss:.4f} F1: {score:.6f}  Accuracy: {accuracy:.6f} roc_auc_score: {roc_auc_score:.6f}') 

    return avg_val_loss, accuracy, roc_auc_score, FPR, FNR


           
def test_baseline_models(args, LOGGER, iteration=0):

    """ Load dataset, dataloaders and models parameters for inference"""
    
    if args.device=='cuda':
        # Set all data on GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(torch.cuda.is_available())
        print(device)
    else:
        device = 'cpu'
    print(device)     
    SEED = 777 
    seed_torch(SEED)

    if not os.path.exists(args.results_path + '{}/{}/'.format(args.model, args.dataset)):
        os.makedirs(args.results_path + '{}/{}/'.format(args.model, args.dataset))
    
    # Load existing csv results file if it exist otherwise, create the results file
    if args.save_results:
        if args.inf_domain_change =='yes':
            results_path = args.results_path + '{}/{}/trained_{}_{}_test_results.csv'.format(args.model, args.dataset, args.dataset_source, args.name)
        else:
            results_path = args.results_path + '{}/{}/{}_test_results.csv'.format(args.model, args.dataset, args.name)
        print("Results file: ", results_path)
        if os.path.isfile(results_path):
            f_test = open(results_path, 'a')
            writer_test = csv.writer(f_test)
        else:
            f_test = open(results_path, 'w')
            # create the csv writer
            writer_test = csv.writer(f_test)
            header_test = ['iteration', 'loss', 'accuracy', 'roc_auc_score', 'FPR', 'FNR']
            writer_test.writerow(header_test)

    # Parameters for model inference. They can be chosen with the flags
    BATCH_SIZE = args.batch_size
    WORKERS = args.workers
    lr = args.learning_rate
    
    if args.model in ['vit_large_patch16_224']:
        WIDTH, HEIGHT = 224, 224
    else: 
        WIDTH, HEIGHT = 299, 299

    N_CLASSES = args.nclasses

    print("*****************************************")
    print("Model {} inference on dataset {}".format(args.model,args.dataset))
    print("*****************************************")


    #load model weights
    model = setup_model(args, N_CLASSES)
    mean, std = get_mean_std(args, model)
    print('fold number :', iteration)

    # Load dataset paths that will be used by the batch generator
    # You can use static path csv to replicate results or custom/random partitionning
    # You can use different type of partitionning: train validation test split or kfold cross-validation 
    # You can use different type of data: templates, clips or cropped clips.
    if args.static == 'no':
        if args.type_split == 'kfold':
            if args.inf_domain_change == 'yes':
                if os.path.exists(os.getcwd() + "/split_kfold/{}/test_split_{}_it_0.csv".format(args.dataset, args.dataset)):
                    print("Loading existing partition: ", "test_split_{}_it_0".format(args.dataset))
                    test_metadata_split = pd.read_csv(os.getcwd() + "/split_kfold/{}/test_split_{}_it_0.csv".format(args.dataset, args.dataset))
                else:
                    print('ERROR : WRONG PATH')
            
            else:
                if os.path.exists(os.getcwd() + "/split_kfold/{}/test_split_{}_it_{}.csv".format(args.dataset, args.dataset, iteration)):
                    print("Loading existing partition: ", "split_{}_it_{}".format(args.dataset, iteration))
                    test_metadata_split = pd.read_csv(os.getcwd() + "/split_kfold/{}/test_split_{}_it_{}.csv".format(args.dataset, args.dataset, iteration))
                else:
                    print('ERROR : WRONG PATH')
        elif args.type_split =='cross':
            if os.path.exists(os.getcwd() + "/split_normal/{}/test_split_{}.csv".format(args.dataset, args.dataset)):
                test_metadata_split = pd.read_csv(os.getcwd() + "/split_normal/{}/test_split_{}.csv".format(args.dataset, args.dataset))
            else:
                print('ERROR : WRONG PATH')
    
    else:
        if args.type_split =='kfold':
            if args.type_data =='templates':
                if os.path.exists(os.getcwd() + "/static/split_kfold/test_split_SIDTD_it_{}.csv".format(iteration)):
                    print("Loading existing partition: ", "split_SIDTD_it_{}".format(iteration))
                    test_metadata_split = pd.read_csv(os.getcwd() + "/static/split_kfold/test_split_SIDTD_it_{}.csv".format(iteration)) 
                else:
                    print('ERROR : WRONG PATH')
                
            elif args.type_data == 'clips':
                if os.path.exists(os.getcwd() + "/static/split_kfold_unbalanced/test_split_clip_background_SIDTD_it_{}.csv".format(iteration)):
                    test_metadata_split = pd.read_csv(os.getcwd() + "/static/split_kfold_unbalanced/test_split_clip_background_SIDTD_it_{}.csv".format(iteration))
                else:
                    print('ERROR : WRONG PATH')

            else:
                if os.path.exists(os.getcwd() + "/static/split_kfold_cropped_unbalanced/test_split_clip_cropped_SIDTD_it_{}.csv".format(iteration)):
                    test_metadata_split = pd.read_csv(os.getcwd() + "/static/split_kfold_cropped_unbalanced/test_split_clip_cropped_SIDTD_it_{}.csv".format(iteration))
                else:
                    print('ERROR : WRONG PATH')

        else:
            if args.type_data =='templates':
                if os.path.exists(os.getcwd() + "/static/split_normal/test_split_SIDTD.csv"):
                    test_metadata_split = pd.read_csv(os.getcwd() + "/static/split_normal/test_split_SIDTD.csv")
                else:
                    print('ERROR : WRONG PATH')

            elif args.type_data == 'clips':
                if os.path.exists(os.getcwd() + "/static/cross_val_unbalanced/test_split_SIDTD.csv"):
                    test_metadata_split = pd.read_csv(os.getcwd() + "/static/cross_val_unbalanced/test_split_SIDTD.csv")
                else:
                    print('ERROR : WRONG PATH')

            else:
                if os.path.exists(os.getcwd() + "/static/cross_val_cropped_unbalanced/test_split_clip_cropped_SIDTD.csv"):
                    test_metadata_split = pd.read_csv(os.getcwd() + "/static/cross_val_cropped_unbalanced/test_split_clip_cropped_SIDTD.csv")
                else:
                    print('ERROR : WRONG PATH')
    
    # Load image paths and labels
    test_paths = test_metadata_split['image_path'].values.tolist()
    test_ids = test_metadata_split['label'].values.tolist()

    # Load Batch Generator function
    test_dataset = TrainDataset(test_paths, test_ids, transform=get_transforms(WIDTH, HEIGHT, mean, std, data='valid'))
    #Dataloaders
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)
    
    
    #Evaluation
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    
    # Load trained models.
    # Choose model depending on if you want to choose a custom trained model or perform inference with our models
    if args.pretrained == 'no':
        print('Test with your own trained models.')
        if args.inf_domain_change == 'yes':
            save_model_path = args.save_model_path + args.model + "_trained_models/" + args.dataset_source + "/"
            PATH = save_model_path + '/{}_{}_best_accuracy_n{}.pth'.format(args.dataset_source, args.name, iteration)
        else:
            save_model_path = args.save_model_path + args.model + "_trained_models/" + args.dataset + "/"
            PATH = save_model_path + '/{}_{}_best_accuracy_n{}.pth'.format(args.dataset, args.name, iteration)
    else:
        print('Test with SIDTD trained models.')
        if args.type_data == 'clips':
            if args.model== 'efficientnet-b3':
                PATH = os.getcwd() + "/pretrained_models/unbalanced_clip_cropped_SIDTD/{}_trained_models/clip_background_MIDV2020_EfficientNet_best_accuracy_n{}.pth".format(args.model, iteration)
            elif args.model== 'resnet50':
                PATH = os.getcwd() + "/pretrained_models/unbalanced_clip_cropped_SIDTD/{}_trained_models/clip_background_MIDV2020_ResNet50_best_accuracy_n{}.pth".format(args.model, iteration)
            else:
                PATH = os.getcwd() + "/pretrained_models/unbalanced_clip_cropped_SIDTD/{}_trained_models/clip_background_MIDV2020_vit_large_patch16_best_accuracy_n{}.pth".format(args.model, iteration)
        elif args.type_data == 'clips_cropped':
            if args.model== 'efficientnet-b3':
                PATH = os.getcwd() + "/pretrained_models/unbalanced_clip_cropped_SIDTD/{}_trained_models/clip_cropped_MIDV2020_EfficientNet_best_accuracy_n{}.pth".format(args.model, iteration)
            elif args.model== 'resnet50':
                PATH = os.getcwd() + "/pretrained_models/unbalanced_clip_cropped_SIDTD/{}_trained_models/clip_cropped_MIDV2020_ResNet50_best_accuracy_n{}.pth".format(args.model, iteration)
            else:
                PATH = os.getcwd() + "/pretrained_models/unbalanced_clip_cropped_SIDTD/{}_trained_models/clip_cropped_MIDV2020_vit_large_patch16_best_accuracy_n{}.pth".format(args.model, iteration)
        elif args.type_data == 'templates':
            PATH = os.getcwd() + "/pretrained_models/balanced_templates_SIDTD/{}_trained_models/MIDV2020_{}_best_accuracy_n{}.pth".format(args.model, args.model, iteration)

    
    print('Use trained model saved in:', PATH)
    print("********      Creating csv stat result file      *********")
    model.load_state_dict(torch.load(PATH))
    model.eval()
    
    loss, accuracy, roc_auc_score, FPR, FNR = test(LOGGER, model, device, criterion, test_loader, N_CLASSES, BATCH_SIZE)

    # write and save inference results on test set in csv
    if args.save_results:
        test_res = [iteration, loss, accuracy, roc_auc_score, FPR, FNR]
        
        writer_test.writerow(test_res)
        

    if args.save_results:
        f_test.close()
