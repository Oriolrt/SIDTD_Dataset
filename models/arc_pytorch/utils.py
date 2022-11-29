import matplotlib
matplotlib.use('Agg')

import os
import matplotlib.pyplot as plt 

import csv 

def plot_loss(opt, training_loss_list, training_acc_list, validation_loss_list, validation_acc_list, training_iteration_list, iteration):

    if not os.path.exists('plots/{}/{}/'.format(opt.model, opt.dataset)):
        os.makedirs('plots/{}/{}/'.format(opt.model, opt.dataset))
            
    plt.figure()
    plt.title("Loss")
    plt.plot(training_iteration_list, training_loss_list, label="train")
    plt.plot(training_iteration_list, validation_loss_list, label="validation")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(opt.plot_path + '{}/{}/{}_loss_n{}.jpg'.format(opt.model, opt.dataset, opt.name, iteration))
    plt.close()
    
    plt.figure()
    plt.title("Accuracy")
    plt.plot(training_iteration_list, training_acc_list, label="training")
    plt.plot(training_iteration_list, validation_acc_list, label="validation")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(opt.plot_path + '{}/{}/{}_accuracy_n{}.jpg'.format(opt.model, opt.dataset, opt.name, iteration))
    plt.close()
    
    
def save_results_test(opt):
    """
    Helper function to create the files to save the final results for each iteration of the kfold
    training as a csv file.

    Parameters
    ----------
    args : Arguments
        args.dataset, args.name : Parameters to decide the name of the output image file

    Returns
    -------
    f_test : file for test
    writer_test : writer for test

    """
    if not os.path.exists(opt.results_path + '{}/{}/'.format(opt.model, opt.dataset)):
        os.makedirs(opt.results_path + '{}/{}/'.format(opt.model, opt.dataset))
        


    if os.path.isfile(opt.results_path + '{}/{}/{}_test_results.csv'.format(opt.model, opt.dataset, opt.name)):
        f_test = open(opt.results_path + '{}/{}/{}_test_results.csv'.format(opt.model, opt.dataset, opt.name), 'a')
        writer_test = csv.writer(f_test)
    else:
        f_test = open(opt.results_path + '{}/{}/{}_test_results.csv'.format(opt.model, opt.dataset, opt.name), 'w')
        # create the csv writer
        writer_test = csv.writer(f_test)
        header_test = ['training_iteration', 'accuracy', 'roc_auc_score', 'FPR', 'FNR']
        writer_test.writerow(header_test)
    
    return f_test, writer_test


def save_results_train(opt):
    """
    Helper function to create the files to save the final results for each iteration of the kfold
    training as a csv file.

    Parameters
    ----------
    args : Arguments
        args.dataset, args.name : Parameters to decide the name of the output image file

    Returns
    -------
    f_val : file for validation
    writer_val : writer for validation

    """
    if not os.path.exists(opt.results_path + '{}/{}/'.format(opt.model, opt.dataset)):
        os.makedirs(opt.results_path + '{}/{}/'.format(opt.model, opt.dataset))
        
    print("Results file: ", opt.results_path + '{}/{}/{}_val_results.csv'.format(opt.model, opt.dataset, opt.name))

    if os.path.isfile(opt.results_path + '{}/{}/{}_val_results.csv'.format(opt.model, opt.dataset, opt.name)):
        f_val = open(opt.results_path + '{}/{}/{}_val_results.csv'.format(opt.model, opt.dataset, opt.name), 'a')
        writer_val = csv.writer(f_val)
    
    else:
        f_val = open(opt.results_path + '{}/{}/{}_val_results.csv'.format(opt.model, opt.dataset, opt.name), 'w')
        # create the csv writer
        writer_val = csv.writer(f_val)
        header_val = ['training_iteration', 'best_loss', 'best_accuracy', 'best_auc_roc'] 
        writer_val.writerow(header_val)

    
    return f_val, writer_val