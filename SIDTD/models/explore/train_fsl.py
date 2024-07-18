import sklearn.metrics 
from sklearn.metrics import accuracy_score
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
import json
import argparse
# from utils.data_tools_clip import TaskSampler, EasySet
from SIDTD.models.fsl_model.utils.data_tools import TaskSampler, TaskSamplerCoAARC, EasySetRandom, EasySetCoAARCRandom
from SIDTD.models.fsl_model.utils.utils import sliding_average
from matplotlib import pyplot as plt
import torchvision
import torchvision.models as models 
from efficientnet_pytorch import EfficientNet
from SIDTD.models.fsl_model.models.modeling import VisionTransformer, CONFIGS
import timm
import os
from torch.nn import Linear
import numpy as np
import csv 
from SIDTD.models.fsl_model.models_binary import ArcBinaryClassifier, CustomResNet50, CoAttn


def plot_loss(args, training_loss_list, training_acc_list, validation_acc_list, training_roc_auc_list, validation_roc_auc_list, training_iteration_list, nb_rep):

    if not os.path.exists(f'plots/{args.model}/{args.dataset}'):
        os.makedirs(f'plots/{args.model}/{args.dataset}')
            
    plt.figure()
    plt.title("Loss")
    plt.plot(training_iteration_list, training_loss_list, label="train")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(f'plots/{args.model}/{args.dataset}/fsl_loss_{args.name}_nb_iteration_{str(nb_rep)}.jpg')
    plt.close()
    
    plt.figure()
    plt.title("Accuracy")
    plt.plot(training_iteration_list, training_acc_list, label="train")
    plt.plot(training_iteration_list, validation_acc_list, label="validation")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(f'plots/{args.model}/{args.dataset}/fsl_accuracy_{args.name}_nb_iteration_{str(nb_rep)}.jpg')
    plt.close()

    plt.figure()
    plt.title("ROC AUC")
    plt.plot(training_iteration_list, training_roc_auc_list, label="train")
    plt.plot(training_iteration_list, validation_roc_auc_list, label="validation")
    plt.xlabel("epoch")
    plt.ylabel("ROC AUC")
    plt.legend()
    plt.savefig(f'plots/{args.model}/{args.dataset}/fsl_lroc_auc_{args.name}_nb_iteration_{str(nb_rep)}.jpg')
    plt.close()


def setup_trans_fg(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prepare model
    model_type = 'ViT-L_16'
    config = CONFIGS[model_type]
    config.split = 'non-overlap'
    config.slide_step = 12
    num_classes = 2
    img_size = 299
    model = VisionTransformer(config = config, img_size=img_size, zero_head=True, num_classes=2, smoothing_value=0.0)
    print('ImageNet pretrained')
    model.load_from(np.load('../transfg/transfg_pretrained/imagenet21k+imagenet2012_ViT-L_16.npz'))
    model.part_head = Linear(config.hidden_size, 1000)
    #load the trained weights on imagenet
    model.to(device)
    
    num_params = count_parameters(model)
    print("{}".format(config))
    print("Total Parameter: \t%2.1fM" % num_params)
    return model 

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def setup_model_baseline(args, pretrained=True):
    print("MODEL SETUP: ", args.model)
    if args.model == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(model.fc.in_features, args.embed_dim)
        
    elif args.model in ['vit_small_patch16_224', 'vit_large_patch16_224']:
        model = timm.create_model(args.model, pretrained=True)
        net_cfg = model.default_cfg
        last_layer = net_cfg['classifier']
        num_ftrs = getattr(model, last_layer).in_features
        setattr(model, last_layer, nn.Linear(num_ftrs,  args.embed_dim))
        
    elif args.model == 'efficientnet-b3':
        model = EfficientNet.from_pretrained(args.model)
        model._fc = nn.Linear(model._fc.in_features,  args.embed_dim)
        
    return model 

class PrototypicalNetworksCoAARC(nn.Module):
    def __init__(self, backbone1: nn.Module,  backbone2: nn.Module,  backbone3: nn.Module):
        super(PrototypicalNetworksCoAARC, self).__init__()


        self.backbone1 = backbone1
        self.backbone2 = backbone2
        self.backbone3 = backbone3
        self.bn1 = nn.BatchNorm1d(1000)
        self.fc1 = nn.Linear(1000, 100)

    def forward(
            self,
            support_images: torch.Tensor,
            support_labels: torch.Tensor,
            query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """
        
        #print(support_images.shape)
        B_support, P_support, C_support, W_support, H_support = support_images.size()
        support_images = self.backbone1.forward(support_images.view(B_support*P_support, C_support, W_support, H_support))
        _, C_support, W_support, H_support = support_images.size()
        support_images = support_images.view(B_support, P_support, C_support, W_support, H_support)
        support_images = self.backbone2.forward(support_images)
        z_support = self.backbone3.forward(support_images)

        B_query, P_query, C_query, W_query, H_query = query_images.size()
        query_images = self.backbone1.forward(query_images.view(B_query*P_query, C_query, W_query, H_query))
        _,C_query, W_query, H_query  = query_images.size()
        query_images = query_images.view(B_query, P_query, C_query, W_query, H_query)
        query_images = self.backbone2.forward(query_images)
        z_query = self.backbone3.forward(query_images)

        
        z_support = F.relu(self.fc1(self.bn1(z_support.float())))
        z_query = F.relu(self.fc1(self.bn1(z_query.float())))

        # Infer the number of different classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        z_proto = torch.cat([z_support[torch.nonzero(support_labels == label)].mean(0) for label in range(n_way)])

        # Compute the euclidean distance from queries to prototypes

        dists = torch.cdist(z_query, z_proto)

        # Distances into classification scores
        scores = -dists
        return scores

class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworks, self).__init__()
        # self.model_clip = model_clip
        # self.model
        self.backbone = backbone
        self.bn1 = nn.BatchNorm1d(1000)
        self.fc1 = nn.Linear(1000, 100)
        
    def _extract_feature_map(self, x):
        for name, module in self.backbone._modules.items():
            if name in ["avgpool", "fc"]:break
            
            x = module(x)
            
        return x


    def forward(
            self,
            support_images: torch.Tensor,
            support_labels: torch.Tensor,
            query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """

        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)
        

        z_support = F.relu(self.fc1(self.bn1(z_support.float())))
        z_query = F.relu(self.fc1(self.bn1(z_query.float())))


        n_way = 2
        # Prototype i is the mean of all instances of features corresponding to labels == i
        z_proto = torch.cat([z_support[torch.nonzero(support_labels == label)].mean(0) for label in range(n_way)])

        # Compute the euclidean distance from queries to prototypes

        dists = torch.cdist(z_query, z_proto)

        # Distances into classification scores
        scores = -dists
        return scores

def main(args):
    def fit(support_images, support_labels, query_images, query_labels, episode_index, N_QUERY):
        """
        Args:
            Torch.tensor
        Returns:
            loss as float
        """
        optimizer.zero_grad()
        classification_scores = model(support_images.cuda(), support_labels.cuda(), query_images.cuda())
        loss = criterion(classification_scores, query_labels.cuda())
        
        loss.backward()
        optimizer.step()

        preds = list(classification_scores.detach().data.argmax(1).to('cpu').numpy())
        reals = list(query_labels.to('cpu').numpy())
        accuracy = accuracy_score(reals, preds) *100

        p_pred = F.softmax(classification_scores.detach().data, dim=1)
        p_preds= p_pred[:,1].to('cpu').numpy()
        roc_auc_score = sklearn.metrics.roc_auc_score(reals, p_preds)

        return loss.item(), accuracy, roc_auc_score


    def evaluate(data_loader: DataLoader, N_QUERY: int):
        # We'll count everything and compute the ratio at the end
        p_preds = []
        preds = []
        reals = []

        # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
        # no_grad() tells torch not to keep in memory the whole computational graph (it's more lightweight this way)
        model.eval()
        with torch.no_grad():
            for episode_index, (
                    support_images,
                    support_labels,
                    query_images,
                    query_labels,
                    class_ids,
                    #         ) in tqdm(enumerate(data_loader), total=len(data_loader)):
            ) in enumerate(data_loader):
                y_preds = model(support_images.cuda(), support_labels.cuda(), query_images.cuda())
                
                preds = preds + list(y_preds.detach().data.argmax(1).to('cpu').numpy())
                
                p_pred = F.softmax(y_preds.detach().data, dim=1)
                p_preds.extend(p_pred[:,1].to('cpu').numpy())
                reals = reals + list(query_labels.to('cpu').numpy())

        roc_auc_score = sklearn.metrics.roc_auc_score(reals, p_preds)
        accuracy = accuracy_score(reals, preds)
        print("Model tested on {} tasks. Accuracy: {}% ROC AUC: {}".format(len(data_loader),(100 * accuracy), roc_auc_score))
        return (accuracy * 100), roc_auc_score 

    N_WAY = 2  # Number of classes in a task (reals, fakes)
    K_SHOT = args.k_shot  # Number of images per class in the support set
    N_QUERY = args.q_shot # Number of images per class in the query set
    N_EVALUATION_TASKS = 100

    # EPISODIC TRAINING

    N_TRAINING_EPISODES = args.episodes
    LEARNING_RATE = 0.0001

    log_update_frequency = 10
    val_episode = 50


    # Generate Train, Val and Test sets
    if args.model in ['vit_small_patch16_224', 'vit_large_patch16_224', 'coaarc']:
        image_size = 224
    else: 
        image_size = 299

    device = "cuda" if torch.cuda.is_available() else "cpu"  
    N_CLASSES = 2
    print('USE CELoss ONLY')
    random_sample = True
    save_folder_name = 'ce_loss'

    partitions_test = ['svk_alb_aze_lva', 'svk_rus_esp_aze', 'grc_esp_fin_srb', 'svk_alb_grc_est', 'alb_grc_rus_lva', 'alb_grc_est_rus', 'svk_rus_lva_fin', 'rus_aze_lva_fin', 'svk_alb_lva_fin', 'svk_alb_rus_fin']

    for nb_rep in range(args.repetition):
        name_repetition = partitions_test[nb_rep]
        test_metaclass = name_repetition.split('_')
        print('Repetition number:', nb_rep, 'out of:', args.repetition)

        l_metaclass = ['alb', 'aze', 'est', 'esp', 'grc', 'fin', 'lva', 'rus', 'svk', 'srb']
        for c in test_metaclass:
            l_metaclass.remove(c)
        train_metaclass = l_metaclass

        if args.model == 'coaarc':
            train_set = EasySetCoAARCRandom(args.dataset, train_metaclass, image_size=image_size, training=True)
            test_set = EasySetCoAARCRandom(args.dataset, test_metaclass, image_size=image_size, training=False)
        else:
            train_set = EasySetRandom(args.dataset, train_metaclass, image_size=image_size, training=True)
            test_set = EasySetRandom(args.dataset, test_metaclass, image_size=image_size, training=False)

        if args.model in ['efficientnet-b3', 'resnet50', 'vit_small_patch16_224', 'vit_large_patch16_224']:
            model_backbone = setup_model_baseline(args)
            for param in model_backbone.parameters():
                param.requires_grad = False
            model = PrototypicalNetworks(model_backbone).cuda()
        elif args.model == 'trans_fg':
            model_backbone = setup_trans_fg(args)
            for param in model_backbone.parameters():
                param.requires_grad = False
            model = PrototypicalNetworks(model_backbone).cuda()
        elif args.model == 'coaarc':
            model_backbone1 = CustomResNet50()
            model_backbone2 = CoAttn()  
            model_backbone3 = ArcBinaryClassifier(num_glimpses=6,
                                                glimpse_h=8,
                                                glimpse_w=8,
                                                channels = 1024,
                                                controller_out=128)


            model_backbone3.dense2 = nn.Linear(64, args.embed_dim)

            for param in model_backbone1.parameters():
                param.requires_grad = False

            for param in model_backbone2.parameters():
                param.requires_grad = False

            for param in model_backbone3.parameters():
                param.requires_grad = False

            model = PrototypicalNetworksCoAARC(backbone1 = model_backbone1, backbone2 = model_backbone2, backbone3 = model_backbone3).cuda()
        else:
            print('WRONG MODEL')

        
        print("Model Created... Running on: ", device)

        if args.model=='coaarc':
            test_sampler = TaskSamplerCoAARC(
                test_set, n_way=N_WAY, n_shot=K_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS, random_sample=random_sample
            )   
            val_sampler = TaskSamplerCoAARC(
                test_set, n_way=N_WAY, n_shot=K_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS, random_sample=random_sample
            )
            train_sampler = TaskSamplerCoAARC(train_set, n_way=N_WAY, n_shot=K_SHOT, n_query=N_QUERY, n_tasks=N_TRAINING_EPISODES, random_sample=random_sample)

        else:
            test_sampler = TaskSampler(
                test_set, n_way=N_WAY, n_shot=K_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS, random_sample=random_sample, forged_data_augm = args.forged_data_augm, training = False
            )   
            val_sampler = TaskSampler(
                test_set, n_way=N_WAY, n_shot=K_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS, random_sample=random_sample, forged_data_augm = args.forged_data_augm, training = False
            )
            train_sampler = TaskSampler(train_set, n_way=N_WAY, n_shot=K_SHOT, n_query=N_QUERY, n_tasks=N_TRAINING_EPISODES, random_sample=random_sample, forged_data_augm = args.forged_data_augm, training = True)

        test_loader = DataLoader(
            test_set,
            batch_sampler=test_sampler,
            num_workers=8,
            pin_memory=True,
            collate_fn=test_sampler.episodic_collate_fn,
        )

        # Validation Loader
        

        val_loader = DataLoader(
            test_set,
            batch_sampler=test_sampler,
            num_workers=8,
            pin_memory=True,
            collate_fn=test_sampler.episodic_collate_fn,
        )
        
        train_loader = DataLoader(
            train_set,
            batch_sampler=train_sampler,
            num_workers=8,
            pin_memory=True,
            collate_fn=train_sampler.episodic_collate_fn,
        )

        # Train the model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        best_accuracy = 0
        all_loss = []
        training_loss_list, training_acc_list, validation_acc_list, training_roc_auc_list, validation_roc_auc_list, training_iteration_list = [],[],[],[], [], []
        model.train()

        with tqdm(enumerate(train_loader), total=len(train_loader)) as tqdm_train:
            for episode_index, (support_images, support_labels, query_images, query_labels, _) in tqdm_train:
                loss_value, train_accuracy, train_roc_auc_score = fit(support_images, support_labels, query_images, query_labels, episode_index, N_QUERY)
                all_loss.append(loss_value)

                if episode_index % log_update_frequency == 0:
                    tqdm_train.set_postfix(loss=sliding_average(all_loss, log_update_frequency))

                if episode_index % val_episode == 0 and episode_index != 0:

                    training_roc_auc_list.append(train_roc_auc_score)
                    training_acc_list.append(train_accuracy)
                    
                    print("Model tested on Train Set. Accuracy: {}% ROC AUC: {}".format((train_accuracy), train_roc_auc_score))
                    
                    accuracy, roc_auc_score = evaluate(val_loader, N_QUERY)
                    validation_roc_auc_list.append(roc_auc_score)
                    validation_acc_list.append(accuracy)

                    training_loss_list.append(loss_value)
                    training_iteration_list.append(episode_index)
                    plot_loss(args, training_loss_list, training_acc_list, validation_acc_list, training_roc_auc_list, validation_roc_auc_list, training_iteration_list, nb_rep)

                    tqdm_train.set_postfix(acc=accuracy)
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        torch.save(model.state_dict(), f"/trained_models/fsl_model/{args.dataset}/{args.model}/{args.name}_{args.model}_kshot_{args.k_shot}_iteration_{name_repetition}.pth")
        print("\nTraining Complete\n!")

        model.load_state_dict(torch.load(f"/trained_models/fsl_model/{args.dataset}/{args.model}/{args.name}_{args.model}_kshot_{args.k_shot}_iteration_{name_repetition}.pth"))
        accuracy, roc_auc_score = evaluate(test_loader, N_QUERY)


        print("\nAccuracy Per Classes\n!")
        episodes = 20
        model.eval()
        p_preds = []
        reals = []
        acc = []

        for episode in tqdm(range(episodes)):

            (example_support_images, example_support_labels, example_query_images,
            example_query_labels, example_class_ids) = next(iter(test_loader))

            example_scores = model(example_support_images.cuda(), example_support_labels.cuda(), example_query_images.cuda())
            preds = torch.argmax(example_scores.detach().data, dim=-1)
            p_pred = F.softmax(example_scores.detach().data, dim=1)
            p_preds.extend(p_pred[:,1].to('cpu').numpy())
            reals = reals + list(example_query_labels.to('cpu').numpy())
            
            _, example_predicted_labels = torch.max(example_scores.detach().data, 1)
            l_correct = list(preds.to('cpu').numpy() == example_query_labels.to('cpu').numpy())
            acc = acc + l_correct


        roc_auc_score_tot = sklearn.metrics.roc_auc_score(reals, p_preds)
        acc_tot = np.mean(acc)
        
        print('Accuracy total {}'.format(acc_tot))
        print('ROC AUC total {}'.format(roc_auc_score_tot))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices = ['vit_large_patch16_224', 'vit_small_patch16_224', 'efficientnet-b3', 'resnet50', 'trans_fg', 'coaarc'], default = 'resnet50', type=str, help= "Model used to perform the training. The model name will also be used to identify the csv/plot results for each model.")
    parser.add_argument('--embed_dim',    default=1000,         type=int,   help='Embedding dimensionality of the network')
    parser.add_argument('--dataset', default='MIDV2020', choices = ['obvio_MIDV2020','MIDV2020', 'clip_cropped_MIDV2020'], help='Dataset name for this configuration. Needed for saving model score in a separate folder.')
    parser.add_argument('--k_shot',    default=5,         type=int,   help='number of shots')
    parser.add_argument('--q_shot',    default=5,         type=int,   help='number of query')
    parser.add_argument('--name', default='few_shot_setting', help='Custom name for this configuration. Needed for saving model score in a separate folder.')
    parser.add_argument('--smoothap', action="store_true", help='enables smoothap loss')
    parser.add_argument('--w_alpha', default=2, type=float, help='Smooth AP weight (named alpha) for the combined loss with Cross Entropy loss and Smooth AP loss: tot_loss = alpha * SmoothAP + beta * CE loss')
    parser.add_argument('--w_beta', default=0.5, type=float, help='Cross Entropy weight (named beta) for the combined loss with Cross Entropy loss and Smooth AP loss: tot_loss = alpha * SmoothAP + beta * CE loss')
    parser.add_argument('--episodes',    default=10000,         type=int,   help='Number of training epìsodes')
    parser.add_argument('--forged_data_augm', action="store_true", help='enables forgery augmentation')
    parser.add_argument("--shift_low", type=int, default=15, help= "shift constant for crop and replace")
    parser.add_argument("--shift_high", type=int, default=20, help= "shift constant for crop and replace")
    parser.add_argument("--shift_fontsize", type=float, default=1, help= "shift constant for crop and replace")
    parser.add_argument('--repetition',    default=10,         type=int,   help='number of repetition')
    args = parser.parse_args()
        
    main(args)
