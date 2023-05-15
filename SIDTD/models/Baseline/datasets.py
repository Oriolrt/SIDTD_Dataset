from SIDTD.utils.transforms import  *  # Load all type of forgery fonction: Copy paste, Crop & Replace and Inpainting

from torch.utils.data import Dataset


import cv2
import sys
import os
######################## dataloader for forgery data augmentation ##################################

flatten = lambda l: [item for sublist in l for item in sublist]

class TrainDatasets_augmented(Dataset):
    """
    This dataset class allows mini-batch formation pre-epoch, for greater speed. 
    This dataloader generate batch composed of 25% of fake data, 25% of fake data generated on-the-fly, 50% of real data.
    """
    def __init__(self, args, image_dict, path_img, transform=None):
        """
        Args:
            image_dict: two-level dict, `super_dict[super_class_id][class_id]` gives the list of
                        image paths having the same super-label and class label
        """
        self.image_dict = image_dict
        self.batch_size = args.batch_size
        self.faker_data_augmentation = args.faker_data_augmentation
        self.shift_crop = args.shift_crop
        self.shift_copy = args.shift_copy
        self.samples_per_class = self.batch_size // 4
        self.path_img = path_img
        
        # checks
        # provide avail_classes
        self.avail_classes = [*self.image_dict]
        # Data augmentation/processing methods.
        self.transform = transform

        self.reshuffle()


    def reshuffle(self):

        image_dict = copy.deepcopy(self.image_dict) 
        print('shuffling data')
        for sub in image_dict:
            random.shuffle(image_dict[sub])

        total_batches = []
        batch = []
        finished = 0
        if len(image_dict[0]) >= 2*len(image_dict[1]):       # One epoch is finished when all genuine document have been loaded
            while finished == 0:
                if (len(image_dict[0]) >= 2*self.samples_per_class):
                    l_clips = []
                    # Randomly 25% of the batch size of fake documents
                    list_fakes_img = list(np.random.choice(image_dict[1],self.samples_per_class))
                    l_clips = l_clips + [(1, x) for x in list_fakes_img]
                
                    # Randomly 50% of the batch size of genuine documents
                    list_reals_img = list(image_dict[0][:2*self.samples_per_class])
                    l_clips = l_clips + [(0, x) for x in list_reals_img]
                    image_dict[0] = image_dict[0][2*self.samples_per_class:] 

                    # Randomly 25% of the batch size of genuine documents that will be forged on-the-fly. 
                    # Label is set at 2 as "to-be-forged". When the document will be forged the label will be set to 1
                    list_forgery_augm_img = list(np.random.choice(list_reals_img,self.samples_per_class))
                    l_clips = l_clips + [(2, x) for x in list_forgery_augm_img]                
                    batch.append(l_clips)

                if len(batch) == 1:
                    total_batches.append(batch)
                    batch = []
                else:
                    finished = 1
        else:    # for the last batch we randomly draw a set of genuine documents, 
                 # hence it is possible to have duplicate of genuine documents in the last batch
            while finished == 0:
                if (len(image_dict[1]) >= self.samples_per_class):
                    l_clips = []
                    list_reals_img = list(np.random.choice(image_dict[0],2*self.samples_per_class))
                    l_clips = l_clips + [(0, x) for x in list_reals_img]
                
                    list_fakes_img = list(image_dict[1][:self.samples_per_class])
                    l_clips = l_clips + [(1, x) for x in list_fakes_img]
                    image_dict[1] = image_dict[1][self.samples_per_class:] 

                    list_forgery_augm_img = list(np.random.choice(list_reals_img,self.samples_per_class))
                    l_clips = l_clips + [(2, x) for x in list_forgery_augm_img]                
                    batch.append(l_clips)
                
                if len(batch) == 1:
                    total_batches.append(batch)
                    batch = []
                else:
                    finished = 1
        
        random.shuffle(total_batches)
        self.dataset = flatten(flatten(total_batches))

    def __getitem__(self, idx):
        # we use SequentialSampler together with SuperLabelTrainDataset,
        # so idx==0 indicates the start of a new epoch
        # if foregery augmentation is set, document with label 2 will be forged, drawing randomly one of the forgery techniques available: copy paste, crope & replace and inpainting.
        batch_item = self.dataset[idx]   # image path
        cls = batch_item[0]    # label
        image = cv2.imread(batch_item[1])        
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            print('ERROR')
        
        l_fake_type = ['crop', 'inpainting', 'copy']
        if self.faker_data_augmentation:
            if batch_item[0] == 2:
                cls = 1   # change label to one as the document will be forged
                fake_type = random.choice(l_fake_type)   # randomly draw one forgery techniques among: copy paste, crope & replace and inpainting
                id_img = batch_item[1].split('/')[-1]
                id_country = id_img[:3]    # ID's country
                path = 'split_kfold/clip_cropped_MIDV2020/annotations/annotation_' + id_country + '.json' 
                annotations = read_json(path) # read json with document annotations of fields area
                    
                # perform copy pasting
                if fake_type == 'copy':
                    image = CopyPaste(image, annotations, self.shift_copy)

                # perform inpainting
                if fake_type == 'inpainting':
                    image = Inpainting(image, annotations, id_country)

                # perform crop & replace
                if fake_type == 'crop':

                    if id_country in ['rus', 'grc']:
                        list_image_field = ['image']    # Russian and greek ID doesn't have signature on ID
                    else:
                        list_image_field = ['image', 'signature']
                        
                    dim_issue = True
                    while dim_issue:
                        # choose a document to crop the image or signature
                        img_path_clips_target = random.choice(self.path_img)
                        id_country_target = img_path_clips_target.split('/')[-1][:3]
                        if id_country_target in ['rus', 'grc']:
                            if 'signature' in list_image_field:
                                list_image_field.remove('signature')
                        image_target = cv2.imread(img_path_clips_target)
                        path = 'split_kfold/clip_cropped_MIDV2020/annotations/annotation_' + id_country_target + '.json'
                        annotations_target = read_json(path)
                            
                        image, dim_issue = CropReplace(image, annotations, image_target, annotations_target, list_image_field, self.shift_crop)

        # perform classic data augmentation on every document
        augmented = self.transform(image=image)
        image = augmented['image']

        return  image, cls   # image array, label


    def __len__(self):
        return len(self.dataset)
