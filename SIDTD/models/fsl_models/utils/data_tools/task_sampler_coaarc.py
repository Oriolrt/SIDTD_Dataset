import random
from typing import List, Tuple

import torch
from torch.utils.data import Sampler, Dataset
from torch.autograd import Variable
import numpy as np


class TaskSamplerCoAARC(Sampler):
    """
    Samples batches in the shape of few-shot classification tasks. At each iteration, it will sample
    n_way classes, and then sample support and query images from these classes.
    """

    def __init__(
        self, dataset: Dataset, n_way: int, n_shot: int, n_query: int, n_tasks: int, random_sample: bool = True
    ):
        """
        Args:
            dataset: dataset from which to sample classification tasks. Must have a field 'label': a
                list of length len(dataset) containing containing the labels of all images.
            n_way: number of classes in one task
            n_shot: number of support images for each class in one task
            n_query: number of query images for each class in one task
            n_tasks: number of tasks to sample
        """
        super().__init__(data_source=None)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_tasks = n_tasks
        self.random_sample = random_sample

        self.items_per_label = {}
        assert hasattr(dataset, "labels") # "TaskSampler needs a dataset with a field 'label' containing the labels of all images."
        for item, label in enumerate(dataset.labels):
            if label in self.items_per_label.keys():
                self.items_per_label[label].append(item)
            else:
                self.items_per_label[label] = [item]

    def __len__(self):
        return self.n_tasks

    def __iter__(self):
        for _ in range(self.n_tasks):

            label_set = [0,1]
            if self.random_sample:
                label_set = random.sample(label_set, self.n_way)
                
            yield torch.cat(
                [
                    # pylint: disable=not-callable
                    torch.tensor(
                        random.sample(
                            self.items_per_label[label], self.n_shot + self.n_query
                        )
                    )
                    # pylint: enable=not-callable
                    for label in random.sample(label_set, self.n_way)
                ]
            )

    def episodic_collate_fn(
        self, input_data: List[Tuple[torch.Tensor, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        Collate function to be used as argument for the collate_fn parameter of episodic
            data loaders.
        Args:
            input_data: each element is a tuple containing:
                - an image as a torch Tensor
                - the label of this image
        Returns:
            tuple(Tensor, Tensor, Tensor, Tensor, list[int]): respectively:
                - support images,
                - their labels,
                - query images,
                - their labels,
                - the dataset class ids of the class sampled in the episode
        """

        true_class_ids = list({x[2] for x in input_data})

        all_images_random = torch.cat([x[0].unsqueeze(0) for x in input_data])
        all_images_random = all_images_random.reshape(
            (self.n_way, self.n_shot + self.n_query, *all_images_random.shape[1:]) # (n_way, B, c, h, w)
            
        )
        all_images_real = torch.cat([x[1].unsqueeze(0) for x in input_data])
        all_images_real = all_images_real.reshape(
            (self.n_way, self.n_shot + self.n_query, *all_images_real.shape[1:]) # (n_way, B, c, h, w)
        )

        all_images = torch.stack([all_images_random, all_images_real], dim=2)  # (n_way, B, 2, c, h, w)
        # pylint: disable=not-callable
        all_labels = torch.tensor(
            [true_class_ids.index(x[2]) for x in input_data]
        ).reshape((self.n_way, self.n_shot + self.n_query))
        # pylint: enable=not-callable

        support_images = all_images[:, : self.n_shot].reshape((-1, *all_images.shape[2:]))
        query_images = all_images[:, self.n_shot :].reshape((-1, *all_images.shape[2:]))
        support_labels = all_labels[:, : self.n_shot].flatten()
        query_labels = all_labels[:, self.n_shot :].flatten()

        return (
            support_images,
            support_labels,
            query_images,
            query_labels,
            true_class_ids,
        )


