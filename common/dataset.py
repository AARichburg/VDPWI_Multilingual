import os

import torch
import torch.nn as nn

from torchtext.vocab import Vectors

from datasets.sick import SICK


class UnknownWordVecCache(object):
    """
    Caches the first randomly generated word vector for a certain size to make it is reused.
    """
    cache = {}

    @classmethod
    def unk(cls, tensor):
        size_tup = tuple(tensor.size())
        if size_tup not in cls.cache:
            cls.cache[size_tup] = torch.Tensor(tensor.size())
            cls.cache[size_tup].normal_(0, 0.01)
        return cls.cache[size_tup]


class DatasetFactory(object):
    """
    Get the corresponding Dataset class for a particular dataset.
    """
    @staticmethod
   # def get_dataset(dataset_name, word_vectors_dir, word_vectors_file, batch_size, device, castor_dir, vector_dim=200, utils_trecqa="utils/trec_eval-9.0.5/trec_eval"):
 #       if dataset_name != None:
#            dataset_root = os.path.join(castor_dir, dataset_name)
#            train_loader, dev_loader, test_loader = SICK.iters(dataset_root, word_vectors_file, word_vectors_dir, batch_size, device=device, unk_init=UnknownWordVecCache.unk)
#            embedding = nn.Embedding.from_pretrained(SICK.TEXT_FIELD.vocab.vectors)
#            return SICK, embedding, train_loader, test_loader, dev_loader
#        elif dataset_name == 'eng_fre':
#            dataset_root = os.path.join(castor_dir, 'eng_fre/')
#            train_loader, dev_loader, test_loader = SICK.iters(dataset_root, word_vectors_file, word_vectors_dir, batch_size, device=device,unk_init=UnknownWordVecCache.unk)
#            embedding = nn.Embedding(len(SICK.TEXT_FIELD.vocab),vector_dim)
#            return SICK, embedding, train_loader, test_loader, dev_loader
#        else:
#            raise ValueError('{} is not a valid dataset.'.format(dataset_name))
    def get_dataset(dataset_name, word_vectors_dir, word_vectors_file, batch_size, device, castor_dir, vector_dim=200, utils_trecqa="utils/trec_eval-9.0.5/trec_eval"):
        dataset_root = os.path.join(castor_dir, dataset_name)
        train_loader, dev_loader, test_loader = SICK.iters(dataset_root, word_vectors_file, word_vectors_dir, batch_size, device=device, unk_init=UnknownWordVecCache.unk)
        embedding = nn.Embedding.from_pretrained(SICK.TEXT_FIELD.vocab.vectors)
        return SICK, embedding, train_loader, test_loader, dev_loader

