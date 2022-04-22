'''
Created on Dec 18, 2018
Tensorflow Implementation of the Baseline Model, BPRMF, in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
from utility.load_data import Data
from utility.load_data import RecomDataset


class BPRMF_loader(RecomDataset):
    def __init__(self, args, path):
        super().__init__(args, path)


    def as_test_feed_dict(self, model, user_batch, item_batch, drop_flag=True):

        feed_dict = {
            model.users: user_batch,
            model.pos_items: item_batch
        }

        return feed_dict  
    def as_train_feed_dict(self, model, users, pos_items, neg_items):
        batch_data = {}
        batch_data['users'] = users
        batch_data['pos_items'] = pos_items
        batch_data['neg_items'] = neg_items
        feed_dict = {
            model.users: batch_data['users'],
            model.pos_items: batch_data['pos_items'],
            model.neg_items: batch_data['neg_items']
        }

        return feed_dict   