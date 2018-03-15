from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from loaders.loader import Loader

class SubjectParser(nn.Module):
    def __init__(self, vocab_size, word_embedding_size, word_vec_size, class_num):
        super(SubjectParser, self).__init__()
        self.embedding = nn.Embedding(vocab_size, word_embedding_size)
        self.mlp = nn.Sequential(nn.Linear(word_embedding_size, word_vec_size),
                                 nn.ReLU(),
                                 nn.Linear(word_vec_size, word_vec_size / 2),
                                 nn.ReLU())
        self.classifier = nn.Linear(word_vec_size / 2, class_num)
        self.tagger = nn.Sequential(nn.Linear(word_vec_size / 2, word_vec_size / 4),
                                    nn.ReLU(),
                                    nn.Linear(word_vec_size / 4, 1))

    def forward(self, input_label):
        embedded = self.embedding(input_label)
        embedded = self.mlp(embedded)

        class_pred = self.classifier(embedded)
        class_pred = F.softmax(class_pred)

        confidence = self.tagger(embedded)
        confidence = F.sigmoid(confidence)

        return class_pred, confidence, embedded

def main(args):
    data_json = '/data/ryli/kcli/refer_seg/MAttNet/cache/prepro/refcoco_unc/data.json'
    data_h5 = '/data/ryli/kcli/refer_seg/MAttNet/cache/prepro/refcoco_unc/data.h5'
    loader = Loader(data_json, data_h5)

    parser = SubjectParser(len(loader.word_to_ix), 512, 512)
    parser.cuda()

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(parser.parameters(), lr=learning_rate)

    def lossFun(loader, optimizer, parser, input_label, class_label):
        parser.train()
        optimizer.zero_grad()

        class_pred, confidence, embedded = parser(input_label)

        cls_error = F.cross_entropy(class_pred, class_label)

        loss.backward()
        optimizer.step()

        return loss.data[0], cls_error, confidence

    sent_count = 0
    for ref in loader.Refs:
        category_id = loader.Anns[ref['ann_id']]['category_id']
        for sent_id in ref['sent_ids']:
            sent = loader.sentences[sent_id]
            sent_count += 1
            print('Sentence %d: id(%d)' % sent_count, sent_id)
            for word in sent:
                loss, cls_error, confidence = lossFun(loader, optimizer, parser,
                    loader.word_to_ix[word], category_id, 90)
                print('\t%s: loss = %.4f, cls_error = %.4f, confidence = %.4f, lr = %.2E' %
                    loss, cls_error, confidence, learning_rate)
