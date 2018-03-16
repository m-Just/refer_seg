from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from loaders.loader import Loader

class SubjectParser(nn.Module):
    def __init__(self, vocab_size, word_embedding_size, word_vec_size,
        hidden_size, class_num, n_layers=1):
        super(SubjectParser, self).__init__()
        self.embedding = nn.Embedding(vocab_size, word_embedding_size)
        self.mlp = nn.Sequential(nn.Linear(word_embedding_size, word_vec_size),
                                 nn.ReLU())
        self.rnn = nn.LSTM(word_vec_size, hidden_size, n_layers, batch_first=True)
        self.classifier = nn.Linear(word_vec_size, class_num)

    def forward(self, input_label):
        input_length = (input_label != 0).sum(1)
        sorted_len, sorted_idx = input_length.sort(descending=True)
        _, recover_idx = sorted_idx.sort()

        input_label = input_label[sorted_idx]

        embedded = self.embedding(input_label)
        embedded = self.mlp(embedded)
        padded_embed = nn.utils.rnn.pack_padded_sequence(embedded, sorted_len, batch_first=True)

        output, hidden = self.rnn(padded_embed)

        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = output[recover_ixs]

        hidden = hidden[0]  # final hidden state
        hidden = hidden[:, recover_ixs, :]
        hidden = hidden.transpose(0,1).contiguous()
        hidden = hidden.view(hidden.size(0), -1)

        class_pred = self.classifier(output)

        return class_pred, output, hidden

def main():
    data_json = '/data/ryli/kcli/refer_seg/MAttNet/cache/prepro/refcoco_unc/data.json'
    data_h5 = '/data/ryli/kcli/refer_seg/MAttNet/cache/prepro/refcoco_unc/data.h5'
    loader = Loader(data_json, data_h5)

    parser = SubjectParser(len(loader.word_to_ix), 512, 512, 90)
    parser.cuda()

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(parser.parameters(), lr=learning_rate)

    def lossFun(loader, optimizer, parser, input_label, class_label):
        parser.train()
        optimizer.zero_grad()

        class_pred, rnn_output, rnn_hidden = parser(input_label)

        cls_error = F.cross_entropy(class_pred, class_label)
        #loss = confidence * torch.exp(F.threshold(-cls_error, -1.0, -1.0)) + 0.1 * (1 - confidence) * cls_error
        loss = cls_error

        loss.backward()
        optimizer.step()

        return loss.data[0], cls_error, class_pred

    sent_count = 0
    avg_accuracy = 0
    for ref_id in loader.Refs:
        ref = loader.Refs[ref_id]
        category_id = loader.Anns[ref['ann_id']]['category_id']
        for sent_id in ref['sent_ids']:
            sent = loader.sentences[sent_id]
            if len(sent['tokens']) > 2: continue
            sent_count += 1
            for word in sent['tokens']:
                word = word if word in loader.word_to_ix else '<UNK>'
                input_label = Variable(torch.cuda.LongTensor([loader.word_to_ix[word]]))
                class_label = Variable(torch.cuda.LongTensor([category_id-1]))
                loss, cls_error, cls_pred = lossFun(loader, optimizer, parser,
                    input_label, class_label)
                _, pred = torch.max(cls_pred, 1)
                if pred.data.cpu().numpy() == category_id - 1:
                    avg_accuracy = avg_accuracy * 0.99 + 0.01
                else:
                    avg_accuracy *= 0.99
            if sent_count % 100 == 0:
                print('Sentence %d: id(%d)' % (sent_count, sent_id))
                print('  %-12s: loss = %f, cls_error = %f, avg_accuracy = %.4f, lr = %.2E' %
                    (word, loss, cls_error, avg_accuracy, learning_rate))

if __name__ == '__main__':
    main()
