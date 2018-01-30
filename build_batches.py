import sys
sys.path.append('/data/ryli/rmi_phrasecut/external/coco/PythonAPI')
import os
import argparse
import numpy as np
import json
import skimage
import skimage.io

from util import im_processing, text_processing
from util.io import load_referit_gt_mask as load_gt_mask
from refer import REFER
from pycocotools import mask as cocomask

def build_referit_batches(setname, T, input_H, input_W):
    # data directory
    im_dir = '/data/ryli/text_objseg/exp-referit/referit-dataset/images/'
    mask_dir = '/data/ryli/text_objseg/exp-referit/referit-dataset/mask/'
    query_file = '/data/ryli/rmi_phrasecut/data/referit/referit_query_' + setname + '.json'
    vocab_file = '/data/ryli/rmi_phrasecut/data/vocabulary_referit.txt'

    # saving directory
    data_folder = './data/referit/' + setname + '_batch/'
    data_prefix = 'referit_' + setname
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    fp = open('./data/referit/trainval_list.txt', 'w')

    # load annotations
    query_dict = json.load(open(query_file))
    im_list = query_dict.keys()
    vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)

    # collect training samples
    samples = []
    for n_im, name in enumerate(im_list):
        im_name = name.split('_', 1)[0] + '.jpg'
        mask_name = name + '.mat'
        for sent in query_dict[name]:
            samples.append((im_name, mask_name, sent))

    # save batches to disk
    num_batch = len(samples)
    for n_batch in range(num_batch):
        print('saving batch %d / %d' % (n_batch + 1, num_batch))
        im_name, mask_name, sent = samples[n_batch]
        fp.write('%d\t%s%s\n' % (n_batch, im_dir, im_name))
        mask = load_gt_mask(mask_dir + mask_name).astype(np.float32)

        text = text_processing.preprocess_sentence(sent, vocab_dict, T)

        np.savez(file = data_folder + data_prefix + '_' + str(n_batch) + '.npz',
            text_batch = text,
            im_name_batch = im_name,
            mask_batch = (mask > 0),
            sent_batch = [sent])
    fp.close()

def build_coco_batches(dataset, setname, T, input_H, input_W):
    im_dir = '/data/ryli/datasets/coco/images'
    im_type = 'train2014'
    vocab_file = '/data/ryli/rmi_phrasecut/data/vocabulary_Gref.txt'

    data_folder = './data/' + dataset + '/' + setname + '_batch/'
    data_prefix = dataset + '_' + setname
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    if dataset == 'Gref':
        refer = REFER('/data/ryli/rmi_phrasecut/external/refer/data', dataset = 'refcocog', splitBy = 'google')
    elif dataset == 'unc':
        refer = REFER('/data/ryli/rmi_phrasecut/external/refer/data', dataset = 'refcoco', splitBy = 'unc')
    elif dataset == 'unc+':
        refer = REFER('/data/ryli/rmi_phrasecut/external/refer/data', dataset = 'refcoco+', splitBy = 'unc')
    else:
        raise ValueError('Unknown dataset %s' % dataset)
    refs = [refer.Refs[ref_id] for ref_id in refer.Refs if refer.Refs[ref_id]['split'] == setname]
    vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)

    n_batch = 0
    for ref in refs:
        im_name = 'COCO_' + im_type + '_' + str(ref['image_id']).zfill(12)
        im = skimage.io.imread('%s/%s/%s.jpg' % (im_dir, im_type, im_name))
        seg = refer.Anns[ref['ann_id']]['segmentation']
        rle = cocomask.frPyObjects(seg, im.shape[0], im.shape[1])
        mask = np.max(cocomask.decode(rle), axis = 2).astype(np.float32)

        for sentence in ref['sentences']:
            print('saving batch %d' % (n_batch + 1))
            sent = sentence['sent']
            text = text_processing.preprocess_sentence(sent, vocab_dict, T)

            np.savez(file = data_folder + data_prefix + '_' + str(n_batch) + '.npz',
                text_batch = text,
                im_name_batch = im_name,
                mask_batch = (mask > 0),
                sent_batch = [sent])
            n_batch += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type = str, default = 'referit') # 'unc', 'unc+', 'Gref'
    parser.add_argument('-t', type = str, default = 'trainval') # 'test', val', 'testA', 'testB'

    args = parser.parse_args()
    T = 20
    input_H = 320
    input_W = 320
    if args.d == 'referit':
        build_referit_batches(setname = args.t,
            T = T, input_H = input_H, input_W = input_W)
    else:
        build_coco_batches(dataset = args.d, setname = args.t,
            T = T, input_H = input_H, input_W = input_W)
