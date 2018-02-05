import sys
sys.path.append('/data/ryli/rmi_phrasecut/external/coco/PythonAPI')
import os
import argparse
import numpy as np
import json
import skimage
import skimage.io

from util import im_processing
from util.io import load_referit_gt_mask as load_gt_mask
from refer import REFER
from pycocotools import mask as cocomask

from skip_thoughts import configuration
from skip_thoughts import encoder_manager

def setup_encoder():
    VOCAB_FILE = '/data/ryli/kcli/skip-thoughts/pretrained/skip_thoughts_uni_2017_02_02/vocab.txt'
    EMBEDDING_MATRIX_FILE = '/data/ryli/kcli/skip-thoughts/pretrained/skip_thoughts_uni_2017_02_02/embeddings.npy'
    CHECKPOINT_PATH = '/data/ryli/kcli/skip-thoughts/pretrained/skip_thoughts_uni_2017_02_02/model.ckpt-501424'

    encoder = encoder_manager.EncoderManager()
    encoder.load_model(configuration.model_config(),
        vocabulary_file=VOCAB_FILE,
        embedding_matrix_file=EMBEDDING_MATRIX_FILE,
        checkpoint_path=CHECKPOINT_PATH)

    return encoder

def build_referit_batches(setname, input_H, input_W):
    pass

def build_coco_batches(dataset, setname,input_H, input_W):
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

    sent_data = []
    encoder = setup_encoder()
    for ref in refs:
        for sentence in ref['sentences']:
            sent_data.append(sentence['sent'].decode('latin-1').strip())
    encodings = encoder.encode(sent_data)

    n_batch = 0
    for ref in refs:
        im_name = 'COCO_' + im_type + '_' + str(ref['image_id']).zfill(12)
        im = skimage.io.imread('%s/%s/%s.jpg' % (im_dir, im_type, im_name))
        seg = refer.Anns[ref['ann_id']]['segmentation']
        rle = cocomask.frPyObjects(seg, im.shape[0], im.shape[1])
        mask = np.max(cocomask.decode(rle), axis = 2).astype(np.float32)

        if 'train' in setname:
            mask = im_processing.resize_and_pad(mask, input_H, input_W)

        for sentence in ref['sentences']:
            print('saving batch %d' % (n_batch + 1))
            sent = sentence['sent']

            np.savez(file = data_folder + data_prefix + '_' + str(n_batch) + '.npz',
                im_name_batch = im_name,
                encoding_batch = encodings[n_batch],
                mask_batch = (mask > 0),
                sent_batch = [sent])
            n_batch += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type = str, default = 'referit') # 'unc', 'unc+', 'Gref'
    parser.add_argument('-t', type = str, default = 'trainval') # 'test', val', 'testA', 'testB'

    args = parser.parse_args()
    input_H = 320
    input_W = 320
    if args.d == 'referit':
        build_referit_batches(setname = args.t, input_H = input_H, input_W = input_W)
    else:
        build_coco_batches(dataset = args.d, setname = args.t, input_H = input_H, input_W = input_W)
