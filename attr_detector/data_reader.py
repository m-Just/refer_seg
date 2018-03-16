from __future__ import print_function

import numpy as np
import os
import json
import skimage, skimage.io
import threading
import Queue as queue

def run_prefetch(prefetch_queue, folder_name, instance_by_attr, attr_num, num_instance, batch_size, num_batch, shuffle):
    assert attr_num >= batch_size

    n_batch_prefetch = 0
    attr_by_index = instance_by_attr.keys()
    attr_to_fetch = np.arange(attr_num)
    while True:
        if shuffle:
            attr_to_fetch = np.random.permutation(attr_num)

        batch = dict()
        batch['im'] = []
        for n_ins in range(batch_size):
            attr = attr_by_index[attr_to_fetch[n_ins]]
            ins_id = np.random.randint(len(instance_by_attr[attr]))
            data = instance_by_attr[attr][ins_id]
            for k in data:
                if k not in batch:
                    batch[k] = list()
                batch[k].append(data[k])

            im_file = os.path.join(folder_name, '%d.jpg' % data['image_id'])
            im = skimage.io.imread(im_file).astype(np.float32)
            if len(im.shape) == 2: im = np.stack([im] * 3, axis=-1)
            batch['im'].append(im)
        np.save('./train_batch/batch_%d.npy' % n_batch_prefetch, batch)
            
        # add loaded batch to fetchqing queue
        prefetch_queue.put(batch, block=True)

        # Move to next batch
        n_batch_prefetch = (n_batch_prefetch + 1) % num_batch

class DataReader:
    def __init__(self, data_file, folder_name, batch_size, attr_num, shuffle=True, prefetch_num=8):
        self.data_file = data_file
        self.folder_name = folder_name
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.attr_num = attr_num
        self.prefetch_num = prefetch_num

        self.n_batch = 0
        self.n_epoch = 0

        with open(data_file) as f:
            data = json.load(f)

        self.instance_by_attr = dict()
        for im_data in data:
            for bbox in im_data['attributes']:
                attrs = map(str, bbox['attributes'])
                if len(attrs) > 1: continue
                instance = {'image_id': im_data['image_id'],
                            'width': im_data['width'],
                            'height': im_data['height'],
                            'bbox_x': bbox['x'],
                            'bbox_y': bbox['y'],
                            'bbox_w': bbox['w'],
                            'bbox_h': bbox['h'],
                            'attrs': attrs}
                for attr in attrs:
                    if attr not in self.instance_by_attr:
                        self.instance_by_attr[attr] = []
                    self.instance_by_attr[attr].append(instance)

        self.num_instance = min(map(len, self.instance_by_attr.values())) *  self.attr_num
        self.num_batch = self.num_instance / batch_size
        print('generated %d batch from %s' %
            (self.num_batch, data_file.split('/')[-1]))

        # Start prefetching thread
        self.prefetch_queue = queue.Queue(maxsize=prefetch_num)
        self.prefetch_thread = threading.Thread(target=run_prefetch,
            args=(self.prefetch_queue, self.folder_name, self.instance_by_attr, self.attr_num,
                  self.num_instance, self.batch_size, self.num_batch, self.shuffle))
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

    def read_batch(self, is_log = True):
        if is_log:
            print('data reader: epoch = %d, batch = %d / %d' % (self.n_epoch, self.n_batch, self.num_batch))

        # Get a batch from the prefetching queue
        if self.prefetch_queue.empty():
            print('data reader: waiting for file input (IO is slow)...')
        batch = self.prefetch_queue.get(block=True)
        self.n_batch = (self.n_batch + 1) % self.num_batch
        self.n_epoch += (self.n_batch == 0)
        return batch
