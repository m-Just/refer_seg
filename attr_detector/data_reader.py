from __future__ import print_function

import numpy as np
import os
import json
import skimage, skimage.io
import threading
import Queue as queue

def run_prefetch(prefetch_queue, folder_name, datalist, num_instance, batch_size, num_batch, shuffle):
    n_batch_prefetch = 0
    fetch_order = np.arange(num_instance)
    while True:
        # Shuffle the batch order for every epoch
        if n_batch_prefetch == 0 and shuffle:
            fetch_order = np.random.permutation(num_instance)

        batch = dict([(k, []) for k in datalist[0]])
        batch['im'] = []
        for n_ins in range(batch_size):
            ins_id = fetch_order[n_ins + n_batch_prefetch * batch_size]
            data = datalist[ins_id]
            for k in data: batch[k].append(data[k])

            im_file = os.path.join(folder_name, '%d.jpg' % data['image_id'])
            im = skimage.io.imread(im_file).astype(np.float32)
            if len(im.shape) == 2: im = np.stack([im] * 3, axis=-1)
            batch['im'].append(im)
            
        # add loaded batch to fetchqing queue
        prefetch_queue.put(batch, block=True)

        # Move to next batch
        n_batch_prefetch = (n_batch_prefetch + 1) % num_batch

class DataReader:
    def __init__(self, data_file, folder_name, batch_size, shuffle=True, prefetch_num=8):
        self.data_file = data_file
        self.folder_name = folder_name
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.prefetch_num = prefetch_num

        self.n_batch = 0
        self.n_epoch = 0

        self.datalist = []
        with open(data_file) as f:
            data = json.load(f)
        for im_data in data:
            for attr in im_data['attributes']:
                instance = {'image_id': im_data['image_id'],
                            'width': im_data['width'],
                            'height': im_data['height'],
                            'bbox_x': attr['x'],
                            'bbox_y': attr['y'],
                            'bbox_w': attr['w'],
                            'bbox_h': attr['h'],
                            'attrs': map(str, attr['attributes'])}
                self.datalist.append(instance)

        self.num_instance = len(self.datalist)
        self.num_batch = self.num_instance / batch_size
        print('found %d batch from %d instances in %s' %
            (self.num_batch, self.num_instance, data_file.split('/')[-1]))

        # Start prefetching thread
        self.prefetch_queue = queue.Queue(maxsize=prefetch_num)
        self.prefetch_thread = threading.Thread(target=run_prefetch,
            args=(self.prefetch_queue, self.folder_name, self.datalist,
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
