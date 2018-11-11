# coding=utf-8
from multiprocessing import Process, Lock
import sharedmem
import time
import tensorflow as tf
import os
import numpy as np
from six.moves import xrange
import pandas as pd
import scipy.io as sio
import glob


def idxAnalysis(cross, vp):

    # define val and test
    testNo = cross
    valNo = (cross + 1) % 10

    train_idx = np.nonzero(np.logical_and(csvLines[:, 1] != testNo, csvLines[:, 1] != valNo) == True)[0]
    valIdx = np.array(np.nonzero(csvLines[:, 1] == valNo)).squeeze()
    testIdx = np.array(np.nonzero(csvLines[:, 1] == testNo)).squeeze()

    train_idx_shuffle = train_idx.copy()
    np.random.seed([(vp + 1) * (cross + 1)])
    np.random.shuffle(train_idx_shuffle)
    train_shuffle_idx_source = range(train_idx.shape[0])
    np.random.seed([(vp + 1) * (cross + 1)])
    np.random.shuffle(train_shuffle_idx_source)

    val_noAug_idx, test_noAug_idx = csvAnaysis(cross)

    idxData = idxLoader(train_idx_shuffle, testIdx, test_noAug_idx, valIdx, val_noAug_idx)

    return idxData


def csvAnaysis(cross):

    val_noAug_idx = [[], []]
    test_noAug_idx = [[], []]
    valcount = 0
    testcount = 0

    # define val and test
    testNo = cross
    valNo = (cross + 1) % 10

    for lineNo in range(allCount):
        if csvLines[lineNo, 1] == valNo:
            # positive without aug
            if csvLines[lineNo, 6] == 1 and csvLines[lineNo, 7] == 0:
                val_noAug_idx[0].append(valcount)
            # negtive
            elif csvLines[lineNo, 6] == 0:
                val_noAug_idx[1].append(valcount)
            valcount += 1
        elif csvLines[lineNo, 1] == testNo:
            # positive without aug
            if csvLines[lineNo, 6] == 1 and csvLines[lineNo, 7] == 0:
                test_noAug_idx[0].append(testcount)
            # negtive
            elif csvLines[lineNo, 6] == 0:
                test_noAug_idx[1].append(testcount)
            testcount += 1
    return val_noAug_idx, test_noAug_idx

class SharedData(object):

    def create(self, sampleShape_local, labelShape_local, batch_size_local, batchPoolSize_local, batchBlockNum_local, latestStep):
        datashape = [batchPoolSize_local, batch_size_local] + sampleShape_local
        labelshape = [batchPoolSize_local, batch_size_local] + labelShape_local

        self.data = np.frombuffer(sharedmem.empty(datashape, dtype=np.float32), dtype=np.float32).reshape(datashape)
        self.label = np.frombuffer(sharedmem.empty(labelshape, dtype=np.int64), dtype=np.int64).reshape(labelshape)
        self.step = np.frombuffer(sharedmem.full(batchPoolSize_local, [latestStep]*batchPoolSize_local, dtype=np.int64), dtype=np.int64)
        # 0: ready to use, 1: used, 2:full of updating, 3 updating conti.
        self.state = np.frombuffer(sharedmem.full(batchPoolSize_local * (batchBlockNum_local + 1), [1]*batchPoolSize_local * (batchBlockNum_local + 1), dtype=np.int8), dtype=np.int8).reshape(batchBlockNum_local + 1, batchPoolSize_local)

    def init(self, latestStep):
        self.step[:] = self.step[:] * 0 + latestStep
        self.state[:] = self.state[:] * 0 + 1


class SubProcessCtrl(object):
    '''This object is created to record the ppid, pid and state-flag of present processors and its child processors.
    It will be used to evaluate and control posterity processors to be terminated correctly and in ordered '''

    def create(self, readProcessorNum, eval_readProcessorNum):
        lenth = max(readProcessorNum, eval_readProcessorNum) +1
        self.table = np.frombuffer(sharedmem.full(lenth*3*4, [-1]*lenth*3*4,  dtype=np.int64), dtype=np.int64).reshape(3, lenth, 4)


    def start(self, mode, row):
        # type: train, val , test, clean

        if mode == 'train':
            dim = 0
        elif mode == 'val' or mode == 'test':
            dim = 1
        elif mode == 'clean':
            dim = 2

        # if a read manager is record,
        if row == -1:
            while self.table[dim, -1, 2] == 1: # in case last read manager has not been dead
                time.sleep(1)
                print 'waiting for last manager to dead'
            self.table[dim, :, :] = -1 #  then initial the record in this sheet

        # record the processor in this row
        self.table[dim, row, 0] = os.getppid()  # present parent processor id
        self.table[dim, row, 1] = os.getpid()   # present processor id
        self.table[dim, row, 2] = 1             # present processor status, -1:initial value, 1: alive, 0: dead
        self.table[dim, row, 3] = 1             # present processor passive operation, -1:initial value, 1:working, 0:terminating

    def activeFinish(self, mode, row):
        # type: train, val , test, clean

        if mode == 'train':
            dim = 0
        elif mode == 'val' or mode == 'test':
            dim = 1
        elif mode == 'clean':
            dim = 2

        # change status to dead
        self.table[dim, row, 2] = 0 # present processor status, -1:initial value, 0:dead, 1: alive

    def passiveFinishAll(self):

        # change operation to finish
        self.table[:, :-1, 3] = 0   # present processor operation, -1:initial value, 0:dead, 1: alive

    def evaluateFinish(self, mode, row):

        if mode == 'train':
            dim = 0
        elif mode == 'val' or mode == 'test':
            dim = 1
        elif mode == 'clean':
            dim = 2

        if self.table[dim, row, 3] == 0:
            return True
        else:
            return False


    # evaluate that the child processor of present pid is all terminated
    def evaluateChild(self, pid, mode ,rows):

        if mode == 'train':
            dim = 0
        elif mode == 'val' or mode == 'test':
            dim = 1
        elif mode == 'clean':
            dim = 2

        if np.all(self.table[dim, :rows, 2] == 0) and np.all(self.table[dim, :rows, 0] == pid):
            return 'dead'
        else:
            return 'alive'

    # evaluate that the child processor of present pid is all terminated
    def evaluateManager(self, pid, train_flag):

        if train_flag[0] == 0:# test mode

            if np.all(self.table[1, -1, 2] == 0) and np.all(self.table[1, -1, 0] == pid):
                return 'dead'
            else:
                return 'alive'

        else: # train and eval mode

            if cleanCacheFlag == 0:

                if np.all(self.table[:-1, -1, 2] == 0) and np.all(self.table[:-1, -1, 0] == pid):
                    return 'dead'
                else:
                    return 'alive'
            else:

                if np.all(self.table[:-1, -1, 2] == 0) and np.all(self.table[:-1, -1, 0] == pid) and self.table[-1, 0, 2] == 0:
                    return 'dead'
                else:
                    return 'alive'



def readData(dataShared, idxData, spcl, **kwargs):

    lock = Lock()
    mode = kwargs['mode']

    spcl.start(mode, -1)# the read manager will be record in the final row


    print '%s read manager - %d - mother start' % (mode, os.getpid())


    if mode == 'train':
        readNum = readProcessorNum
    elif mode =='val' or mode =='test':
        readNum = eval_readProcessorNum

    # a flag which tell the readData manager that the fill slaver is activated.
    token = sharedmem.full(readNum, [0]*readNum, dtype=np.int8)

    manager = [[]] * readNum

    for num in range(readNum):
        manager[num] = Process(target=fillData, args=(lock, dataShared, idxData, spcl, num, readNum, token), kwargs=kwargs)
        manager[num].start()
    for num in range(readNum):
        manager[num].join()

    while True:
        if sum(token) == readNum and spcl.evaluateChild(os.getpid(), mode, readNum) == 'dead':
            break
        time.sleep(1)

    spcl.activeFinish(mode, -1)
    print '%s read manager - %d - mother dead' % (mode, os.getpid())
    print np.concatenate((spcl.table[0, :, :], spcl.table[1, :, :], spcl.table[2, :, :]), 1)


def fillData(lock, dataShared, idxData, spcl, num, readNum, token, **kwargs):

    for key in kwargs:
        exec (key + " = kwargs[key] ")

    token[num] = 1
    spcl.start(mode, num)
    print '%s fill data processors -%d --- %d/%d activated' % (mode, os.getpid(), num+1, readNum)


    data_shared = dataShared.data
    dataLabel_shared = dataShared.label
    step_info = dataShared.step
    state_info = dataShared.state


    if mode == 'val':
        # maxStep_ = idxData.valIdx.shape[0] * 1.0 / eval_batch_size
        # maxStep = math.ceil(maxStep_)
        dataIdxList = idxData.valIdx
        maxStep = len(dataIdxList) // eval_batch_size + 1
        batchBlockNum_ = eval_batchBlockNum

    if mode == 'test':
        # maxStep_ = idxData.testIdx.shape[0] * 1.0 / eval_batch_size
        # maxStep = math.ceil(maxStep_)
        dataIdxList = idxData.testIdx
        maxStep = len(dataIdxList) // eval_batch_size + 1
        batchBlockNum_ = eval_batchBlockNum

    if mode == 'train':
        dataIdxList = idxData.train_idx_shuffle
        maxStep = idxData.maxStep
        batchBlockNum_ = batchBlockNum

    fimg = open(imgName, 'rb')
    while True:

        # print 'im am read data'

        # if the read processor operation is 0, then terminate this processor
        if spcl.evaluateFinish(mode, num):
            break

        lock.acquire()
        if step_info.max() == maxStep - 1:
            if state_info[0, np.nonzero(step_info==step_info.max())[0][0]] != 3:
                lock.release()
                break

        if np.any(np.logical_or(state_info[0, :] == 1, state_info[0, :] == 3)): # the first row in state, it means the state of each batch

            idx_column_list = np.nonzero(state_info[0, :] == 3)[0]
            if idx_column_list.size == 0:
                idx_column_list = np.nonzero(state_info[0, :] == 1)[0]

            idx_column = idx_column_list[0]

            step_info_copy = step_info.copy()
            state_info_copy = state_info.copy()
            block_idx_list= np.nonzero(state_info_copy[1:, idx_column] == 1)[0]  # get which block is this batch is avaiable to update
            try: # 0: ready to use, 1: used, 2:full of updating, 3 updating conti.
                block_idx = block_idx_list.min()
            except Exception as e:
                f = open(os.path.join(getPath(projectPath), 'errorlog.txt'), 'w')
                f.write(str(e))
                f.write('\ncurrent pid is %d and parent pid is %d' % (os.getpid(), os.getppid()))
                f.write('\nidx_column is %d' % idx_column)
                f.write('\norg state_info as follows:\n')
                np.savetxt(f, np.c_[state_info_copy], fmt='%d')
                f.write('\norg step_info as follows:\n')
                np.savetxt(f, np.c_[step_info_copy], fmt='%d')
                f.write('\nnew state_info as follows:\n')
                np.savetxt(f, np.c_[state_info], fmt='%d')
                f.write('\nnew step_info as follows:\n')
                np.savetxt(f, np.c_[step_info], fmt='%d')
                f.close()
                lock.release()
                continue

            idx_row = block_idx + 1

            state_info[idx_row, idx_column] = 2 # tell other processor , I am going to update this block in this batch

            if idx_row == 1:
                step_info[idx_column] = step_info.max() + 1 # if this is the first block in this batch, updating the step # that this batch represents
                state_info[0, idx_column] = 3
                if mode == 'train':
                    read_timer_batch[step_info[idx_column], 0] = time.time()
            if idx_row == batchBlockNum_: # this means it is the final block in this batch
                state_info[0, idx_column] = 2  # so, this batch is full of processors to update
                final_flag = True
            else:
                final_flag = False
                
            step = step_info[idx_column].copy()
            lock.release()

            
            # update data_shared(idx[0][0]) based on step .....

            if mode == 'train':
                offset_st = (step * batch_size) % len(dataIdxList)
                offset_end = ((step + 1) * batch_size) % len(dataIdxList)
            elif mode == 'val' or mode == 'test':
                offset_st = (step * eval_batch_size) % len(dataIdxList)
                offset_end = ((step+1) * eval_batch_size) % len(dataIdxList)

            if offset_st < offset_end:
                batch_idx = range(offset_st, offset_end)
            else:
                batch_idx = range(offset_st, len(dataIdxList))
                batch_idx.extend(range(offset_end))

            groupNum = len(batch_idx) // batchBlockNum_

            if block_idx == batchBlockNum_ - 1:
                start = block_idx * groupNum
                end = len(batch_idx)
            else:
                start = block_idx * groupNum
                end = (block_idx + 1) * groupNum

            for m in range(start, end):
                idx_bin = dataIdxList[batch_idx[m]]

                ########### data augmentation can be added here


                #### .bin file decoder here#################
                fimg.seek(tLength * idx_bin)
                bufferData = fimg.read(tLength)
                dataLabel_shared[idx_column, m, :, :] = np.frombuffer(bufferData[0], dtype=np.uint8).astype(np.int64)
                data_shared[idx_column, m, :, :, :] = (np.frombuffer(bufferData[1:], dtype=np.float32)).reshape(rows, columns, 1)

            if mode == 'train' and final_flag:
                read_timer_batch[step_info[idx_column], 1] = time.time()

            # after filling the data, change the state to 0, means ready to use
            state_info[idx_row, idx_column] = 0

            # if not any '1' or '2' in each block of this batch, it means that this batch is ready to use
            if not state_info[1:, idx_column].any():
                state_info[0, idx_column] = 0

            # print step_info
            # print state_info

        else:
            lock.release()
    fimg.close()


    # change the status of this processor
    spcl.activeFinish(mode, num)
    print '%s fill data processors -%d --- %d/%d terminated' % (mode, os.getpid(), num + 1, readNum)


class idxLoader(object):
    def __init__(self, load1, load2, load3, load4, load5):
        self.train_idx_shuffle = load1
        self.testIdx = load2
        self.test_noAug_idx = load3
        self.valIdx = load4
        self.val_noAug_idx = load5
        self.train_size = load1.shape[0]
        self.test_size = load2.shape[0]
        self.val_size = load4.shape[0]
        self.maxStep = int(num_epochs * self.train_size) // batch_size + 1


def cleanMem(step_info, maxStep, spcl):

    spcl.start('clean', 0)

    while True:
        os.system("sync &&  echo 3 > /proc/sys/vm/drop_caches")

        if step_info.max() == maxStep - 1:
            break

        # if the read processor operation is 0, then terminate this processor
        if spcl.evaluateFinish('clean', 0):
            break

    # change the status of this processor, finally
    spcl.activeFinish('clean', 0)


def getPath(path):
    datatime = time.strftime("%Y")+time.strftime("%m")+time.strftime("%d")+'_'+time.strftime("%H")+time.strftime("%M")+time.strftime("%S")
    # return '/home/yanchaocc/data/luna_JP_exp/exp-data-20161212_110824'
    os.makedirs(os.path.join(path, 'exp-data-' + datatime))
    return os.path.join(path, 'exp-data-' + datatime)



def weight(num_channels, num_label_classes, SEED):
    # define the weight parameter sizes
    Wb = {
        'W1': tf.Variable(tf.truncated_normal([5, 5, num_channels, 24], stddev=0.1, seed=SEED, dtype=data_type())),
        'b1': tf.Variable(tf.zeros([24], dtype=data_type())),
        'W2': tf.Variable(tf.truncated_normal([3, 3, 24, 32], stddev=0.1, seed=SEED, dtype=data_type())),
        'b2': tf.Variable(tf.zeros([32], dtype=data_type())),
        'W3': tf.Variable(tf.truncated_normal([3, 3, 32, 48], stddev=0.1, seed=SEED, dtype=data_type())),
        'b3': tf.Variable(tf.zeros([48], dtype=data_type())),
        'fcw1': tf.Variable(tf.truncated_normal([6 * 6 * 48, 16], stddev=0.1, seed=SEED, dtype=data_type())),
        'fcb1': tf.Variable(tf.zeros([16], dtype=data_type())),
        'fcw2': tf.Variable(tf.truncated_normal([16, num_label_classes], stddev=0.1, seed=SEED, dtype=data_type())),
        'fcb2': tf.Variable(tf.zeros([num_label_classes], dtype=data_type()))}
    return Wb

# define the cnn model structure
def model(data, Wb, SEED, train=False):
    with tf.variable_scope('conv1'):
        conv = tf.nn.conv2d(data, Wb['W1'], strides=[1, 1, 1, 1], padding='VALID')
        relu = tf.nn.relu(tf.nn.bias_add(conv, Wb['b1']))
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[
            1, 2, 2, 1], padding='VALID')
    with tf.variable_scope('conv2'):
        conv = tf.nn.conv2d(pool, Wb['W2'], strides=[1, 1, 1, 1], padding='VALID')
        relu = tf.nn.relu(tf.nn.bias_add(conv, Wb['b2']))
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[
            1, 2, 2, 1], padding='VALID')
    with tf.variable_scope('conv3'):
        conv = tf.nn.conv2d(pool, Wb['W3'], strides=[1, 1, 1, 1], padding='VALID')
        relu = tf.nn.relu(tf.nn.bias_add(conv, Wb['b3']))
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[
            1, 2, 2, 1], padding='VALID')
    with tf.variable_scope('reshape'):
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [pool_shape[0], pool_shape[
            1] * pool_shape[2] * pool_shape[3]])
    with tf.variable_scope('fc1'):
        hidden = tf.nn.relu(tf.matmul(reshape, Wb['fcw1']) + Wb['fcb1'])
    with tf.variable_scope('dropout'):
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    with tf.variable_scope('fc2'):
        out = tf.matmul(hidden, Wb['fcw2']) + Wb['fcb2']
    return out


# Small utility function to evaluate a dataset by feeding batches of data to
# {eval_data} and pulling the results from {eval_predictions}.
# Saves memory and enables this to run on smaller GPUs.
def tf_eval_in_batch(sess, valDataShared, idxData, mode, eval_data, eval_prediction):

    data = valDataShared.data
    dataLabel = valDataShared.label
    step_info = valDataShared.step
    state_info = valDataShared.state

    if mode == 'val':
        dataIdxList = idxData.valIdx
        maxStep = len(dataIdxList) // eval_batch_size + 1
        noAug_idx = idxData.val_noAug_idx
    if mode == 'test':
        dataIdxList = idxData.testIdx
        maxStep = len(dataIdxList) // eval_batch_size + 1
        noAug_idx = idxData.test_noAug_idx

    predictions = np.ndarray(shape=(len(dataIdxList), num_label_classes), dtype=np.float32)
    gt_labels = np.ndarray(shape=(len(dataIdxList)), dtype=np.float32)

    for step in xrange(maxStep):

        # Compute the offset of the current minibatch in the data.
        # Note that we could use better randomization across epochs.

        # this while loop is used to wait avaiable data
        while True:
            try:
                idx_column = np.nonzero(np.logical_and(step_info == step, state_info[0, :] == 0))[0][0]
                break
            except Exception:
                time.sleep(waitPeriod)
                continue

        if step == maxStep-1 and len(dataIdxList)%eval_batch_size == 0:
            continue
        else:
            batch_data = data[idx_column, :, :, :, :]
            batch_labels = np.squeeze(dataLabel[idx_column, :, :, :])
            batch_predictions = sess.run(eval_prediction, feed_dict={eval_data: batch_data})

            if step == maxStep-1 and len(dataIdxList)%eval_batch_size != 0:# final
                predictions[step * eval_batch_size:, :] = batch_predictions[:len(dataIdxList)%eval_batch_size, ...]
                gt_labels[step * eval_batch_size:] = batch_labels[:len(dataIdxList)%eval_batch_size]
            else:# not final
                predictions[step * eval_batch_size:(step + 1) * eval_batch_size, :] = batch_predictions
                gt_labels[step * eval_batch_size:(step + 1) * eval_batch_size] = batch_labels
        # after eval this batch, unlock the tag of this batch
        state_info[:, idx_column] = 1

    err = error_rate(predictions, gt_labels)
    posNoAug_err = error_rate(predictions[noAug_idx[0], ...], gt_labels[noAug_idx[0]])
    negNoAug_err = error_rate(predictions[noAug_idx[1], ...], gt_labels[noAug_idx[1]])

    return err, posNoAug_err, negNoAug_err, predictions, gt_labels

def delete_save_data(dire, epoch, step):

    name = 'epoch-%04d-model.ckpt-%d*' % (epoch, step)
    listing = glob.glob(os.path.join(dire, name))
    for candidate in listing:
        try:
            os.remove(candidate)
        except Exception:
            pass

def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0])

class PrLog(object):

    def create(self, numOfEpoch, *args):
        self.args = args
        self.data = dict()
        for item in self.args:
            self.data[item] = np.frombuffer(sharedmem.full(numOfEpoch, [0]*numOfEpoch, dtype=np.float64), dtype=np.float64)
        self.data['tried_times'] = np.frombuffer(sharedmem.full(1, [0], dtype=np.int8), dtype=np.int8)

    def ini(self, *args):
        for item in args:
            self.data[item][:] = self.data[item][:] * 0

    def save(self, dire, step, batch_size, train_size, maxKeep):

        epoch_idx = epochPeriod(step, batch_size, train_size) - 1
        saveData = dict()
        for item in self.args:
            saveData[item] = self.data[item][:epoch_idx+1]
        sio.savemat(os.path.join(dire, 'epoch-%04d-model.ckpt-%d.mat' % (epoch_idx+1, step)), saveData)

        # log mat datas that outside the max_keep
        if epoch_idx + 1 - maxKeep >= 1:
            delete_save_data(dire, epoch_idx + 1 - maxKeep, self.data['step'][epoch_idx - maxKeep])

    def reload(self, dire, train_flag):

        # load old data
        self.loaded_data_from_mat = sio.loadmat(glob.glob(os.path.join(dire, '*%d.mat' % train_flag))[0])

        # find matches
        match = list(set(self.loaded_data_from_mat.keys()).intersection(self.data.keys()))

        # update matches
        for item in match:

            # bug fixed on 20th, Dec, 2016
            # try:
            #     self.data[item][:np.squeeze(self.loaded_data_from_mat[item]).shape[0]] = np.squeeze(self.loaded_data_from_mat[item])
            # except Exception, e:
            #     print e
            #     self.data[item] = np.squeeze(self.loaded_data_from_mat[item])[:np.squeeze(self.data[item]).shape[0]]
            #     continue
            #

            try:
                self.data[item][:self.loaded_data_from_mat[item].shape[1]] = self.loaded_data_from_mat[item]
            except Exception as e:
                print e
                self.data[item] = self.loaded_data_from_mat[item][:self.data[item].shape[1]]
                continue

    def getData(self, step, batch_size, train_size, item):

        epoch_idx = epochPeriod(step, batch_size, train_size) -1

        return self.data[item][epoch_idx:epoch_idx+1]

    def retry_record(self):
        self.data['tried_times'][0] += 1

    def stop_condition_eval(self, dire, step, batch_size, train_size, item, backLoop):

        epoch_idx = epochPeriod(step, batch_size, train_size) -1

        if epoch_idx >= backLoop:

            result = []

            for n in range(backLoop):

                if self.data[item][epoch_idx - n] > self.data[item][epoch_idx - backLoop]:
                    result.extend([True])
                else:
                    result.extend([False])

            if np.all(result): # continue increase for backLoop-th epoches
                # delete final backLoop saved files
                for n in range(backLoop):
                    delete_save_data(dire, epoch_idx+1 - n, self.data['step'][epoch_idx - n])
                # and return a correct flag
                return self.data['step'][epoch_idx-backLoop]
            else:
                return 'continue'
        else:
            return 'continue'


class TimeRecorder(object):
    def __init__(self):
        self.cost = 0.0
    def tic(self):
        self.st = time.time()
    def toc(self):
        self.cost += time.time() - self.st
    def freq(self, freq, size):
        return size * freq / self.cost
    def ini(self):
        self.cost = 0.0

def epochPeriod(step, batch_size, train_size):
    return step * batch_size //train_size  + 1

def epochFloat(step, batch_size, train_size):
    return float(step) * batch_size / train_size


def tf_things(train_flag, idxData, prlog, resultDir, trainDataShared):

    # create sub-processor-state recorder
    spcl = SubProcessCtrl()  # subProcess control flag
    spcl.create(readProcessorNum, eval_readProcessorNum)

    # judge initial run/ continue train/ testing
    if train_flag[0] == -1: # it is initial run
        tf_trainAndVal(idxData, prlog, train_flag, resultDir, trainDataShared, spcl)
    else:# it is reload run
        # reload log parameters
        prlog.reload(resultDir, train_flag[0])
        if prlog.data['tried_times'][0] >= reTryNum:
            tf_test(idxData, train_flag, resultDir, spcl)
            train_flag[0] = 0
        else:
            prlog.retry_record()
            tf_trainAndVal(idxData, prlog, train_flag, resultDir, trainDataShared, spcl)


    # return condition, which make sure that all child-processors are terminated correct
    while True:
        if spcl.evaluateManager(os.getpid(), train_flag) == 'dead':
            break
        time.sleep(1)
        print 'I am biggest monther- %d waiting child process' % os.getpid()
        print np.concatenate((spcl.table[0, :, :], spcl.table[1, :, :], spcl.table[2, :, :]), 1)
    return

def tf_test(idxData, train_flag, resultDir, spcl):

    # 3.1 creat shared memory for test data
    testDataShared = SharedData()
    testDataShared.create(sampleShape, labelShape, eval_batch_size, batchPoolSize, eval_batchBlockNum, latestStep=-1)

    # 3.2 creat a processor to manager reading test data
    Process(target=readData, args=(testDataShared, idxData, spcl), kwargs={'mode': 'test'}).start()

    # initialise the CNN-forward model
    Wb = weight(num_channels, num_label_classes, SEED)
    eval_data = tf.placeholder(data_type(), shape=(eval_batch_size, width, height, num_channels))
    eval_prediction = tf.nn.softmax(model(eval_data, Wb, SEED))

    # necessory parametes
    train_size = idxData.train_size

    saver = tf.train.Saver(max_to_keep=max_keep)
    with tf.Session() as sess:

        # reload model parameters
        model_path = unicode(os.path.join(resultDir, 'epoch-%04d-model.ckpt-%d' % (epochPeriod(train_flag[0], batch_size, train_size), train_flag[0])))
        print('Restored and testing from model:' + model_path)
        saver.restore(sess, model_path)

        # 3.3 val the data and save parameters
        err, posNoAug_err, negNoAug_err, pred, gt = tf_eval_in_batch(sess, testDataShared, idxData, 'test', eval_data, eval_prediction)

        # Finally print the result!

        print('Test err: %.3f%%, Test+noAug err: %.1f%%, Test-noAug err: %.1f%%' % (err, posNoAug_err, negNoAug_err))

        # save the
        # predictResult = tf.argmax(preds, 1).eval()
        with open(os.path.join(resultDir, 'result.txt'), 'w') as f:
            np.savetxt(f, np.c_[pred, gt])

    # terminate reading processors, # stop all children process
    spcl.passiveFinishAll()



def tf_trainAndVal(idxData, prlog, train_flag, resultDir, trainDataShared, spcl):

    # creat shared memory variable
    # data_shared, dataLabel_shared, data_shared_stepInfo, data_shared_stateInfo = sharedDataCreat(batch_size, batchBlockNum, global_step_local=train_flag[0])

    read_timer_batch_org = sharedmem.full(idxData.maxStep*2, [-1]*idxData.maxStep*2, dtype=np.float64)
    read_timer_batch = np.frombuffer(read_timer_batch_org, dtype=np.float64).reshape(idxData.maxStep, 2)

    # creat a processor to manage reading data simutanously
    Process(target=readData, args=(trainDataShared, idxData, spcl), kwargs={'mode': 'train', 'read_timer_batch': read_timer_batch}).start()

    # a process to clean cache simutanously, for performance testing
    if cleanCacheFlag == 1:
        Process(target=cleanMem, args=(trainDataShared.step, idxData.maxStep, spcl)).start()

    # fucking train now
    tfTrain(trainDataShared, idxData, prlog, train_flag, resultDir, spcl)

    # terminate reading processors, # stop all children process
    spcl.passiveFinishAll()


def tfTrain(trainDataShared, idxData, prlog, train_flag, resultDir, spcl):

    '''
    １： define the weight and build the graph
    ２： open a sess ,and construct a loop to train data batch by batch
    ３： evaluate the validation data in each epoch
    ４： if the val err continusly increased by n epochs, break the loop, delete n latest epoches, and return the final step number of the (n+1)th epoch from the latest.
    '''

    # 1.1 define parametes
    train_size = idxData.train_size
    max_step = idxData.maxStep

    data_shared = trainDataShared.data
    dataLabel_shared = trainDataShared.label
    step_info = trainDataShared.step
    state_info = trainDataShared.state
    learning_rate = tf.constant(initialLR * decentRate ** prlog.data['tried_times'][0])

    # 1.2 define kernal weight
    Wb = weight(num_channels, num_label_classes, SEED)

    # 1.3 define placehold, which represents the imported data size of the tensorflow graph
    train_data_node = tf.placeholder(data_type(), shape=(batch_size, width, height, num_channels))
    train_labels_node = tf.placeholder(tf.int64, shape=(batch_size,))
    eval_data = tf.placeholder(data_type(), shape=(eval_batch_size, width, height, num_channels))

    # 1.4 Training computation: logits + cross-entropy loss.
    logits = model(train_data_node, Wb, SEED, True)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=train_labels_node))  # sigmoid_cross_entropy_with_logits
    # sigmoid_cross_entropy_with_logits

    # # L2 regularization for the fully connected parameters.
    # regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
    #                 tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    # # Add the regularization term to the loss.
    # loss += 5e-4 * regularizers

    # # Optimizer: set up a variable that's incremented once per batch and
    # # controls the learning rate decay.
    # batch = tf.Variable(0)
    # # Decay once per epoch, using an exponential schedule starting at 0.01.
    # # Base learning rate.
    # # Current index into the dataset.
    # # Decay step.
    # # Decay rate.
    # learning_rate = tf.train.exponential_decay(0.01, batch * batch_size, train_size, 0.95, staircase=True)
    # # Use simple momentum for the optimization.
    # optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)
    # # optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, name='RMSProp').minimize(loss, global_step=batch)

    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)

    # 1.5 Predictions for the current training minibatch.
    train_prediction = tf.nn.softmax(logits)

    # 1.6 Predictions for the test and validation, which we'll compute less often.
    eval_prediction = tf.nn.softmax(model(eval_data, Wb, SEED))

    # 2.0 Create a local session to run the training.
    saver = tf.train.Saver(max_to_keep=max_keep)
    with tf.Session() as sess:
        # sess = tf.Session()

        # Run all the initializers to prepare the trainable parameters.
        if train_flag[0] == -1:
            # tf.initialize_all_variables().run()
            st = time.time()
            tf.global_variables_initializer().run()
            global_step = 0
            print('Initialized! takes %.5f') % (time.time()-st)
        else:
            # model_path = unicode(glob.glob(os.path.join(resultDir, '*%d' % train_flag[0]))[0])
            model_path = unicode(os.path.join(resultDir, 'epoch-%04d-model.ckpt-%d' % (epochPeriod(train_flag[0], batch_size, train_size), train_flag[0])))
            print('Restored and continue training on:' + model_path)
            st = time.time()
            saver.restore(sess, model_path)
            print ('Restored takes %.5f') % (time.time()-st)
            global_step = train_flag[0] + 1

        # record the cnn graph
        st = time.time()
        summary_writer = tf.summary.FileWriter(resultDir, sess.graph)
        # print ('Restored takes %.5f') % (time.time()-st)

        wt = TimeRecorder() # wait data time
        rt = TimeRecorder() # reference time
        tt = TimeRecorder() # train time

        # Loop through training steps.
        for step in xrange(global_step, max_step):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.

            # this while loop is used to wait avaiable data
            wt.tic()
            while True:
                try:
                    idx_column = np.nonzero(np.logical_and(step_info == step, state_info[0, :] == 0))[0][0]
                    break
                except Exception:
                    time.sleep(waitPeriod)
                    continue
            wt.toc()

            # create batch data reference
            rt.tic()
            batch_data = data_shared[idx_column, :, :, :, :]
            batch_labels = np.squeeze(dataLabel_shared[idx_column, :, :, :])
            rt.toc()

            tt.tic()
            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph it should be fed to.
            feed_dict = {train_data_node: batch_data, train_labels_node: batch_labels}
            # Run the graph and fetch some of the nodes.
            _, l, lr, predictions = sess.run([optimizer, loss, learning_rate, train_prediction], feed_dict=feed_dict)
            tt.toc()

            # after training this batch, unlock the tag of this batch
            # enable the blocks and enable the door. :), bug fixed on 20th, Dec, 2016
            state_info[1:, idx_column] = 1
            state_info[0, idx_column] = 1

            # record running average loss
            if step == 0:
                mv_loss = l
            elif step == train_flag[0] + 1:
                mv_loss = 0.99 * prlog.getData(train_flag[0], batch_size, train_size, 'mv_loss_mat')[0] + 0.01 * l
            else:
                mv_loss = 0.99 * mv_loss + 0.01 * l

            # real-time print infomation
            if step - global_step != 0 and (step - global_step) % loss_frequency == 0:
                print 'Step %d (epoch %.2f), mv-los: %.5f, bh-los: %.3f, lr: %.5f, %.1f img/s, [wt:rf:tt=%.2f:%.2f:%.2f]' % (step, epochFloat(step, batch_size, train_size), mv_loss, l, lr, loss_frequency*batch_size/(wt.cost+rt.cost+tt.cost), wt.cost/(wt.cost+rt.cost+tt.cost), rt.cost/(wt.cost+rt.cost+tt.cost), tt.cost/(wt.cost+rt.cost+tt.cost))
                wt.ini()
                rt.ini()
                tt.ini()


            # save the model and log parameters for each epoch
            if epochPeriod(step+1, batch_size, train_size) - epochPeriod(step, batch_size, train_size) == 1:
                # save the model
                checkpoint_path = os.path.join(resultDir, 'epoch-%04d-model.ckpt' % (epochPeriod(step, batch_size, train_size)))
                saver.save(sess, checkpoint_path, global_step=step)

                # 3 evaluate the validation data

                # 3.1 creat shared memory for validation data
                valDataShared = SharedData()
                valDataShared.create(sampleShape, labelShape, eval_batch_size, batchPoolSize, eval_batchBlockNum, latestStep=-1)
                # 3.2 creat a processor to manager reading val data
                Process(target=readData, args=(valDataShared, idxData, spcl), kwargs={'mode': 'val'}).start()

                # 3.3 val the data and save parameters
                err, posNoAug_err, negNoAug_err, _, _ = tf_eval_in_batch(sess, valDataShared, idxData, 'val', eval_data, eval_prediction)

                prlog.getData(step, batch_size, train_size, 'var_err')[0] = err
                prlog.getData(step, batch_size, train_size, 'valPosnoAug_err')[0] = posNoAug_err
                prlog.getData(step, batch_size, train_size, 'valNegnoAug_err')[0] = negNoAug_err
                prlog.getData(step, batch_size, train_size, 'step')[0] = step
                prlog.getData(step, batch_size, train_size, 'mv_loss_mat')[0] = mv_loss
                prlog.getData(step, batch_size, train_size, 'lr_mat')[0] = lr
                prlog.save(resultDir, step, batch_size, train_size, max_keep)

                # 3.5 real-time print validation performance
                print('Val err: %.3f%%, Val+noAug err: %.1f%%, Val-noAug err: %.1f%% in epoch %.2f' % (err, posNoAug_err, negNoAug_err, epochFloat(step, batch_size, train_size)))

                # 4.0 evaluate the val-err performance
                flag = prlog.stop_condition_eval(resultDir, step, batch_size, train_size, 'var_err', backLoop=2)

                if flag == 'continue':
                    # continue train
                    continue
                else:
                    train_flag[0] = flag
                    break


# specify your training log output dir
projectPath = getPath('/home/huhui/test/yj_test/yj_test')

# specify your data dir
imgName = '/media/huhui/testDisk/view0.bin'

csvName = '/media/huhui/testDisk/LUNASTable.csv'
csvLines = pd.read_csv(csvName).values[0:]
allCount = len(csvLines)

# data shape parameters
width, height, num_channels= 64, 64, 1
label_num_channels = 1

columns, rows, byteLength = 64, 64, 4
tLength = 1 + columns * rows * byteLength

sampleShape = [height, width, num_channels]
labelShape = [1, label_num_channels]

# data parallel reading parameters
cleanCacheFlag = 0 # if equeal to 1, the .py script must be executed in root mode.
batchPoolSize = 10
waitPeriod = 0.001

batch_size = 128
batchBlockNum = 2 # a divider of batch_size
readProcessorNum = 8 # a multiple of batchBlockNum

eval_batch_size = 1280
eval_batchBlockNum = 8 # a divider of batch_size
eval_readProcessorNum = 8 # a multiple of batchBlockNum

# CNN-training parametes
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
num_epochs = 20
SEED = 66478
num_label_classes = 2
max_keep = 5
loss_frequency = 200 # how much steps to print the loss infomation
def data_type():
    return tf.float32

# early stop parametes
train_flag = sharedmem.full(1, [-1], dtype=np.int64)
initialLR = 0.01
reTryNum = 5
decentRate = 0.5
stopLR = initialLR * (decentRate**reTryNum)


def main():

    for vp in range(10):
        for cross in range(10):

            # setup result output path
            if not os.path.isdir(os.path.join(projectPath, 'View{}'.format(vp))):
                os.makedirs(os.path.join(projectPath, 'View{}'.format(vp)))
            if not os.path.isdir(os.path.join(projectPath, 'View{}'.format(vp), 'Cross{}'.format(cross))):
                os.makedirs(os.path.join(projectPath, 'View{}'.format(vp), 'Cross{}'.format(cross)))
            resultDir = os.path.join(projectPath, 'View{}'.format(vp), 'Cross{}'.format(cross))
            os.system('chmod -R 777 '+ projectPath)

            # analysis data idx
            idxData = idxAnalysis(cross, vp)

            # init parametes
            train_flag[0] = -1

            prlog = PrLog()
            prlog.create(num_epochs, 'step', 'var_err', 'valPosnoAug_err', 'valNegnoAug_err', 'mv_loss_mat', 'lr_mat')

            trainDataShared = SharedData()
            trainDataShared.create(sampleShape, labelShape, batch_size, batchPoolSize, batchBlockNum, latestStep=train_flag[0])

            while train_flag[0] != 0:

                p = Process(target=tf_things, args=(train_flag, idxData, prlog, resultDir, trainDataShared))
                p.start()
                p.join()
                # tf_things(train_flag, idxData, prlog, resultDir, trainDataShared)
                trainDataShared.init(latestStep=train_flag[0])
                prlog.ini('step', 'lr_mat', 'var_err', 'valPosnoAug_err', 'valNegnoAug_err', 'mv_loss_mat')

if __name__ == '__main__':
    main()
