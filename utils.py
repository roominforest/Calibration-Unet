from libtiff import TIFF
import os
import yaml
import numpy as np
from loss import *
import torch.nn.functional as F

class Config(object):
    PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

    @classmethod
    def load(cls, file_path):
        stream = open(file_path)
        data = yaml.load(stream)
        for (k, v) in data.items():
            setattr(cls, k, v)
        return cls

class LossLoader():
    def __init__(self):
        pass

    @staticmethod
    def load(lossname, config):
        ignore_index = config.data['ignore_index']

        object = globals()[lossname]
        return object(ignore_index=ignore_index)

def binary(pred):
    pred = F.sigmoid(pred)
    return pred >0.5

def neg(idx):
    if idx == 0:
        return None
    else:
        return -idx

def mypad(x, pad, padv):
    def neg(b):
        if b==0:
            return None
        else:
            return -b
    pad = np.array(pad)
    x2 = np.zeros(np.array(x.shape)+np.sum(pad,1), dtype = x.dtype)+padv
    x2[pad[0,0]:neg(pad[0,1]), pad[1,0]:neg(pad[1,1]), pad[2,0]:neg(pad[2,1]), pad[3,0]:neg(pad[3,1])] = x
    return x2

class Averager():
    def __init__(self):
        self.n = 0
        self.total = 0
        self.type = 'scalar'

    def update(self, x):
        if isinstance(x, tuple):
            # print('tuple')
            if self.n == 0:
                self.list = [Averager() for i in range(len(x))]

            assert len(x) == len(self.list)
            for i in range(len(x)):
                self.list[i].update(x[i])
            self.n += 1
            self.type = 'list'
        elif isinstance(x, list):
            assert False, 'only tuple or array or tensor'

        elif len(x.shape)==1:
            self.n += len(x)
            self.total += x.sum()
        else:
            assert x.shape == ()
            self.n += 1
            self.total += x

    def val(self):
        if self.type == 'scalar':
            return  self.total/self.n
        else:
            return (tuple([l.val() for l in self.list]))


class SplitComb():
    def __init__(self, config):
        margin = np.array(config.parameters['margin'])
        side_len = np.array(config.parameters['crop_size']) - margin * 2
        stride = config.parameters['seg_stride']
        self.pad_mode = config.parameters['pad_mode']
        self.pad_value = config.parameters['pad_value']

        if isinstance(side_len, int):
            side_len = [side_len] * 3
        if isinstance(stride, int):
            stride = [stride] * 3
        if isinstance(margin, int):
            margin = [margin] * 3

        self.side_len = np.array(side_len)
        self.stride = np.array(stride)
        self.margin = np.array(margin)

    @staticmethod
    def getse(izhw, nzhw, crop_size, side_len, shape_post):
        se = []
        for i, n, crop, side, shape in zip(izhw, nzhw, crop_size, side_len, shape_post):
            if i == n - 1 and i > 0:
                e = shape
                s = e - crop
            else:
                s = i * side
                e = s + crop
            se += [s, e]
        return se

    @staticmethod
    def getse2(izhw, nzhw, crop_size, side_len, shape_len):
        se = []
        for i, n, crop, side, shape in zip(izhw, nzhw, crop_size, side_len, shape_len):
            if i == n - 1 and i > 0:
                e = shape
                s = e - side
            else:
                s = i * side
                e = s + side
            se += [s, e]
        return se

    def split(self, data, side_len=None, margin=None):
        if side_len is None:
            side_len = self.side_len
        if margin is None:
            margin = self.margin
        crop_size = side_len + margin * 2

        assert (np.all(side_len > margin))

        splits = []
        _, z, h, w = data.shape

        nz = int(np.ceil(float(z) / side_len[0]))
        nh = int(np.ceil(float(h) / side_len[1]))
        nw = int(np.ceil(float(w) / side_len[2]))

        shape_pre = [z, h, w]

        pad = [[0, 0],
               [margin[0], np.max([margin[0], crop_size[0] - z - margin[0]])],
               [margin[1], np.max([margin[1], crop_size[1] - h - margin[1]])],
               [margin[2], np.max([margin[2], crop_size[2] - w - margin[2]])]]
        #         print(data.shape)
        #         print(side_len[1])
        #         print(side_len[1]-h-margin[1])
        #         print(pad)
        if self.pad_mode == 'constant':
            data = mypad(data, pad, self.pad_value)
        else:
            data = np.pad(data, pad, self.pad_mode)
        shape_post = list(data.shape[1:])
        shapes = np.array([shape_pre, shape_post])
        self.shapes = shapes
        splits = []
        id = 0
        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz, ez, sh, eh, sw, ew = self.getse([iz, ih, iw], [nz, nh, nw], crop_size, side_len, shape_post)
                    splits.append(data[:, sz:ez, sh:eh, sw:ew])
                    id += 1
        splits = (np.array(splits))
        return splits, shapes, pad

    def combine(self, output, shapes=None, side_len=None, stride=None, margin=None):

        if side_len is None:
            side_len = self.side_len
        if stride is None:
            stride = self.stride
        if margin is None:
            margin = self.margin
        if shapes is None:
            shape = self.shapes

        shape_pre, shape_post = shapes
        shape_pre = shape_pre.numpy()
        z, h, w = shape_pre
        nz = int(np.ceil(float(z) / side_len[0]))
        nh = int(np.ceil(float(h) / side_len[1]))
        nw = int(np.ceil(float(w) / side_len[2]))

        assert (np.all(side_len % stride == 0))
        assert (np.all(margin % stride == 0))

        newshape = (np.array([z, h, w]) / stride).astype(np.int)
        side_len = (self.side_len / stride).astype(np.int)
        margin = (self.margin / stride).astype(np.int)
        crop_size = side_len + margin * 2

        splits = []
        for i in range(len(output)):
            splits.append(output[i])

        if isinstance(output[0], torch.Tensor):
            occur = torch.zeros((1,
                                 nz * side_len[0],
                                 nh * side_len[1],
                                 nw * side_len[2]), dtype=output[0].dtype, device=output[0].device)
            output = torch.ones((splits[0].shape[0],
                                 nz * side_len[0],
                                 nh * side_len[1],
                                 nw * side_len[2]), dtype=output[0].dtype, device=output[0].device)
        else:
            occur = np.zeros((
                1,
                nz * side_len[0],
                nh * side_len[1],
                nw * side_len[2]), output[0].dtype)
            #
            output = -1000000 * np.ones((
                splits[0].shape[0],
                nz * side_len[0],
                nh * side_len[1],
                nw * side_len[2]), output[0].dtype)
        #         print(output.shape)
        idx = 0
        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz, ez, sh, eh, sw, ew = self.getse2([iz, ih, iw], [nz, nh, nw], crop_size, side_len, shape_pre)
                    #                     print(sz, ez, sh, eh, sw, ew)
                    split = splits[idx][:, margin[0]:margin[0] + side_len[0], margin[1]:margin[1] + side_len[1],
                            margin[2]:margin[2] + side_len[2]]

                    output[:, sz:ez, sh:eh, sw:ew] += split
                    occur[:, sz:ez, sh:eh, sw:ew] += 1
                    idx += 1

        return output[:, :newshape[0], :newshape[1], :newshape[2]] / occur[:, :newshape[0], :newshape[1], :newshape[2]]

def read_file(file):
    name = []
    with open(file, 'r') as f:
        content = f.readlines()
    for c in content:
        c = c.strip()
        name.append(c)
    return name

def tiff2array(file):
    """
    :param file: .tif file
    :return: numpy.array
    """
    tif = TIFF.open(file, mode="r")
    ims = np.array(list(tif.iter_images()))
    return ims


def roi_data(img, mask, roi, roi_margin):
    _, z, h, w = img.shape
    z_l, z_h, h_l, h_h, w_l, w_h = roi

    zl = np.max([0, z_l - roi_margin[0]])
    zh = np.min([z, z_h + roi_margin[0]])

    hl = np.max([0, h_l - roi_margin[1]])
    hh = np.min([h, h_h + roi_margin[1]])

    wl = np.max([0, w_l - roi_margin[2]])
    wh = np.min([w, w_h + roi_margin[2]])

    img_roi = img[:, zl:zh, hl:hh, wl:wh]
    mask_roi = mask[:, zl:zh, hl:hh, wl:wh]
    return img_roi, mask_roi