# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import copy
import numpy as np
import math
import re
import sys
import argparse
import string
from copy import deepcopy
import logging
import random
import os

###### test ########
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class DetTransform:
    """检测数据处理基类
    """

    def __init__(self):
        pass


class Compose(DetTransform):
    """根据数据预处理/增强列表对输入数据进行操作。
       所有操作的输入图像流形状均是[H, W, C]，其中H为图像高，W为图像宽，C为图像通道数。
    Args:
        transforms (list): 数据预处理/增强列表。
    Raises:
        TypeError: 形参数据类型不满足需求。
        ValueError: 数据长度不匹配。
    """

    def __init__(self, transforms):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        if len(transforms) < 1:
            raise ValueError('The length of transforms ' + \
                            'must be equal or larger than 1!')
        self.transforms = transforms
        self.batch_transforms = None
        self.use_mixup = False
        self.data_type = np.uint8
        self.to_rgb = True
        for t in self.transforms:
            if type(t).__name__ == 'MixupImage':
                self.use_mixup = True

    def __call__(self, im, im_info=None, label_info=None):
        """
        Args:
            im (str/np.ndarray): 图像路径/图像np.ndarray数据。
            im_info (dict): 存储与图像相关的信息，dict中的字段如下：
                - im_id (np.ndarray): 图像序列号，形状为(1,)。
                - image_shape (np.ndarray): 图像原始大小，形状为(2,)，
                                        image_shape[0]为高，image_shape[1]为宽。
                - mixup (list): list为[im, im_info, label_info]，分别对应
                                与当前图像进行mixup的图像np.ndarray数据、图像相关信息、标注框相关信息；
                                注意，当前epoch若无需进行mixup，则无该字段。
            label_info (dict): 存储与标注框相关的信息，dict中的字段如下：
                - gt_bbox (np.ndarray): 真实标注框坐标[x1, y1, x2, y2]，形状为(n, 4)，
                                   其中n代表真实标注框的个数。
                - gt_class (np.ndarray): 每个真实标注框对应的类别序号，形状为(n, 1)，
                                    其中n代表真实标注框的个数。
                - gt_score (np.ndarray): 每个真实标注框对应的混合得分，形状为(n, 1)，
                                    其中n代表真实标注框的个数。
                - gt_poly (list): 每个真实标注框内的多边形分割区域，每个分割区域由点的x、y坐标组成，
                                  长度为n，其中n代表真实标注框的个数。
                - is_crowd (np.ndarray): 每个真实标注框中是否是一组对象，形状为(n, 1)，
                                    其中n代表真实标注框的个数。
                - difficult (np.ndarray): 每个真实标注框中的对象是否为难识别对象，形状为(n, 1)，
                                     其中n代表真实标注框的个数。
        Returns:
            tuple: 根据网络所需字段所组成的tuple；
                字段由transforms中的最后一个数据预处理操作决定。
        """

        def decode_image(im_file, im_info, label_info, input_channel=3):
            if im_info is None:
                im_info = dict()
            if isinstance(im_file, np.ndarray):
                if len(im_file.shape) != 3:
                    raise Exception(
                        "im should be 3-dimensions, but now is {}-dimensions".
                        format(len(im_file.shape)))
                im = im_file
            else:
                try:
                    if input_channel == 3:
                        im = cv2.imread(im_file, cv2.IMREAD_ANYDEPTH |
                                        cv2.IMREAD_ANYCOLOR | cv2.IMREAD_COLOR)
                    else:
                        im = cv2.imread(im_file, cv2.IMREAD_ANYDEPTH |
                                        cv2.IMREAD_ANYCOLOR)
                        if im.ndim < 3:
                            im = np.expand_dims(im, axis=-1)
                except:
                    raise TypeError('Can\'t read The image file {}!'.format(
                        im_file))
            self.data_type = im.dtype
            im = im.astype('float32')
            if input_channel == 3 and self.to_rgb:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            # make default im_info with [h, w, 1]
            im_info['im_resize_info'] = np.array(
                [im.shape[0], im.shape[1], 1.], dtype=np.float32)
            im_info['image_shape'] = np.array([im.shape[0],
                                               im.shape[1]]).astype('int32')
            if not self.use_mixup:
                if 'mixup' in im_info:
                    del im_info['mixup']
            # decode mixup image
            if 'mixup' in im_info:
                im_info['mixup'] = \
                  decode_image(im_info['mixup'][0],
                               im_info['mixup'][1],
                               im_info['mixup'][2],
                               input_channel)
            if label_info is None:
                return (im, im_info)
            else:
                return (im, im_info, label_info)

        input_channel = getattr(self, 'input_channel', 3)
        outputs = decode_image(im, im_info, label_info, input_channel)
        im = outputs[0]
        im_info = outputs[1]
        if len(outputs) == 3:
            label_info = outputs[2]
        for op in self.transforms:
            if im is None:
                return None
            if isinstance(op, DetTransform):
                if op.__class__.__name__ == 'RandomDistort':
                    op.to_rgb = self.to_rgb
                    op.data_type = self.data_type
                outputs = op(im, im_info, label_info)
                im = outputs[0]
        return outputs

    def add_augmenters(self, augmenters):
        if not isinstance(augmenters, list):
            raise Exception(
                "augmenters should be list type in func add_augmenters()")
        transform_names = [type(x).__name__ for x in self.transforms]
        for aug in augmenters:
            if type(aug).__name__ in transform_names:
                logging.error(
                    "{} is already in ComposedTransforms, need to remove it from add_augmenters().".
                    format(type(aug).__name__))
        self.transforms = augmenters + self.transforms


class ResizeByShort(DetTransform):
    """根据图像的短边调整图像大小（resize）。

    1. 获取图像的长边和短边长度。
    2. 根据短边与short_size的比例，计算长边的目标长度，
       此时高、宽的resize比例为short_size/原图短边长度。
       若short_size为数组，则随机从该数组中挑选一个数值
       作为short_size。
    3. 如果max_size>0，调整resize比例：
       如果长边的目标长度>max_size，则高、宽的resize比例为max_size/原图长边长度。
    4. 根据调整大小的比例对图像进行resize。

    Args:
        short_size (int|list): 短边目标长度。默认为800。
        max_size (int): 长边目标长度的最大限制。默认为1333。

     Raises:
        TypeError: 形参数据类型不满足需求。
    """

    def __init__(self, short_size=800, max_size=1333):
        self.max_size = int(max_size)
        if not (isinstance(short_size, int) or isinstance(short_size, list)):
            raise TypeError(
                "Type of short_size is invalid. Must be Integer or List, now is {}".
                format(type(short_size)))
        self.short_size = short_size
        if not (isinstance(self.max_size, int)):
            raise TypeError("max_size: input type is invalid.")

    def __call__(self, im, im_info=None, label_info=None):
        """
        Args:
            im (numnp.ndarraypy): 图像np.ndarray数据。
            im_info (dict, 可选): 存储与图像相关的信息。
            label_info (dict, 可选): 存储与标注框相关的信息。

        Returns:
            tuple: 当label_info为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                   当label_info不为空时，返回的tuple为(im, im_info, label_info)，分别对应图像np.ndarray数据、
                   存储与标注框相关信息的字典。
                   其中，im_info更新字段为：
                       - im_resize_info (np.ndarray): resize后的图像高、resize后的图像宽、resize后的图像相对原始图的缩放比例
                                                 三者组成的np.ndarray，形状为(3,)。

        Raises:
            TypeError: 形参数据类型不满足需求。
            ValueError: 数据长度不匹配。
        """
        if im_info is None:
            im_info = dict()
        if not isinstance(im, np.ndarray):
            raise TypeError("ResizeByShort: image type is not numpy.")
        if len(im.shape) != 3:
            raise ValueError('ResizeByShort: image is not 3-dimensional.')
        im_short_size = min(im.shape[0], im.shape[1])
        im_long_size = max(im.shape[0], im.shape[1])
        if isinstance(self.short_size, list):
            # Case for multi-scale training
            selected_size = random.choice(self.short_size)
        else:
            selected_size = self.short_size
        scale = float(selected_size) / im_short_size
        if self.max_size > 0 and np.round(scale *
                                          im_long_size) > self.max_size:
            scale = float(self.max_size) / float(im_long_size)
        resized_width = int(round(im.shape[1] * scale))
        resized_height = int(round(im.shape[0] * scale))
        im_resize_info = [resized_height, resized_width, scale]
        im = cv2.resize(
            im, (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR)
        if im.ndim < 3:
            im = np.expand_dims(im, axis=-1)
        im_info['im_resize_info'] = np.array(im_resize_info).astype(np.float32)
        if label_info is None:
            return (im, im_info)
        else:
            return (im, im_info, label_info)


class Padding(DetTransform):
    """1.将图像的长和宽padding至coarsest_stride的倍数。如输入图像为[300, 640],
       `coarest_stride`为32，则由于300不为32的倍数，因此在图像最右和最下使用0值
       进行padding，最终输出图像为[320, 640]。
       2.或者，将图像的长和宽padding到target_size指定的shape，如输入的图像为[300，640]，
         a. `target_size` = 960，在图像最右和最下使用0值进行padding，最终输出
            图像为[960, 960]。
         b. `target_size` = [640, 960]，在图像最右和最下使用0值进行padding，最终
            输出图像为[640, 960]。

    1. 如果coarsest_stride为1，target_size为None则直接返回。
    2. 获取图像的高H、宽W。
    3. 计算填充后图像的高H_new、宽W_new。
    4. 构建大小为(H_new, W_new, 3)像素值为0的np.ndarray，
       并将原图的np.ndarray粘贴于左上角。

    Args:
        coarsest_stride (int): 填充后的图像长、宽为该参数的倍数，默认为1。
        target_size (int|list|tuple): 填充后的图像长、宽，默认为None，coarset_stride优先级更高。

    Raises:
        TypeError: 形参`target_size`数据类型不满足需求。
        ValueError: 形参`target_size`为(list|tuple)时，长度不满足需求。
    """

    def __init__(self, coarsest_stride=1, target_size=None):
        self.coarsest_stride = coarsest_stride
        if target_size is not None:
            if not isinstance(target_size, int):
                if not isinstance(target_size, tuple) and not isinstance(
                        target_size, list):
                    raise TypeError(
                        "Padding: Type of target_size must in (int|list|tuple)."
                    )
                elif len(target_size) != 2:
                    raise ValueError(
                        "Padding: Length of target_size must equal 2.")
        self.target_size = target_size

    def __call__(self, im, im_info=None, label_info=None):
        """
        Args:
            im (numnp.ndarraypy): 图像np.ndarray数据。
            im_info (dict, 可选): 存储与图像相关的信息。
            label_info (dict, 可选): 存储与标注框相关的信息。

        Returns:
            tuple: 当label_info为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                   当label_info不为空时，返回的tuple为(im, im_info, label_info)，分别对应图像np.ndarray数据、
                   存储与标注框相关信息的字典。

        Raises:
            TypeError: 形参数据类型不满足需求。
            ValueError: 数据长度不匹配。
            ValueError: coarsest_stride，target_size需有且只有一个被指定。
            ValueError: target_size小于原图的大小。
        """
        if im_info is None:
            im_info = dict()
        if not isinstance(im, np.ndarray):
            raise TypeError("Padding: image type is not numpy.")
        if len(im.shape) != 3:
            raise ValueError('Padding: image is not 3-dimensional.')
        im_h, im_w, im_c = im.shape[:]

        if isinstance(self.target_size, int):
            padding_im_h = self.target_size
            padding_im_w = self.target_size
        elif isinstance(self.target_size, list) or isinstance(self.target_size,
                                                              tuple):
            padding_im_w = self.target_size[0]
            padding_im_h = self.target_size[1]
        elif self.coarsest_stride > 0:
            padding_im_h = int(
                np.ceil(im_h / self.coarsest_stride) * self.coarsest_stride)
            padding_im_w = int(
                np.ceil(im_w / self.coarsest_stride) * self.coarsest_stride)
        else:
            raise ValueError(
                "coarsest_stridei(>1) or target_size(list|int) need setting in Padding transform"
            )
        pad_height = padding_im_h - im_h
        pad_width = padding_im_w - im_w
        if pad_height < 0 or pad_width < 0:
            raise ValueError(
                'the size of image should be less than target_size, but the size of image ({}, {}), is larger than target_size ({}, {})'
                .format(im_w, im_h, padding_im_w, padding_im_h))
        padding_im = np.zeros(
            (padding_im_h, padding_im_w, im_c), dtype=np.float32)
        padding_im[:im_h, :im_w, :] = im
        if label_info is None:
            return (padding_im, im_info)
        else:
            return (padding_im, im_info, label_info)


class Maskrcnn_Normalize(DetTransform):
    """对图像进行标准化。

    1.像素值减去min_val
    2.像素值除以(max_val-min_val)
    3.对图像进行减均值除以标准差操作。

    Args:
        mean (list): 图像数据集的均值。默认值[0.5, 0.5, 0.5]。
        std (list): 图像数据集的标准差。默认值[0.5, 0.5, 0.5]。
        min_val (list): 图像数据集的最小值。默认值[0, 0, 0]。
        max_val (list): 图像数据集的最大值。默认值[255.0, 255.0, 255.0]。

    Raises:
        TypeError: 形参数据类型不满足需求。
    """

    def __init__(self,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 min_val=[0, 0, 0],
                 max_val=[255.0, 255.0, 255.0]):
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

        if not (isinstance(self.mean, list) and isinstance(self.std, list)):
            raise TypeError("NormalizeImage: input type is invalid.")

        if not (isinstance(self.min_val, list) and isinstance(self.max_val,
                                                              list)):
            raise ValueError("{}: input type is invalid.".format(self))

        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise TypeError('NormalizeImage: std is invalid!')

    def __call__(self, im, im_info=None, label_info=None):
        """
        Args:
            im (numnp.ndarraypy): 图像np.ndarray数据。
            im_info (dict, 可选): 存储与图像相关的信息。
            label_info (dict, 可选): 存储与标注框相关的信息。

        Returns:
            tuple: 当label_info为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                   当label_info不为空时，返回的tuple为(im, im_info, label_info)，分别对应图像np.ndarray数据、
                   存储与标注框相关信息的字典。
        """
        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]
        im = normalize(im, mean, std, self.min_val, self.max_val)
        if label_info is None:
            return (im, im_info)
        else:
            return (im, im_info, label_info)


class ArrangeMaskRCNN(DetTransform):
    """获取MaskRCNN模型训练/验证/预测所需信息。

    Args:
        mode (str): 指定数据用于何种用途，取值范围为['train', 'eval', 'test', 'quant']。

    Raises:
        ValueError: mode的取值不在['train', 'eval', 'test', 'quant']之内。
    """

    def __init__(self, mode=None):
        mode = 'quant'
        if mode not in ['train', 'eval', 'test', 'quant']:
            raise ValueError(
                "mode must be in ['train', 'eval', 'test', 'quant']!")
        self.mode = mode

    def __call__(self, im, im_info=None, label_info=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (dict, 可选): 存储与图像相关的信息。
            label_info (dict, 可选): 存储与标注框相关的信息。

        Returns:
            tuple: 当mode为'train'时，返回(im, im_resize_info, gt_bbox, gt_class, is_crowd, gt_masks)，分别对应
                图像np.ndarray数据、图像相当对于原图的resize信息、真实标注框、真实标注框对应的类别、真实标注框内是否是一组对象、
                真实分割区域；当mode为'eval'时，返回(im, im_resize_info, im_id, im_shape)，分别对应图像np.ndarray数据、
                图像相当对于原图的resize信息、图像id、图像大小信息；当mode为'test'或'quant'时，返回(im, im_resize_info, im_shape)，
                分别对应图像np.ndarray数据、图像相当对于原图的resize信息、图像大小信息。

        Raises:
            TypeError: 形参数据类型不满足需求。
            ValueError: 数据长度不匹配。
        """
        im = permute(im, False)
        if self.mode == 'train':
            if im_info is None or label_info is None:
                raise TypeError(
                    'Cannot do ArrangeTrainMaskRCNN! ' +
                    'Becasuse the im_info and label_info can not be None!')
            if len(label_info['gt_bbox']) != len(label_info['gt_class']):
                raise ValueError("gt num mismatch: bbox and class.")
            im_resize_info = im_info['im_resize_info']
            gt_bbox = label_info['gt_bbox']
            gt_class = label_info['gt_class']
            is_crowd = label_info['is_crowd']
            assert 'gt_poly' in label_info
            segms = label_info['gt_poly']
            if len(segms) != 0:
                assert len(segms) == is_crowd.shape[0]
            gt_masks = []
            valid = True
            for i in range(len(segms)):
                segm = segms[i]
                gt_segm = []
                if is_crowd[i]:
                    gt_segm.append([[0, 0]])
                else:
                    for poly in segm:
                        if len(poly) == 0:
                            valid = False
                            break
                        gt_segm.append(np.array(poly).reshape(-1, 2))
                if (not valid) or len(gt_segm) == 0:
                    break
                gt_masks.append(gt_segm)
            outputs = (im, im_resize_info, gt_bbox, gt_class, is_crowd,
                       gt_masks)
        else:
            if im_info is None:
                raise TypeError('Cannot do ArrangeMaskRCNN! ' +
                                'Becasuse the im_info can not be None!')
            im_resize_info = im_info['im_resize_info']
            im_shape = np.array(
                (im_info['image_shape'][0], im_info['image_shape'][1], 1),
                dtype=np.float32)
            if self.mode == 'eval':
                im_id = im_info['im_id']
                outputs = (im, im_resize_info, im_id, im_shape)
            else:
                outputs = (im, im_resize_info, im_shape)
        return outputs


def generate_minibatch(batch_data, label_padding_value=255, mapper=None):
    if mapper is not None and mapper.batch_transforms is not None:
        for op in mapper.batch_transforms:
            batch_data = op(batch_data)
    # if batch_size is 1, do not pad the image
    if len(batch_data) == 1:
        return batch_data
    width = [data[0].shape[2] for data in batch_data]
    height = [data[0].shape[1] for data in batch_data]
    # if the sizes of images in a mini-batch are equal,
    # do not pad the image
    if len(set(width)) == 1 and len(set(height)) == 1:
        return batch_data
    max_shape = np.array([data[0].shape for data in batch_data]).max(axis=0)
    padding_batch = []
    for data in batch_data:
        # pad the image to a same size
        im_c, im_h, im_w = data[0].shape[:]
        padding_im = np.zeros(
            (im_c, max_shape[1], max_shape[2]), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = data[0]
        if len(data) > 1:
            if isinstance(data[1], np.ndarray):
                if data[1].ndim == 3:
                    # padding the image and label of segmentation during the training
                    # the data[1] of segmentation is a image array, so data[1].ndim is 3.
                    padding_label = np.zeros(
                        (1, max_shape[1], max_shape[2]
                         )).astype('int64') + label_padding_value
                    _, label_h, label_w = data[1].shape
                    padding_label[:, :label_h, :label_w] = data[1]
                    padding_batch.append((padding_im, padding_label))
                else:
                    # padding the image of detection
                    padding_batch.append((padding_im, ) + tuple(data[1:]))
            elif isinstance(data[1], (list, tuple)):
                # padding the image and insert 'padding' into `im_info` of segmentation
                # during evaluating or inferring phase.
                if len(data[1]) == 0 or 'padding' not in [
                        data[1][i][0] for i in range(len(data[1]))
                ]:
                    data[1].append(('padding', [im_h, im_w]))
                padding_batch.append((padding_im, ) + tuple(data[1:]))
            else:
                # padding the image of classification during the train/eval  phase
                # padding the image of classification during the trainging
                # and evaluating phase
                padding_batch.append((padding_im, ) + tuple(data[1:]))
        else:
            # padding the image of classification during the infering phase
            padding_batch.append((padding_im, ))
    return padding_batch


def normalize(im, mean, std, min_value=[0, 0, 0], max_value=[255, 255, 255]):
    # Rescaling (min-max normalization)
    range_value = [max_value[i] - min_value[i] for i in range(len(max_value))]
    im = (im - min_value) / range_value

    # Standardization (Z-score Normalization)
    im -= mean
    im /= std
    return im.astype('float32')


def permute(im, to_bgr=False):
    im = np.swapaxes(im, 1, 2)
    im = np.swapaxes(im, 1, 0)
    if to_bgr:
        im = im[[2, 1, 0], :, :]
    return im


def maskrcnn_postprocess(res, batch_size, num_classes, mask_head_resolution, labels):
    clsid2catid = dict({i: i for i in range(num_classes)})
    xywh_results = bbox2out([res], clsid2catid)
    segm_results = mask2out([res], clsid2catid, mask_head_resolution)
    preds = [[] for i in range(batch_size)]
    import pycocotools.mask as mask_util
    for index, xywh_res in enumerate(xywh_results):
        image_id = xywh_res['image_id']
        del xywh_res['image_id']
        xywh_res['mask'] = mask_util.decode(segm_results[index][
            'segmentation'])
        xywh_res['category'] = labels[xywh_res['category_id']]
        preds[image_id].append(xywh_res)

    return preds


def offset_to_lengths(lod):
    offset = lod
    lengths = [
        offset[i + 1] - offset[i] for i in range(len(offset) - 1)
    ]
    return [lengths]


def bbox2out(results, clsid2catid, is_bbox_normalized=False):
    """
    Args:
        results: request a dict, should include: `bbox`, `im_id`,
                 if is_bbox_normalized=True, also need `im_shape`.
        clsid2catid: class id to category id map of COCO2017 dataset.
        is_bbox_normalized: whether or not bbox is normalized.
    """
    xywh_res = []
    for t in results:
        bboxes = t['bbox'][0]
        lengths = t['bbox'][1][0]
        im_ids = np.array(t['im_id'][0]).flatten()
        if bboxes.shape == (1, 1) or bboxes is None:
            continue

        k = 0
        for i in range(len(lengths)):
            num = lengths[i]
            im_id = int(im_ids[i])
            for j in range(num):
                dt = bboxes[k]
                clsid, score, xmin, ymin, xmax, ymax = dt.tolist()
                catid = (clsid2catid[int(clsid)])

                if is_bbox_normalized:
                    xmin, ymin, xmax, ymax = \
                            clip_bbox([xmin, ymin, xmax, ymax])
                    w = xmax - xmin
                    h = ymax - ymin
                    im_shape = t['im_shape'][0][i].tolist()
                    im_height, im_width = int(im_shape[0]), int(im_shape[1])
                    xmin *= im_width
                    ymin *= im_height
                    w *= im_width
                    h *= im_height
                else:
                    w = xmax - xmin + 1
                    h = ymax - ymin + 1

                bbox = [xmin, ymin, w, h]
                coco_res = {
                    'image_id': im_id,
                    'category_id': catid,
                    'bbox': bbox,
                    'score': score
                }
                xywh_res.append(coco_res)
                k += 1
    return xywh_res


def mask2out(results, clsid2catid, resolution, thresh_binarize=0.5):
    # matplotlib.use() must be called *before* pylab, matplotlib.pyplot,
    # or matplotlib.backends is imported for the first time
    # pycocotools import matplotlib
    import matplotlib
    matplotlib.use('Agg')
    import pycocotools.mask as mask_util
    scale = (resolution + 2.0) / resolution

    segm_res = []

    # for each batch
    for t in results:
        bboxes = t['bbox'][0]

        lengths = t['bbox'][1][0]
        im_ids = np.array(t['im_id'][0])
        if bboxes.shape == (1, 1) or bboxes is None:
            continue
        if len(bboxes.tolist()) == 0:
            continue

        masks = t['mask'][0]

        s = 0
        # for each sample
        for i in range(len(lengths)):
            num = lengths[i]
            im_id = int(im_ids[i][0])
            im_shape = t['im_shape'][0][i]

            bbox = bboxes[s:s + num][:, 2:]
            clsid_scores = bboxes[s:s + num][:, 0:2]
            mask = masks[s:s + num]
            s += num

            im_h = int(im_shape[0])
            im_w = int(im_shape[1])

            expand_bbox = expand_boxes(bbox, scale)
            expand_bbox = expand_bbox.astype(np.int32)

            padded_mask = np.zeros(
                (resolution + 2, resolution + 2), dtype=np.float32)

            for j in range(num):
                xmin, ymin, xmax, ymax = expand_bbox[j].tolist()
                clsid, score = clsid_scores[j].tolist()
                clsid = int(clsid)
                padded_mask[1:-1, 1:-1] = mask[j, clsid, :, :]

                catid = clsid2catid[clsid]

                w = xmax - xmin + 1
                h = ymax - ymin + 1
                w = np.maximum(w, 1)
                h = np.maximum(h, 1)

                resized_mask = cv2.resize(padded_mask, (w, h))
                resized_mask = np.array(
                    resized_mask > thresh_binarize, dtype=np.uint8)
                im_mask = np.zeros((im_h, im_w), dtype=np.uint8)

                x0 = min(max(xmin, 0), im_w)
                x1 = min(max(xmax + 1, 0), im_w)
                y0 = min(max(ymin, 0), im_h)
                y1 = min(max(ymax + 1, 0), im_h)

                im_mask[y0:y1, x0:x1] = resized_mask[(y0 - ymin):(y1 - ymin), (
                    x0 - xmin):(x1 - xmin)]
                segm = mask_util.encode(
                    np.array(
                        im_mask[:, :, np.newaxis], order='F'))[0]
                catid = clsid2catid[clsid]
                segm['counts'] = segm['counts'].decode('utf8')
                coco_res = {
                    'image_id': im_id,
                    'category_id': catid,
                    'segmentation': segm,
                    'score': score
                }
                segm_res.append(coco_res)
    return segm_res


def expand_boxes(boxes, scale):
    """
    Expand an array of boxes by a given scale.
    """
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = np.zeros(boxes.shape)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half

    return boxes_exp


def clip_bbox(bbox):
    xmin = max(min(bbox[0], 1.), 0.)
    ymin = max(min(bbox[1], 1.), 0.)
    xmax = max(min(bbox[2], 1.), 0.)
    ymax = max(min(bbox[3], 1.), 0.)
    return xmin, ymin, xmax, ymax


class DetResizeForTest(object):
    def __init__(self, **kwargs):
        super(DetResizeForTest, self).__init__()
        self.resize_type = 0
        if 'image_shape' in kwargs:
            self.image_shape = kwargs['image_shape']
            self.resize_type = 1
        elif 'limit_side_len' in kwargs:
            self.limit_side_len = kwargs['limit_side_len']
            self.limit_type = kwargs.get('limit_type', 'min')
        elif 'resize_short' in kwargs:
            self.limit_side_len = 736
            self.limit_type = 'min'
        else:
            self.resize_type = 2
            self.resize_long = kwargs.get('resize_long', 960)

    def __call__(self, data):
        img = deepcopy(data)
        src_h, src_w, _ = img.shape

        if self.resize_type == 0:
            img, [ratio_h, ratio_w] = self.resize_image_type0(img)
        elif self.resize_type == 2:
            img, [ratio_h, ratio_w] = self.resize_image_type2(img)
        else:
            img, [ratio_h, ratio_w] = self.resize_image_type1(img)

        return img

    def resize_image_type1(self, img):
        resize_h, resize_w = self.image_shape
        ori_h, ori_w = img.shape[:2]  # (h, w, c)
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        return img, [ratio_h, ratio_w]

    def resize_image_type0(self, img):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        limit_side_len = self.limit_side_len
        h, w, _ = img.shape

        # limit the max side
        if self.limit_type == 'max':
            if max(h, w) > limit_side_len:
                if h > w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        else:
            if min(h, w) < limit_side_len:
                if h < w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        resize_h = int(round(resize_h / 32) * 32)
        resize_w = int(round(resize_w / 32) * 32)

        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            img = cv2.resize(img, (int(resize_w), int(resize_h)))
        except:
            # print(img.shape, resize_w, resize_h)
            sys.exit(0)
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        # return img, np.array([h, w])
        return img, [ratio_h, ratio_w]

    def resize_image_type2(self, img):
        h, w, _ = img.shape

        resize_w = w
        resize_h = h

        # Fix the longer side
        if resize_h > resize_w:
            ratio = float(self.resize_long) / resize_h
        else:
            ratio = float(self.resize_long) / resize_w

        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        image = img[..., ::-1]
        draw_img_save = "/paddle/inference_results/"
        if not os.path.exists(draw_img_save):
            os.makedirs(draw_img_save)
        cv2.imwrite(
            os.path.join(draw_img_save, os.path.basename('resize.jpg')),
            image[:, :, ::-1])

        return img, [ratio_h, ratio_w]


class BaseRecLabelDecode(object):
    """ Convert between text-label and text-index """

    def __init__(self, config):
        support_character_type = [
            'ch', 'en', 'EN_symbol', 'french', 'german', 'japan', 'korean',
            'it', 'xi', 'pu', 'ru', 'ar', 'ta', 'ug', 'fa', 'ur', 'rs', 'oc',
            'rsc', 'bg', 'uk', 'be', 'te', 'ka', 'chinese_cht', 'hi', 'mr',
            'ne', 'EN'
        ]
        character_type = config['character_type']
        character_dict_path = config['character_dict_path']
        use_space_char = True
        assert character_type in support_character_type, "Only {} are supported now but get {}".format(
            support_character_type, character_type)

        self.beg_str = "sos"
        self.end_str = "eos"

        if character_type == "en":
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        elif character_type == "EN_symbol":
            # same with ASTER setting (use 94 char).
            self.character_str = string.printable[:-6]
            dict_character = list(self.character_str)
        elif character_type in support_character_type:
            self.character_str = ""
            assert character_dict_path is not None, "character_dict_path should not be None when character_type is {}".format(
                character_type)
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str += line
            if use_space_char:
                self.character_str += " "
            dict_character = list(self.character_str)

        else:
            raise NotImplementedError
        self.character_type = character_type
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        score_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                            batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][
                    idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text))
            score_list.append(conf_list)
        return result_list, score_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(
            self,
            config,
            #character_dict_path=None,
            #character_type='ch',
            #use_space_char=False,
            **kwargs):
        super(CTCLabelDecode, self).__init__(config)

    def __call__(self, preds, label=None, *args, **kwargs):
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character


class CharacterOps(object):
    """ Convert between text-label and text-index """

    def __init__(self, config):
        self.character_type = config['character_type']
        self.loss_type = config['loss_type']
        if self.character_type == "en":
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        elif self.character_type == "ch":
            character_dict_path = config['character_dict_path']
            self.character_str = ""
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str += line
            dict_character = list(self.character_str)
            # print(self.character_str)
            # print(len(self.character_str))
            # print(len(dict_character))
        elif self.character_type == "en_sensitive":
            # same with ASTER setting (use 94 char).
            self.character_str = string.printable[:-6]
            dict_character = list(self.character_str)
        else:
            self.character_str = None
        assert self.character_str is not None, \
            "Nonsupport type of the character: {}".format(self.character_str)
        self.beg_str = "sos"
        self.end_str = "eos"
        if self.loss_type == "attention":
            dict_character = [self.beg_str, self.end_str] + dict_character
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character
        # print(len(self.character))

    def encode(self, text):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        if self.character_type == "en":
            text = text.lower()

        text_list = []
        for char in text:
            if char not in self.dict:
                continue
            text_list.append(self.dict[char])
        text = np.array(text_list)
        return text

    def decode(self, text_index, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        char_list = []
        char_num = self.get_char_num()

        if self.loss_type == "attention":
            beg_idx = self.get_beg_end_flag_idx("beg")
            end_idx = self.get_beg_end_flag_idx("end")
            ignored_tokens = [beg_idx, end_idx]
        else:
            ignored_tokens = [char_num]

        for idx in range(len(text_index)):
            if text_index[idx] in ignored_tokens:
                continue
            if is_remove_duplicate:
                if idx > 0 and text_index[idx - 1] == text_index[idx]:
                    continue
            char_list.append(self.character[text_index[idx]])
        text = ''.join(char_list)
        return text

    def get_char_num(self):
        return len(self.character)

    def get_beg_end_flag_idx(self, beg_or_end):
        if self.loss_type == "attention":
            if beg_or_end == "beg":
                idx = np.array(self.dict[self.beg_str])
            elif beg_or_end == "end":
                idx = np.array(self.dict[self.end_str])
            else:
                assert False, "Unsupport type %s in get_beg_end_flag_idx"\
                    % beg_or_end
            return idx
        else:
            err = "error in get_beg_end_flag_idx when using the loss %s"\
                % (self.loss_type)
            assert False, err


class OCRReader(object):
    def __init__(self,
                 algorithm="CRNN",
                 image_shape=[3, 32, 320],
                 char_type="ch",
                 batch_num=1,
                 char_dict_path="./ppocr_keys_v1.txt"):
        self.rec_image_shape = image_shape
        self.character_type = char_type
        self.rec_batch_num = batch_num
        char_ops_params = {}
        char_ops_params["character_type"] = char_type
        char_ops_params["character_dict_path"] = char_dict_path
        char_ops_params['loss_type'] = 'ctc'
        self.char_ops = CharacterOps(char_ops_params)
        self.label_ops = CTCLabelDecode(char_ops_params)

    def get_sorted_boxes(self, img, points):
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        # logger.info('left is: {}'.format(left))
        # logger.info('right is: {}'.format(right))
        # logger.info('top is: {}'.format(top))
        # logger.info('bottom is: {}'.format(bottom))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        img_crop_width = int(np.linalg.norm(points[0] - points[1]))
        img_crop_height = int(np.linalg.norm(points[0] - points[3]))
        pts_std = np.float32([[0, 0], [img_crop_width, 0], \
                      [img_crop_width, img_crop_height], [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img_crop,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        # [x1,x2,y1,y2]

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        if self.character_type == "ch":
            imgW = int(32 * max_wh_ratio)
        h = img.shape[0]
        w = img.shape[1]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)

        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def preprocess(self, img_list):
        img_num = len(img_list)
        norm_img_batch = []
        max_wh_ratio = 0
        for ino in range(img_num):
            h, w = img_list[ino].shape[0:2]
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(max_wh_ratio, wh_ratio)

        for ino in range(img_num):
            norm_img = self.resize_norm_img(img_list[ino], max_wh_ratio)
            norm_img = norm_img[np.newaxis, :]
            norm_img_batch.append(norm_img)
        norm_img_batch = np.concatenate(norm_img_batch)
        norm_img_batch = norm_img_batch.copy()

        return norm_img_batch[0]

    def postprocess_old(self, outputs, with_score=False):
        rec_res = []
        rec_idx_lod = outputs["ctc_greedy_decoder_0.tmp_0.lod"]
        rec_idx_batch = outputs["ctc_greedy_decoder_0.tmp_0"]
        if with_score:
            predict_lod = outputs["softmax_0.tmp_0.lod"]
        for rno in range(len(rec_idx_lod) - 1):
            beg = rec_idx_lod[rno]
            end = rec_idx_lod[rno + 1]
            if isinstance(rec_idx_batch, list):
                rec_idx_tmp = [x[0] for x in rec_idx_batch[beg:end]]
            else:  #nd array
                rec_idx_tmp = rec_idx_batch[beg:end, 0]
            preds_text = self.char_ops.decode(rec_idx_tmp)
            if with_score:
                beg = predict_lod[rno]
                end = predict_lod[rno + 1]
                if isinstance(outputs["softmax_0.tmp_0"], list):
                    outputs["softmax_0.tmp_0"] = np.array(outputs[
                        "softmax_0.tmp_0"]).astype(np.float32)
                probs = outputs["softmax_0.tmp_0"][beg:end, :]
                ind = np.argmax(probs, axis=1)
                blank = probs.shape[1]
                valid_ind = np.where(ind != (blank - 1))[0]
                score = np.mean(probs[valid_ind, ind[valid_ind]])
                rec_res.append([preds_text, score])
            else:
                rec_res.append([preds_text])
        return rec_res

    def postprocess(self, outputs, with_score=False):
        preds = outputs["softmax_0.tmp_0"]
        try:
            preds = preds.numpy()
        except:
            pass
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text, score = self.label_ops.decode(
            preds_idx, preds_prob, is_remove_duplicate=True)
        return text, preds, score


