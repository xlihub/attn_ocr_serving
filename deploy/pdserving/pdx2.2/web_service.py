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

# from paddle_serving_server.web_service import WebService, Op
import yaml
from paddle_serving_server.pipeline import Op, RequestOp, ResponseOp
from paddle_serving_server.pipeline import PipelineServer
from paddle_serving_server.pipeline.channel import ProductErrCode
from paddle_serving_server.pipeline import ResponseOp
from paddle_serving_server.pipeline.proto import pipeline_service_pb2
from paddle_serving_server.pipeline.channel import ChannelDataErrcode, ChannelDataType

import logging
import numpy as np
import cv2
import sys
import base64
# from paddle_serving_app.reader import OCRReader
from ocr_reader import OCRReader, DetResizeForTest, Maskrcnn_Normalize, ArrangeMaskRCNN, generate_minibatch, maskrcnn_postprocess, offset_to_lengths
from paddle_serving_app.reader import *
from PIL import Image
# from tools.infer.utility import draw_ocr
import os
import copy

_LOGGER = logging.getLogger()


class DrawBoxes:
    list = []


class Maskrcnn_Op(Op):
    def init_op(self):
        self.maskrcnn_preprocess = Sequential([
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], False),
            Resize((800, 1333)), PadStride(32), Transpose((2, 0, 1))
        ])
        self.threshold = 0.9
        self.bitchsize = 1
        global_config = yaml.load(
            open("/paddle/maskrcnn_serving_server/model.yml", 'rb'), Loader=yaml.Loader)
        self.num_classes = global_config['_Attributes']['num_classes']
        self.labels = global_config['_Attributes']['labels']

    def preprocess(self, input_dicts, data_id, log_id):
        (_, input_dict), = input_dicts.items()
        data = base64.b64decode(input_dict["image"].encode('utf8'))
        data = np.fromstring(data, np.uint8)
        # Note: class variables(self.var) can only be used in process op mode
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        self.raw_image = image.astype('float32')
        batch_data = list()
        batch_data.append(self.maskrcnn_preprocess(image))
        padding_batch = generate_minibatch(batch_data)
        im = padding_batch[0]
        im_shape = np.array(list(im.shape[1:])).reshape(-1)
        res = dict()
        res['im_shape'] = np.array(list(im.shape[1:])).reshape(-1)[np.newaxis, :]
        res['image'] = im[np.newaxis, :]
        res['scale_factor'] = np.array([1.0, 1.0]).reshape(-1)[np.newaxis, :]
        self.im_shape = im_shape
        return res, False, None, ""

    def postprocess(self, input_dicts, fetch_dict, data_id, log_id):
        results = list()
        mask_pred = [fetch_dict["mask_pred"], fetch_dict['mask_pred.lod'].tolist()]
        bbox_pred = [fetch_dict["multiclass_nms_0.tmp_0"], fetch_dict['multiclass_nms_0.tmp_0.lod'].tolist()]
        results.append(bbox_pred)
        results.append(mask_pred)
        res = {'bbox': (results[0][0], offset_to_lengths(results[0][1])), }
        res['im_id'] = (np.array(
            [[i] for i in range(1)]).astype('int32'), [[]])
        res['mask'] = (results[1][0], offset_to_lengths(results[1][1]))
        res['im_shape'] = (self.im_shape, [])
        preds = maskrcnn_postprocess(
            res, self.bitchsize, self.num_classes,
            28, self.labels)
        pred = preds[0]
        from utils import visualize_detection
        visualize_detection(self.raw_image, pred, save_dir="/paddle/inference_results/")
        keep_results = []
        areas = []
        for dt in np.array(pred):
            cname, bbox, score = dt['category'], dt['bbox'], dt['score']
            if score < self.threshold:
                continue
            keep_results.append(dt)
            areas.append(bbox[2] * bbox[3])
        areas = np.asarray(areas)
        sorted_idxs = np.argsort(-areas).tolist()
        keep_results = [keep_results[k]
                        for k in sorted_idxs] if len(keep_results) > 0 else []
        out_list = []
        if len(keep_results) > 0:
            for i, keep_result in enumerate(keep_results):
                top = int(keep_result['bbox'][1])
                bottom = int(keep_result['bbox'][1]) + int(keep_result['bbox'][3])
                left = int(keep_result['bbox'][0])
                right = int(keep_result['bbox'][0]) + int(keep_result['bbox'][2])
                raw_image_show = self.raw_image[..., ::
                                                     -1]
                img_crop = self.raw_image[top:bottom, left:right, :].copy()
                img_crop_show = img_crop[..., ::-1]
                draw_img_save = "/paddle/inference_results/"
                if not os.path.exists(draw_img_save):
                    os.makedirs(draw_img_save)
                cv2.imwrite(
                    os.path.join(draw_img_save, os.path.basename('maskrcnn_bbox.jpg')),
                    raw_image_show[:, :, ::-1])
                cv2.imwrite(
                    os.path.join(draw_img_save, os.path.basename('maskrcnn' + str(i) + '.jpg')),
                    img_crop_show[:, :, ::-1])
                out_dict = {"im_type": keep_result['category'], "image": img_crop}
                out_list.append(out_dict)
            return {"list": out_list}, None, ""
        else:
            out_dict = {"im_type": 'unknown'}
            out_list.append(out_dict)
            return {"list": out_list}, 901, "unknown type"


class DetOp(Op):
    def init_op(self):
        self.det_preprocess = Sequential([
            DetResizeForTest(), Div(255),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), Transpose(
                (2, 0, 1))
        ])
        self.filter_func = FilterBoxes(10, 10)
        self.post_func = DBPostProcess({
            "thresh": 0.3,
            "box_thresh": 0.5,
            "max_candidates": 1000,
            "unclip_ratio": 1.88,
            "min_size": 3
        })

    def preprocess(self, input_dicts, data_id, log_id):
        (_, input_dict), = input_dicts.items()
        self.im_list = input_dict["list"]
        feed_list = []
        img_list = []
        max_wh_ratio = 0
        ## Many mini-batchs, the type of feed_data is list.
        max_batch_size = len(self.im_list)  # len(dt_boxes)

        # If max_batch_size is 0, skipping predict stage
        if max_batch_size == 0:
            return {}, True, None, ""
        boxes_size = len(self.im_list)
        batch_size = boxes_size // max_batch_size
        rem = boxes_size % max_batch_size
        for bt_idx in range(0, batch_size + 1):
            imgs = None
            boxes_num_in_one_batch = 0
            if bt_idx == batch_size:
                if rem == 0:
                    continue
                else:
                    boxes_num_in_one_batch = rem
            elif bt_idx < batch_size:
                boxes_num_in_one_batch = max_batch_size
            else:
                _LOGGER.error("batch_size error, bt_idx={}, batch_size={}".
                              format(bt_idx, batch_size))
                break

            start = bt_idx * max_batch_size
            end = start + boxes_num_in_one_batch
            img_list = []
            self.im_info_list = []
            for box_idx in range(start, end):
                boximg = self.im_list[box_idx]
                img_list.append(boximg)
                self.raw_im = boximg["image"]
                self.im_type = boximg["im_type"]
                # data = np.fromstring(data, np.uint8)
                # # Note: class variables(self.var) can only be used in process op mode
                # im = cv2.imdecode(data, cv2.IMREAD_COLOR)
                im = boximg["image"]
                self.ori_h, self.ori_w, _ = im.shape

                if self.im_type == 'train_ticket_tw':
                    self.det_preprocess = Sequential([
                        DetResizeForTest(resize_long=800), Div(255),
                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), Transpose(
                            (2, 0, 1))
                    ])
                    self.post_func = DBPostProcess({
                        "thresh": 0.3,
                        "box_thresh": 0.5,
                        "max_candidates": 1000,
                        "unclip_ratio": 1.6,
                        "min_size": 3
                    })
                elif self.im_type == 'zhdx_type2':
                    self.det_preprocess = Sequential([
                        DetResizeForTest(resize_long=1500), Div(255),
                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), Transpose(
                            (2, 0, 1))
                    ])
                    self.post_func = DBPostProcess({
                        "thresh": 0.3,
                        "box_thresh": 0.5,
                        "max_candidates": 1000,
                        "unclip_ratio": 1.8,
                        "min_size": 3
                    })
                elif self.im_type == 'Impo_tw':
                    self.det_preprocess = Sequential([
                        DetResizeForTest(resize_long=1500), Div(255),
                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), Transpose(
                            (2, 0, 1))
                    ])
                    self.post_func = DBPostProcess({
                        "thresh": 0.3,
                        "box_thresh": 0.5,
                        "max_candidates": 1000,
                        "unclip_ratio": 1.6,
                        "min_size": 3
                    })
                elif self.im_type == 'invoice_A5':
                    self.det_preprocess = Sequential([
                        DetResizeForTest(resize_long=1500), Div(255),
                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), Transpose(
                            (2, 0, 1))
                    ])
                det_img = self.det_preprocess(im)
                _, self.new_h, self.new_w = det_img.shape
                self.im_info_list.append(
                    {'ori_h': self.ori_h, 'ori_w': self.ori_w, 'new_h': self.new_h, 'new_w': self.new_w})
                feed = {"x": det_img[np.newaxis, :].copy()}
                feed_list.append(feed)
        return feed_list, False, None, ""


    def postprocess(self, input_dicts, fetch_dict, data_id, log_id):
        if len(fetch_dict) > 0:
            out_list = []
            self.boxes = []
            for i, dict in enumerate(fetch_dict):
                det_out = dict["sigmoid_0.tmp_0"]
                ratio_list = [
                    float(self.im_info_list[i]['new_h']) / self.im_info_list[i]['ori_h'], float(self.im_info_list[i]['new_w']) / self.im_info_list[i]['ori_w']
                ]
                dt_boxes_list = self.post_func(det_out, [ratio_list])
                dt_boxes = self.filter_func(dt_boxes_list[0], [self.im_info_list[i]['ori_h'], self.im_info_list[i]['ori_w']])
                out_dict = {"dt_boxes": dt_boxes, "image": self.im_list[i]["image"], "im_type": self.im_list[i]["im_type"]}
                self.boxes.append(dt_boxes)
                out_list.append(out_dict)
            return {"list": out_list}, None, ""


def draw_ocr(image,
             boxes):
    """
    Visualize the results of OCR detection and recognition
    args:
        image(Image|array): RGB image
        boxes(list): boxes with shape(N, 4, 2)
        txts(list): the texts
        scores(list): txxs corresponding scores
        drop_score(float): only scores greater than drop_threshold will be visualized
        font_path: the path of font which is used to draw text
    return(array):
        the visualized img
    """
    box_num = len(boxes)
    for i in range(box_num):
        box = np.reshape(np.array(boxes[i]), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    return image


class RecOp(Op):
    def init_op(self):
        self.ocr_reader = OCRReader(
            char_dict_path="../../ppocr/utils/dict/chinese_cht_dict.txt")

        self.get_rotate_crop_image = GetRotateCropImage()
        self.sorted_boxes = SortedBoxes()

    def preprocess(self, input_dicts, data_id, log_id):
        (_, dict), = input_dicts['det'].items()
        feed_list = []
        self.rec_dict_list = []
        for i, input_dict in enumerate(dict):
            self.raw_im = input_dict["image"]
            self.im_type = input_dict["im_type"]
            # data = np.frombuffer(self.raw_im, np.uint8)
            # im = cv2.imdecode(data, cv2.IMREAD_COLOR)
            dt_boxes = input_dict["dt_boxes"]
            dt_boxes = self.sorted_boxes(dt_boxes)
            DrawBoxes.list = copy.deepcopy(dt_boxes)
            self.rec_dict_list.append({"dt_boxes": DrawBoxes.list, "im_type": input_dict["im_type"]})
            image = self.raw_im[..., ::-1]
            # cv2.namedWindow('first', cv2.WINDOW_NORMAL)
            # cv2.imshow('first', image_cv)
            boxes = DrawBoxes.list

            draw_img = draw_ocr(
                image,
                boxes)
            draw_img_save = "/paddle/inference_results/"
            if not os.path.exists(draw_img_save):
                os.makedirs(draw_img_save)
            cv2.imwrite(
                os.path.join(draw_img_save, os.path.basename('result' + str(i) + '.jpg')),
                draw_img[:, :, ::-1])
            # DrawBoxes.list = self.sorted_boxes(DrawBoxes.list)
            img_list = []
            max_wh_ratio = 0
            ## Many mini-batchs, the type of feed_data is list.
            max_batch_size = len(dt_boxes)  # len(dt_boxes)

            # If max_batch_size is 0, skipping predict stage
            if max_batch_size == 0:
                return {}, True, None, ""
            boxes_size = len(dt_boxes)
            batch_size = boxes_size // max_batch_size
            rem = boxes_size % max_batch_size
            for bt_idx in range(0, batch_size + 1):
                imgs = None
                boxes_num_in_one_batch = 0
                if bt_idx == batch_size:
                    if rem == 0:
                        continue
                    else:
                        boxes_num_in_one_batch = rem
                elif bt_idx < batch_size:
                    boxes_num_in_one_batch = max_batch_size
                else:
                    _LOGGER.error("batch_size error, bt_idx={}, batch_size={}".
                                  format(bt_idx, batch_size))
                    break

                start = bt_idx * max_batch_size
                end = start + boxes_num_in_one_batch
                img_list = []
                for box_idx in range(start, end):
                    boximg = self.get_rotate_crop_image(self.raw_im, dt_boxes[box_idx])
                    self.ocr_reader.get_sorted_boxes(self.raw_im, dt_boxes[box_idx])
                    img_list.append(boximg)
                    h, w = boximg.shape[0:2]
                    wh_ratio = w * 1.0 / h
                    max_wh_ratio = max(max_wh_ratio, wh_ratio)
                _, w, h = self.ocr_reader.resize_norm_img(img_list[0],
                                                          max_wh_ratio).shape

                imgs = np.zeros((boxes_num_in_one_batch, 3, w, h)).astype('float32')
                for id, img in enumerate(img_list):
                    norm_img = self.ocr_reader.resize_norm_img(img, max_wh_ratio)
                    imgs[id] = norm_img
                feed = {"x": imgs.copy()}
                feed_list.append(feed)
        return feed_list, False, None, ""

    def postprocess(self, input_dicts, fetch_data, data_id, log_id):
        res_list = []
        if len(fetch_data) == 1:
            fetch_data = fetch_data[0]
        if isinstance(fetch_data, dict):
            text_list = []
            preb_list = []
            if len(fetch_data) > 0:
                text, preb = self.ocr_reader.postprocess(
                    fetch_data, with_score=True)
                # for res in rec_batch_res:
                   # res_list.append(res)
                text_list.append(text)
                preb_list.append(preb)
                image = self.raw_im[..., ::-1]
                # cv2.namedWindow('first', cv2.WINDOW_NORMAL)
                # cv2.imshow('first', image_cv)
                boxes = DrawBoxes.list

                draw_img = draw_ocr(
                    image,
                    boxes)
                draw_img_save = "/paddle/inference_results/"
                if not os.path.exists(draw_img_save):
                    os.makedirs(draw_img_save)
                cv2.imwrite(
                    os.path.join(draw_img_save, os.path.basename('result.jpg')),
                    draw_img[:, :, ::-1])
                res_ob = {"text": text_list, "preb": preb_list, "boxes": DrawBoxes.list}
                res_text = []
                res_boxes = []
                res_preb = []
                for text in text_list:
                    res_text += text
                for preb in preb_list:
                    batch_size = len(preb)
                    for batch_idx in range(batch_size):
                        item = preb[batch_idx]
                        preds_idx = item.argmax(axis=1)
                        array_shape = preds_idx.shape
                        array_data_type = preds_idx.dtype.name
                        item_str = preds_idx.tobytes()
                        new_arr = np.frombuffer(item_str, dtype=array_data_type).reshape(array_shape)
                        # item_str = ''.join(np.array2string(item, separator=',', threshold=11e3).splitlines())
                        res_preb.append({'bytes': item_str, 'dtype': array_data_type, 'shape': array_shape})
                        # preds_idx = item.argmax(axis=1)
                        # preds_prob = item.max(axis=1)
                        # print(preds_idx)
                        # print(item)
                        # char_list = []
                        # conf_list = []
                        # for idx in range(len(text_index[batch_idx])):
                        #     if text_index[batch_idx][idx] in ignored_tokens:
                        #         continue
                        #     if is_remove_duplicate:
                        #         # only for predict
                        #         if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                        #             batch_idx][idx]:
                        #             continue
                        #     char_list.append(self.character[int(text_index[batch_idx][
                        #                                             idx])])
                        #     if text_prob is not None:
                        #         conf_list.append(text_prob[batch_idx][idx])
                        #     else:
                        #         conf_list.append(1)
                        # text = ''.join(char_list)
                        # result_list.append((text))
                    # for batch in np.nditer(preb):
                    #     print(batch)
                for boxes in DrawBoxes.list:
                    b1 = boxes[0].tolist()
                    b3 = boxes[2].tolist()
                    bb = b1 + b3
                    res_boxes.append(bb)

                res = {"text": str(res_text), "preb": str(res_preb), "boxes": str(res_boxes), "im_type": self.im_type}
                print(str(res_text))
                return res, None, ""
        elif isinstance(fetch_data, list):
            for i, one_batch in enumerate(fetch_data):
                text_list = []
                preb_list = []
                text, preb = self.ocr_reader.postprocess(
                    one_batch, with_score=True)
                # for res in one_batch_res:
                text_list.append(text)
                preb_list.append(preb)
                image = self.raw_im[..., ::-1]
                # cv2.namedWindow('first', cv2.WINDOW_NORMAL)
                # cv2.imshow('first', image_cv)
                boxes = DrawBoxes.list

                draw_img = draw_ocr(
                    image,
                    boxes)
                draw_img_save = "/paddle/inference_results/"
                if not os.path.exists(draw_img_save):
                    os.makedirs(draw_img_save)
                cv2.imwrite(
                    os.path.join(draw_img_save, os.path.basename('result.jpg')),
                    draw_img[:, :, ::-1])
                res_ob = {"text": text_list, "preb": preb_list, "boxes": DrawBoxes.list}
                res_text = []
                res_boxes = []
                res_preb = []
                for text in text_list:
                    res_text += text
                for preb in preb_list:
                    batch_size = len(preb)
                    for batch_idx in range(batch_size):
                        item = preb[batch_idx]
                        preds_idx = item.argmax(axis=1)
                        array_shape = preds_idx.shape
                        array_data_type = preds_idx.dtype.name
                        item_str = preds_idx.tobytes()
                        new_arr = np.frombuffer(item_str, dtype=array_data_type).reshape(array_shape)
                        # item_str = ''.join(np.array2string(item, separator=',', threshold=11e3).splitlines())
                        res_preb.append({'bytes': item_str, 'dtype': array_data_type, 'shape': array_shape})
                for boxes in self.rec_dict_list[i]['dt_boxes']:
                    b1 = boxes[0].tolist()
                    b3 = boxes[2].tolist()
                    bb = b1 + b3
                    res_boxes.append(bb)

                res = {"text": str(res_text), "preb": str(res_preb), "boxes": str(res_boxes), "im_type": self.rec_dict_list[i]['im_type']}
                print(str(res_text))
                res_list.append(res)
            return {'res': str(res_list)}, None, ""

class DetChOp(Op):
    def init_op(self):
        self.det_preprocess = Sequential([
            DetResizeForTest(), Div(255),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), Transpose(
                (2, 0, 1))
        ])
        self.filter_func = FilterBoxes(10, 10)
        self.post_func = DBPostProcess({
            "thresh": 0.3,
            "box_thresh": 0.5,
            "max_candidates": 1000,
            "unclip_ratio": 1.5,
            "min_size": 3
        })

    def preprocess(self, input_dicts, data_id, log_id):
        (_, input_dict), = input_dicts.items()
        data = base64.b64decode(input_dict["image"].encode('utf8'))
        self.raw_im = data
        data = np.fromstring(data, np.uint8)
        # Note: class variables(self.var) can only be used in process op mode
        im = cv2.imdecode(data, cv2.IMREAD_COLOR)
        self.ori_h, self.ori_w, _ = im.shape
        det_img = self.det_preprocess(im)
        _, self.new_h, self.new_w = det_img.shape
        return {"x": det_img[np.newaxis, :].copy()}, False, None, ""

    def postprocess(self, input_dicts, fetch_dict, data_id, log_id):
        det_out = fetch_dict["sigmoid_0.tmp_0"]
        ratio_list = [
            float(self.new_h) / self.ori_h, float(self.new_w) / self.ori_w
        ]
        dt_boxes_list = self.post_func(det_out, [ratio_list])
        dt_boxes = self.filter_func(dt_boxes_list[0], [self.ori_h, self.ori_w])
        out_dict = {"dt_boxes": dt_boxes, "image": self.raw_im}

        return out_dict, None, ""

class RecChOp(Op):
    def init_op(self):
        self.ocr_reader = OCRReader(
            char_dict_path="../../ppocr/utils/ppocr_keys_v1.txt")

        self.get_rotate_crop_image = GetRotateCropImage()
        self.sorted_boxes = SortedBoxes()

    def preprocess(self, input_dicts, data_id, log_id):
        (_, input_dict), = input_dicts.items()
        raw_im = input_dict["image"]
        data = np.frombuffer(raw_im, np.uint8)
        im = cv2.imdecode(data, cv2.IMREAD_COLOR)
        dt_boxes = input_dict["dt_boxes"]
        dt_boxes = self.sorted_boxes(dt_boxes)
        feed_list = []
        img_list = []
        max_wh_ratio = 0
        ## Many mini-batchs, the type of feed_data is list.
        max_batch_size = 6  # len(dt_boxes)

        # If max_batch_size is 0, skipping predict stage
        if max_batch_size == 0:
            return {}, True, None, ""
        boxes_size = len(dt_boxes)
        batch_size = boxes_size // max_batch_size
        rem = boxes_size % max_batch_size
        for bt_idx in range(0, batch_size + 1):
            imgs = None
            boxes_num_in_one_batch = 0
            if bt_idx == batch_size:
                if rem == 0:
                    continue
                else:
                    boxes_num_in_one_batch = rem
            elif bt_idx < batch_size:
                boxes_num_in_one_batch = max_batch_size
            else:
                _LOGGER.error("batch_size error, bt_idx={}, batch_size={}".
                              format(bt_idx, batch_size))
                break

            start = bt_idx * max_batch_size
            end = start + boxes_num_in_one_batch
            img_list = []
            for box_idx in range(start, end):
                boximg = self.get_rotate_crop_image(im, dt_boxes[box_idx])
                img_list.append(boximg)
                h, w = boximg.shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            _, w, h = self.ocr_reader.resize_norm_img(img_list[0],
                                                      max_wh_ratio).shape

            imgs = np.zeros((boxes_num_in_one_batch, 3, w, h)).astype('float32')
            for id, img in enumerate(img_list):
                norm_img = self.ocr_reader.resize_norm_img(img, max_wh_ratio)
                imgs[id] = norm_img
            feed = {"x": imgs.copy()}
            feed_list.append(feed)

        return feed_list, False, None, ""

    def postprocess(self, input_dicts, fetch_data, data_id, log_id):
        res_list = []
        if isinstance(fetch_data, dict):
            if len(fetch_data) > 0:
                rec_batch_res = self.ocr_reader.postprocess(
                    fetch_data, with_score=True)
                for res in rec_batch_res:
                    res_list.append(res[0])
        elif isinstance(fetch_data, list):
            for one_batch in fetch_data:
                one_batch_res = self.ocr_reader.postprocess(
                    one_batch, with_score=True)
                for res in one_batch_res:
                    res_list.append(res[0])

        res = {"res": str(res_list)}
        return res, None, ""
# class OcrRequestOp(RequestOp):
#     def unpack_request_package(self, request):
#         dictdata = {}
#         for idx, key in enumerate(request.key):
#             if key != "words":
#                 continue
#             words = request.value[idx]
#             word_ids, _ = self.imdb_dataset.get_words_and_label(words)
#             word_len = len(word_ids)
#             dictdata[key] = np.array(word_ids).reshape(word_len, 1)
#             dictdata["{}.lod".format(key)] = np.array([0, word_len])
#
#         log_id = None
#         if request.logid is not None:
#             log_id = request.logid
#         return dictdata, log_id, None, ""


# class OcrService(WebService):
#     def get_pipeline_response(self, read_op):
#         maskrcnn_op = Maskrcnn_Op(name="maskrcnn", input_ops=[read_op])
#         det_op = DetOp(name="det", input_ops=[maskrcnn_op])
#         rec_op = RecOp(name="rec", input_ops=[det_op])
#         return rec_op


class OcrResponseOp(ResponseOp):
    # Here ImdbResponseOp is consistent with the default ResponseOp implementation
    def pack_response_package(self, channeldata):
        resp = pipeline_service_pb2.Response()
        error_code = channeldata.error_code
        error_info = ""
        if error_code == ChannelDataErrcode.OK.value:
            # Framework level errors
            if channeldata.datatype == ChannelDataType.CHANNEL_NPDATA.value:
                feed = channeldata.parse()
                # ndarray to string:
                # https://stackoverflow.com/questions/30167538/convert-a-numpy-ndarray-to-stringor-bytes-and-convert-it-back-to-numpy-ndarray
                np.set_printoptions(threshold=sys.maxsize)
                for name, var in feed.items():
                    resp.value.append(var.__repr__())
                    resp.key.append(name)
            elif channeldata.datatype == ChannelDataType.DICT.value:
                feed = channeldata.parse()
                for name, var in feed.items():
                    if not isinstance(var, str):
                        error_code = ChannelDataErrcode.TYPE_ERROR.value
                        error_info = self._log(
                            "fetch var type must be str({}).".format(
                                type(var)))
                        _LOGGER.error("(logid={}) Failed to pack RPC "
                                      "response package: {}".format(
                            channeldata.id, resp.err_msg))
                        break
                    resp.value.append(var)
                    resp.key.append(name)
            else:
                error_code = ChannelDataErrcode.TYPE_ERROR.value
                error_info = self._log("error type({}) in datatype.".format(
                    channeldata.datatype))
                _LOGGER.error("(logid={}) Failed to pack RPC response"
                              " package: {}".format(channeldata.id, error_info))
        else:
            # Product level errors
            if error_code == ChannelDataErrcode.PRODUCT_ERROR.value:
                # rewrite error_code when product errors occured
                if channeldata.prod_error_code == 901:
                    channeldata.error_code = 0
                    error_code = 0
                    resp.value.append('unknown')
                    resp.key.append('im_type')

        # pack results
        if error_code is None:
            error_code = 0
        resp.err_no = error_code
        resp.err_msg = error_info

        return resp

read_op = RequestOp()
maskrcnn_op = Maskrcnn_Op(name="maskrcnn", input_ops=[read_op])
det_op = DetOp(name="det", input_ops=[maskrcnn_op])
rec_op = RecOp(name="rec", input_ops=[det_op])
response_op = OcrResponseOp(input_ops=[rec_op])

detch_op = DetChOp(name="det", input_ops=[read_op])
recch_op = RecChOp(name="recch", input_ops=[detch_op])
responsech_op = OcrResponseOp(input_ops=[recch_op])

# use default ResponseOp implementation
#response_op = ResponseOp(input_ops=[combine_op])

server = PipelineServer()
server.set_response_op(response_op)
server.prepare_server('config.yml')
server.run_server()
# uci_service = OcrService(name="ocr")
# uci_service.prepare_pipeline_config("config.yml")
# uci_service.set_response_op(OcrResponseOp)
# uci_service.run_service()