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
from ocr_reader import OCRReader, DetResizeForTest, Maskrcnn_Normalize, ResizeByShort, Padding, Compose, ArrangeMaskRCNN, generate_minibatch, maskrcnn_postprocess, offset_to_lengths
from paddle_serving_app.reader import Sequential, ResizeByFactor
from paddle_serving_app.reader import Div, Normalize, Transpose
from paddle_serving_app.reader import DBPostProcess, FilterBoxes, GetRotateCropImage, SortedBoxes
from PIL import Image
# from tools.infer.utility import draw_ocr
import os
import copy

_LOGGER = logging.getLogger()

class DrawBoxes:
    list = []


class Maskrcnn_Op(Op):
    def init_op(self):
        self.maskrcnn_preprocess = Compose([
            Maskrcnn_Normalize(), ResizeByShort(), Padding(
                32), ArrangeMaskRCNN()
        ])
        self.threshold = 0.7
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
        im = np.array([data[0] for data in padding_batch])
        im_resize_info = np.array([data[1] for data in padding_batch])
        im_shape = np.array([data[2] for data in padding_batch])
        res = dict()
        res['image'] = im
        res['im_info'] = im_resize_info
        res['im_shape'] = im_shape
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
            if dt['category'] == 'invoice_sy' or 'invoice_ey':
                self.threshold = 0.1
            elif dt['category'] == 'invoice_A5' or dt['category'] == 'invoice_A4':
                self.threshold = 0.5
            else:
                self.threshold = 0.5
            if score < self.threshold:
                continue
            keep_results.append(dt)
            areas.append(bbox[2] * bbox[3])
        areas = np.asarray(areas)
        sorted_idxs = np.argsort(-areas).tolist()
        keep_results = [keep_results[k]
                        for k in sorted_idxs] if len(keep_results) > 0 else []
        out_list = []
        mask_list = []
        if len(keep_results) > 0:
            keep_results = keep_results[:1]
            for i, keep_result in enumerate(keep_results):
                mask_list.append({'box': keep_result['bbox'], 'type': keep_result['category']})
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
                # 前处理invoice_A4与invoice_A5的图片，去除表格竖线
                if keep_result['category'] == 'invoice_A5' or keep_result['category'] == 'invoice_A4':
                    cv2.imwrite(
                        os.path.join(draw_img_save, os.path.basename('raw_image.jpg')),
                        img_crop[:, :, ::-1])
                    img = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
                    img = img.astype('uint8')
                    # 二值化
                    # binary = cv2.adaptiveThreshold(
                    #    ~img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

                    ret2, image_otsu = cv2.threshold(~img, 0, 255, cv2.THRESH_OTSU)
                    ret, binary = cv2.threshold(~img, ret2, 255, cv2.THRESH_BINARY)
                    # 开运算-去除白点
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
                    binary_of_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                    # 闭运算-加强虚线
                    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
                    binary_of_close = cv2.morphologyEx(binary_of_open, cv2.MORPH_CLOSE, kernel2)
                    # 找线
                    src_copy = binary_of_close
                    scale = 50
                    kernel3 = cv2.getStructuringElement(
                        cv2.MORPH_RECT, (1, int(src_copy.shape[0] / scale)))
                    erorsion_img = cv2.erode(src_copy, kernel3, (-1, -1))
                    erorsion_img = cv2.dilate(erorsion_img, kernel3, (-1, -1))
                    contours, _ = cv2.findContours(
                        image=erorsion_img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
                    for i in contours:
                        x, y, w, h = cv2.boundingRect(i)
                        if h >= 800 and w <= 20:
                            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    img_crop = img
                    img_show = img[..., ::-1]
                    cv2.imwrite('/paddle/inference_results/dotted_line.jpg', img_show[:, :, ::-1])
                out_dict = {"im_type": keep_result['category'], "image": img_crop, "mask_list": mask_list}
                out_list.append(out_dict)
                if keep_result['category'] == 'invoice_sk':
                    img = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
                    img = img.astype('uint8')
                    # 先利用二值化去除图片噪声
                    ret2, image_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
                    ret, binary1 = cv2.threshold(img, ret2, 255, cv2.THRESH_BINARY)
                    img2 = binary1.copy()
                    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
                    img_show2 = img2[..., ::-1]
                    cv2.imwrite('/paddle/inference_results/sk_output2.jpg', img_show2[:, :, ::-1])
                    ret40, binary2 = cv2.threshold(img, ret2 + 40, 255, cv2.THRESH_BINARY)
                    img3 = binary2.copy()
                    img3 = cv2.cvtColor(img3, cv2.COLOR_GRAY2BGR)

                    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    n = len(contours)  # 轮廓的个数
                    cv_contours = []
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area <= 1000:
                            cv_contours.append(contour)
                        else:
                            continue

                    cv2.fillPoly(img, cv_contours, (255, 255, 255))
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    img_show = img[..., ::-1]
                    cv2.imwrite('/paddle/inference_results/sk_output.jpg', img_show[:, :, ::-1])
                    hight = int(img.shape[0])
                    width = int(img.shape[1])

                    # 统编
                    crop_1 = img[int(0.15 * hight):int(0.25 * hight), int(0.11 * width):int(0.44 * width)]
                    img_crop_1 = cv2.cvtColor(crop_1, cv2.COLOR_BGR2GRAY)
                    img_crop_1 = img_crop_1.astype('uint8')
                    ret2, image_otsu = cv2.threshold(~img_crop_1, 0, 255, cv2.THRESH_OTSU)
                    ret, binary = cv2.threshold(~img_crop_1, ret2, 255, cv2.THRESH_BINARY)
                    # binary_1 = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                    # img_show = binary_1[..., ::-1]
                    # cv2.imwrite('/paddle/inference_results/binary_1.jpg', img_show[:, :, ::-1])

                    # gray = cv2.cvtColor(crop_1, cv2.COLOR_BGR2GRAY)
                    # # 此步骤形态学变换的预处理，得到可以查找矩形的图片
                    # # 参数：输入矩阵、输出矩阵数据类型、设置1、0时差分方向为水平方向的核卷积，设置0、1为垂直方向,ksize：核的尺寸
                    # sobel = cv2.Sobel(gray, cv2.CV_8U, 0, 1, ksize=3)
                    # # 二值化
                    # ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
                    # binary_2 = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                    # img_show = binary_2[..., ::-1]
                    # cv2.imwrite('/paddle/inference_results/binary_2.jpg', img_show[:, :, ::-1])
                    #
                    # # 设置膨胀和腐蚀操作的核函数
                    # element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
                    # element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))
                    # # 膨胀一次，让轮廓突出
                    # dilation = cv2.dilate(binary, element2, iterations=1)
                    # # 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
                    # erosion = cv2.erode(dilation, element1, iterations=1)
                    # # aim = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,element1, 1 )   #此函数可实现闭运算和开运算
                    # # 以上膨胀+腐蚀称为闭运算，具有填充白色区域细小黑色空洞、连接近邻物体的作用
                    # # 再次膨胀，让轮廓明显一些
                    # dilation2 = cv2.dilate(erosion, element2, iterations=3)
                    #  查找和筛选文字区域
                    region = []
                    #  查找轮廓
                    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    # 利用以上函数可以得到多个轮廓区域，存在一个列表中。
                    #  筛选那些面积小的
                    for i in range(len(contours)):
                        # 遍历所有轮廓
                        # cnt是一个点集
                        cnt = contours[i]
                        # 计算该轮廓的面积
                        area = cv2.contourArea(cnt)
                        # 面积小的都筛选掉、这个1000可以按照效果自行设置
                        if (area < 200):
                            continue
                        #     # 将轮廓形状近似到另外一种由更少点组成的轮廓形状，新轮廓的点的数目由我们设定的准确度来决定
                        #     # 轮廓近似，作用很小
                        #     # 计算轮廓长度
                        #     epsilon = 0.001 * cv2.arcLength(cnt, True)
                        #     #
                        # #     approx = cv2.approxPolyDP(cnt, epsilon, True)
                        # 找到最小的矩形，该矩形可能有方向
                        rect = cv2.minAreaRect(cnt)
                        # 打印出各个矩形四个点的位置
                        # print("rect is: ")
                        # print(rect)
                        # box是四个点的坐标
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        # 计算高和宽
                        box_height = abs(box[0][1] - box[2][1])
                        box_width = abs(box[0][0] - box[2][0])
                        # 筛选那些太细的矩形，留下扁的
                        if (box_height > box_width * 1.3):
                            continue
                        region.append(box)
                    # 用绿线画出这些找到的轮廓
                    # for box in region:
                    #     cv2.drawContours(crop_1, [box], 0, (255, 255, 255), 2)
                    # img_show = crop_1[..., ::-1]
                    # cv2.imwrite('/paddle/inference_results/output_1.jpg', img_show[:, :, ::-1])
                    dst_show = crop_1[..., ::-1]
                    cv2.imwrite("/paddle/inference_results/sk_crop_1.jpg", dst_show[:, :, ::-1])
                    out_dict = {"im_type": "extra", "ext_key": "UNINO", "image": crop_1}
                    out_list.append(out_dict)

                    # 日期
                    crop_3 = img[int(0.15 * hight):int(0.25 * hight), int(0.45 * width):int(0.75 * width)]
                    img_crop_3 = cv2.cvtColor(crop_3, cv2.COLOR_BGR2GRAY)
                    img_crop_3 = img_crop_3.astype('uint8')
                    ret2, image_otsu = cv2.threshold(~img_crop_3, 0, 255, cv2.THRESH_OTSU)
                    ret, binary = cv2.threshold(~img_crop_3, ret2, 255, cv2.THRESH_BINARY)
                    binary_3 = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                    img_show = binary_3[..., ::-1]
                    cv2.imwrite('/paddle/inference_results/binary_3.jpg', img_show[:, :, ::-1])
                    dst_show = crop_3[..., ::-1]
                    cv2.imwrite("/paddle/inference_results/sk_crop_3.jpg", dst_show[:, :, ::-1])
                    out_dict = {"im_type": "extra", "ext_key": "MMDD", "image": crop_3}
                    out_list.append(out_dict)

                    # 未税,税额,总计
                    crop_2 = img[int(0.62 * hight):int(0.86 * hight), int(0.45 * width):int(0.65 * width)]
                    dst_show = crop_2[..., ::-1]
                    cv2.imwrite("/paddle/inference_results/crop_2.jpg", dst_show[:, :, ::-1])
                    img_crop_2 = cv2.cvtColor(crop_2, cv2.COLOR_BGR2GRAY)
                    img_crop_2 = img_crop_2.astype('uint8')
                    ret2, image_otsu = cv2.threshold(~img_crop_2, 0, 255, cv2.THRESH_OTSU)
                    ret, binary = cv2.threshold(~img_crop_2, ret2, 255, cv2.THRESH_BINARY)

                    #  查找和筛选文字区域
                    region = []
                    #  查找轮廓
                    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    # 利用以上函数可以得到多个轮廓区域，存在一个列表中。
                    #  筛选那些面积小的
                    for i in range(len(contours)):
                        # 遍历所有轮廓
                        # cnt是一个点集
                        cnt = contours[i]
                        # 计算该轮廓的面积
                        area = cv2.contourArea(cnt)
                        # 面积小的都筛选掉、这个1000可以按照效果自行设置
                        if (area < 10000):
                            continue
                        #     # 将轮廓形状近似到另外一种由更少点组成的轮廓形状，新轮廓的点的数目由我们设定的准确度来决定
                        #     # 轮廓近似，作用很小
                        #     # 计算轮廓长度
                        #     epsilon = 0.001 * cv2.arcLength(cnt, True)
                        #     #
                        # #     approx = cv2.approxPolyDP(cnt, epsilon, True)
                        # 找到最小的矩形，该矩形可能有方向
                        rect = cv2.minAreaRect(cnt)
                        # 打印出各个矩形四个点的位置
                        # print("rect is: ")
                        # print(rect)
                        # box是四个点的坐标
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        # 计算高和宽
                        box_height = abs(box[0][1] - box[2][1])
                        box_width = abs(box[0][0] - box[2][0])
                        # 筛选那些太细的矩形，留下扁的
                        if (box_height > box_width * 1.3):
                            continue
                        region.append(box)
                    # 用绿线画出这些找到的轮廓
                    # for box in region:
                    #     cv2.drawContours(crop_2, [box], 0, (255, 255, 255), 2)
                    # img_show = crop_2[..., ::-1]
                    # cv2.imwrite('/paddle/inference_results/output_2.jpg', img_show[:, :, ::-1])
                    dst_show = crop_2[..., ::-1]
                    cv2.imwrite("/paddle/inference_results/sk_crop_2.jpg", dst_show[:, :, ::-1])
                    out_dict = {"im_type": "extra", "ext_key": "AMTN", "image": crop_2}
                    out_list.append(out_dict)

                    # # 税额
                    # crop_3 = img[int(0.7 * hight):int(0.78 * hight), int(0.45 * width):int(0.65 * width)]
                    # dst_show = crop_3[..., ::-1]
                    # cv2.imwrite("/paddle/inference_results/sk_crop_3.jpg", dst_show[:, :, ::-1])
                    # out_dict = {"im_type": "extra", "ext_key": "TAX", "image": crop_3}
                    # out_list.append(out_dict)
                    #
                    # # 总计
                    # crop_4 = img[int(0.78 * hight):int(0.86 * hight), int(0.45 * width):int(0.65 * width)]
                    # dst_show = crop_4[..., ::-1]
                    # cv2.imwrite("/paddle/inference_results/sk_crop_4.jpg", dst_show[:, :, ::-1])
                    # out_dict = {"im_type": "extra", "ext_key": "AMTN", "image": crop_4}
                    # out_list.append(out_dict)

                    dst = img[int(0 * hight):int(0.11 * hight), int(0 * width):int(0.08 * width)]
                    dst_show = dst[..., ::-1]
                    cv2.imwrite("/paddle/inference_results/sk_crop.jpg", dst_show[:, :, ::-1])
                    out_dict = {"im_type": "extra", "ext_key": "INV_NO1", "image": dst}
                    out_list.append(out_dict)
                    dst2 = img2[int(0 * hight):int(0.11 * hight), int(0.9 * width):int(0.99 * width)]
                    dst2_show = dst2[..., ::-1]
                    cv2.imwrite("/paddle/inference_results/sk_crop2.jpg", dst2_show[:, :, ::-1])
                    out_dict2 = {"im_type": "extra", "ext_key": "MM", "image": dst2}
                    out_list.append(out_dict2)
                    dst3 = img2[int(0.55 * hight):int(0.95 * hight), int(0.65 * width):int(0.95 * width)]
                    dst3_show = dst3[..., ::-1]
                    cv2.imwrite("/paddle/inference_results/sk_crop3.jpg", dst3_show[:, :, ::-1])
                    out_dict3 = {"im_type": "extra", "ext_key": "SK_S_UNINO", "image": dst3}
                    out_list.append(out_dict3)
                    dst4 = img3[int(0.55 * hight):int(0.95 * hight), int(0.65 * width):int(0.95 * width)]
                    dst4_show = dst4[..., ::-1]
                    cv2.imwrite("/paddle/inference_results/sk_crop4.jpg", dst4_show[:, :, ::-1])
                    out_dict4 = {"im_type": "extra", "ext_key": "SK_S_UNINO_2", "image": dst4}
                    out_list.append(out_dict4)
                if keep_result['category'] == 'invoice_ey':
                    img = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
                    img = img.astype('uint8')
                    # 先利用二值化去除图片噪声
                    ret2, image_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
                    ret, img = cv2.threshold(img, ret2 - 100, 255, cv2.THRESH_BINARY)

                    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    n = len(contours)  # 轮廓的个数
                    cv_contours = []
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area <= 300:
                            cv_contours.append(contour)
                        else:
                            continue

                    cv2.fillPoly(img, cv_contours, (255, 255, 255))
                    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    n = len(contours)  # 轮廓的个数
                    if n == 1:
                        img = img_crop
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    img_show = img[..., ::-1]
                    cv2.imwrite('/paddle/inference_results/ey_output.jpg', img_show[:, :, ::-1])
                    hight = int(img.shape[0])
                    width = int(img.shape[1])
                    dst = img[int(0.7 * hight):int(0.99 * hight), int(0 * width):int(0.99 * width)]
                    dst_show = dst[..., ::-1]
                    cv2.imwrite("/paddle/inference_results/ey_crop.jpg", dst_show[:, :, ::-1])
                    out_dict = {"im_type": "extra", "ext_key": "INV_NO2", "image": dst}
                    out_list.append(out_dict)
                if keep_result['category'] == 'invoice_sy':
                    img = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
                    img = img.astype('uint8')
                    # 先利用二值化去除图片噪声
                    ret2, image_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
                    ret, img = cv2.threshold(img, ret2, 255, cv2.THRESH_BINARY)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    img_show = img[..., ::-1]
                    cv2.imwrite('/paddle/inference_results/sy_output.jpg', img_show[:, :, ::-1])
                    hight = int(img.shape[0])
                    width = int(img.shape[1])
                    dst = img[int(0.2 * hight):int(0.3 * hight), int(0.1 * width):int(0.48 * width)]
                    dst_show = dst[..., ::-1]
                    cv2.imwrite("/paddle/inference_results/sy_crop.jpg", dst_show[:, :, ::-1])
                    out_dict = {"im_type": "extra", "ext_key": "YYMM", "image": dst}
                    out_list.append(out_dict)
                    dst2 = img[int(0.15 * hight):int(0.35 * hight), int(0.45 * width):int(0.6 * width)]
                    dst2_show = dst2[..., ::-1]
                    cv2.imwrite("/paddle/inference_results/sy_crop2.jpg", dst2_show[:, :, ::-1])
                    out_dict2 = {"im_type": "extra", "ext_key": "SY_NO1", "image": dst2}
                    out_list.append(out_dict2)
                    dst3 = img[int(0.15 * hight):int(0.35 * hight), int(0.55 * width):int(0.9 * width)]
                    dst3_show = dst3[..., ::-1]
                    cv2.imwrite("/paddle/inference_results/sy_crop3.jpg", dst3_show[:, :, ::-1])
                    out_dict3 = {"im_type": "extra", "ext_key": "SY_NO2", "image": dst3}
                    out_list.append(out_dict3)
                if keep_result['category'] == 'Impo_tw':
                    width = right - left
                    hight = bottom - top
                    if width > 2000 and width/hight > 2:
                        img = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
                        img = img.astype('uint8')
                        B_channel, G_channel, R_channel = cv2.split(img_crop)  # 注意cv2.split()返回通道顺序
                        # 红色通道阈值(调节好函数阈值为160时效果最好，太大一片白，太小干扰点太多)
                        _, BlueThresh = cv2.threshold(B_channel, 110, 255, cv2.THRESH_BINARY)

                        # 膨胀操作（可以省略）
                        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                        erode = cv2.erode(BlueThresh, element)
                        cv2.imwrite('/paddle/inference_results/impo_output_B.jpg', erode)
                        gray = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        img_show = gray[..., ::-1]
                        cv2.imwrite('/paddle/inference_results/impo_output.jpg', img_show[:, :, ::-1])
                        hight = int(erode.shape[0])
                        width = int(erode.shape[1])
                        dst = erode[int(0 * hight):int(0.2 * hight), int(0.6 * width):int(0.9 * width)]
                        # adaptiveThresh = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                        #                                        35, 90)
                        dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
                        dst_show = dst[..., ::-1]
                        cv2.imwrite("/paddle/inference_results/impo_crop.jpg", dst_show[:, :, ::-1])
                        out_dict = {"im_type": "extra", "ext_key": "IMPO_INV_NO", "image": dst}
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
                elif self.im_type == 'zhdx_type2' or self.im_type == 'zhdx_type1':
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
                        DetResizeForTest(resize_long=2000), Div(255),
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
                elif self.im_type == 'invoice_sy':
                    self.det_preprocess = Sequential([
                        DetResizeForTest(resize_long=3000), Div(255),
                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), Transpose(
                            (2, 0, 1))
                    ])
                    self.post_func = DBPostProcess({
                        "thresh": 0.3,
                        "box_thresh": 0.5,
                        "max_candidates": 1000,
                        "unclip_ratio": 1.6,
                        "min_size": 2
                    })
                elif self.im_type == 'invoice_ey' or self.im_type == 'tgddx':
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
                        "min_size": 5
                    })
                elif self.im_type == 'invoice_sk':
                    self.det_preprocess = Sequential([
                        DetResizeForTest(resize_long=1000), Div(255),
                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), Transpose(
                            (2, 0, 1))
                    ])
                    self.post_func = DBPostProcess({
                        "thresh": 0.3,
                        "box_thresh": 0.5,
                        "max_candidates": 1000,
                        "unclip_ratio": 2.0,
                        "min_size": 1
                    })
                elif self.im_type == 'extra':
                    if boximg["ext_key"] == 'INV_NO1':
                        self.det_preprocess = Sequential([
                            DetResizeForTest(resize_long=150), Div(255),
                            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), Transpose(
                                (2, 0, 1))
                        ])
                        self.post_func = DBPostProcess({
                            "thresh": 0.3,
                            "box_thresh": 0.5,
                            "max_candidates": 1000,
                            "unclip_ratio": 1.6,
                            "min_size": 1
                        })
                    if boximg["ext_key"] == 'UNINO':
                        self.det_preprocess = Sequential([
                            DetResizeForTest(resize_long=640), Div(255),
                            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), Transpose(
                                (2, 0, 1))
                        ])
                        self.post_func = DBPostProcess({
                            "thresh": 0.3,
                            "box_thresh": 0.5,
                            "max_candidates": 1000,
                            "unclip_ratio": 1.6,
                            "min_size": 1
                        })
                    if boximg["ext_key"] == 'AMTN_NET' or boximg["ext_key"] == 'TAX' or boximg["ext_key"] == 'AMTN':
                        self.det_preprocess = Sequential([
                            DetResizeForTest(resize_long=640), Div(255),
                            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), Transpose(
                                (2, 0, 1))
                        ])
                        self.post_func = DBPostProcess({
                            "thresh": 0.3,
                            "box_thresh": 0.5,
                            "max_candidates": 1000,
                            "unclip_ratio": 1.6,
                            "min_size": 1
                        })
                    if boximg["ext_key"] == 'SK_S_UNINO' or boximg["ext_key"] == 'SK_S_UNINO_2':
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
                            "min_size": 1
                        })
                    if boximg["ext_key"] == 'INV_NO2':
                        self.det_preprocess = Sequential([
                            DetResizeForTest(resize_long=2000), Div(255),
                            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), Transpose(
                                (2, 0, 1))
                        ])
                        self.post_func = DBPostProcess({
                            "thresh": 0.3,
                            "box_thresh": 0.5,
                            "max_candidates": 1000,
                            "unclip_ratio": 1.8,
                            "min_size": 1
                        })
                    if boximg["ext_key"] == 'SY_NO1':
                        self.det_preprocess = Sequential([
                            DetResizeForTest(resize_long=300), Div(255),
                            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), Transpose(
                                (2, 0, 1))
                        ])
                        self.post_func = DBPostProcess({
                            "thresh": 0.3,
                            "box_thresh": 0.5,
                            "max_candidates": 1000,
                            "unclip_ratio": 1.8,
                            "min_size": 1
                        })
                    if boximg["ext_key"] == 'SY_NO2':
                        self.det_preprocess = Sequential([
                            DetResizeForTest(resize_long=500), Div(255),
                            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), Transpose(
                                (2, 0, 1))
                        ])
                        self.post_func = DBPostProcess({
                            "thresh": 0.3,
                            "box_thresh": 0.5,
                            "max_candidates": 1000,
                            "unclip_ratio": 1.8,
                            "min_size": 1
                        })
                    if boximg["ext_key"] == 'YYMM':
                        self.det_preprocess = Sequential([
                            DetResizeForTest(resize_long=400), Div(255),
                            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), Transpose(
                                (2, 0, 1))
                        ])
                        self.post_func = DBPostProcess({
                            "thresh": 0.3,
                            "box_thresh": 0.5,
                            "max_candidates": 1000,
                            "unclip_ratio": 1.8,
                            "min_size": 1
                        })
                    if boximg["ext_key"] == 'MM':
                        self.det_preprocess = Sequential([
                            DetResizeForTest(resize_long=200), Div(255),
                            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), Transpose(
                                (2, 0, 1))
                        ])
                        self.post_func = DBPostProcess({
                            "thresh": 0.3,
                            "box_thresh": 0.5,
                            "max_candidates": 1000,
                            "unclip_ratio": 1.8,
                            "min_size": 1
                        })
                    if boximg["ext_key"] == 'dotted_line':
                        self.det_preprocess = Sequential([
                            DetResizeForTest(resize_long=1200), Div(255),
                            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), Transpose(
                                (2, 0, 1))
                        ])
                        self.post_func = DBPostProcess({
                            "thresh": 0.3,
                            "box_thresh": 0.5,
                            "max_candidates": 1000,
                            "unclip_ratio": 1.8,
                            "min_size": 1
                        })
                    if boximg["ext_key"] == 'IMPO_INV_NO':
                        self.det_preprocess = Sequential([
                            DetResizeForTest(resize_long=1000), Div(255),
                            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), Transpose(
                                (2, 0, 1))
                        ])
                        self.post_func = DBPostProcess({
                            "thresh": 0.3,
                            "box_thresh": 0.5,
                            "max_candidates": 1000,
                            "unclip_ratio": 1.8,
                            "min_size": 1
                        })
                elif self.im_type == 'Power_tw':
                    self.det_preprocess = Sequential([
                        DetResizeForTest(resize_long=3000), Div(255),
                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), Transpose(
                            (2, 0, 1))
                    ])
                    self.post_func = DBPostProcess({
                        "thresh": 0.3,
                        "box_thresh": 0.5,
                        "max_candidates": 1000,
                        "unclip_ratio": 1.8,
                        "min_size": 5
                    })
                elif self.im_type == 'Water_tw' or self.im_type == 'Water_twc' or self.im_type == 'Water_twz':
                    self.det_preprocess = Sequential([
                        DetResizeForTest(resize_long=4000), Div(255),
                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), Transpose(
                            (2, 0, 1))
                    ])
                    self.post_func = DBPostProcess({
                        "thresh": 0.3,
                        "box_thresh": 0.5,
                        "max_candidates": 1000,
                        "unclip_ratio": 1.8,
                        "min_size": 5
                    })
                elif self.im_type == 'invoice_A5':
                    self.det_preprocess = Sequential([
                        DetResizeForTest(resize_long=1500), Div(255),
                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), Transpose(
                            (2, 0, 1))
                    ])
                    self.post_func = DBPostProcess({
                        "thresh": 0.3,
                        "box_thresh": 0.5,
                        "max_candidates": 1000,
                        "unclip_ratio": 2.0,
                        "min_size": 5
                    })
                elif self.im_type == 'invoice_A4':
                    self.det_preprocess = Sequential([
                        DetResizeForTest(resize_long=2500), Div(255),
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
                if self.im_list[i]['im_type'] == 'extra':
                    out_dict['ext_key'] = self.im_list[i]['ext_key']
                    draw_img = draw_ocr(
                        self.im_list[i]["image"],
                        dt_boxes)
                    draw_img_save = "/paddle/inference_results/"
                    if not os.path.exists(draw_img_save):
                        os.makedirs(draw_img_save)
                    cv2.imwrite(
                        '/paddle/inference_results/extra_result.jpg',
                        draw_img[:, :, ::-1])
                else:
                    out_dict['mask_list'] = self.im_list[i]['mask_list']
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
            self.im_type = input_dict["im_type"]
            # data = np.frombuffer(self.raw_im, np.uint8)
            # im = cv2.imdecode(data, cv2.IMREAD_COLOR)
            dt_boxes = input_dict["dt_boxes"]
            dt_boxes = self.sorted_boxes(dt_boxes)
            if len(dt_boxes) > 0:
                DrawBoxes.list = copy.deepcopy(dt_boxes)
                det_dic = {"dt_boxes": DrawBoxes.list, "im_type": input_dict["im_type"]}
                if input_dict["im_type"] == 'extra':
                    det_dic['ext_key'] = input_dict['ext_key']
                else:
                    self.raw_im = input_dict["image"]
                    self.mask_list = input_dict["mask_list"]
                    det_dic['mask_list'] = input_dict["mask_list"]
                rec_img = input_dict["image"]
                self.rec_dict_list.append(det_dic)
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
                    image[:, :, ::-1])
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
                        boximg = self.get_rotate_crop_image(rec_img, dt_boxes[box_idx])
                        self.ocr_reader.get_sorted_boxes(rec_img, dt_boxes[box_idx])
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
                text, preb, score = self.ocr_reader.postprocess(
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
                # if not os.path.exists(draw_img_save + 'raw_image.jpg'):
                cv2.imwrite(
                    os.path.join(draw_img_save, os.path.basename('raw_image.jpg')),
                    image[:, :, ::-1])
                res_ob = {"text": text_list, "preb": preb_list, "boxes": DrawBoxes.list}
                res_text = []
                res_boxes = []
                res_preb = []
                res_score = []
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
                        # new_arr = np.frombuffer(item_str, dtype=array_data_type).reshape(array_shape)
                        # item_str = ''.join(np.array2string(item, separator=',', threshold=11e3).splitlines())
                        res_preb.append({'bytes': item_str, 'dtype': array_data_type, 'shape': array_shape})
                        preds_prob = item.max(axis=1)
                        prob_array_shape = preds_prob.shape
                        prob_array_type = preds_prob.dtype.name
                        prob_str = preds_prob.tobytes()
                        # new_arr = np.frombuffer(item_str, dtype=array_data_type).reshape(array_shape)
                        # item_str = ''.join(np.array2string(item, separator=',', threshold=11e3).splitlines())
                        res_score.append({'bytes': prob_str, 'dtype': prob_array_type, 'shape': prob_array_shape})
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

                res = {"text": str(res_text), "preb": str(res_preb), "boxes": str(res_boxes), "im_type": self.im_type, "score": str(score), "raw_score": str(res_score), "mask_list": str(self.mask_list[0])}
                print(str(res_text))
                return res, None, ""
        elif isinstance(fetch_data, list):
            for i, one_batch in enumerate(fetch_data):
                text_list = []
                preb_list = []
                text, preb, score = self.ocr_reader.postprocess(
                    one_batch, with_score=True)
                # for res in one_batch_res:
                text_list.append(text)
                preb_list.append(preb)
                # cv2.namedWindow('first', cv2.WINDOW_NORMAL)
                # cv2.imshow('first', image_cv)
                boxes = DrawBoxes.list

                res_ob = {"text": text_list, "preb": preb_list, "boxes": DrawBoxes.list}
                res_text = []
                res_boxes = []
                res_preb = []
                res_score = []
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
                        preds_prob = item.max(axis=1)
                        prob_array_shape = preds_prob.shape
                        prob_array_type = preds_prob.dtype.name
                        prob_str = preds_prob.tobytes()
                        # new_arr = np.frombuffer(item_str, dtype=array_data_type).reshape(array_shape)
                        # item_str = ''.join(np.array2string(item, separator=',', threshold=11e3).splitlines())
                        res_score.append({'bytes': prob_str, 'dtype': prob_array_type, 'shape': prob_array_shape})
                for boxes in self.rec_dict_list[i]['dt_boxes']:
                    b1 = boxes[0].tolist()
                    b3 = boxes[2].tolist()
                    bb = b1 + b3
                    res_boxes.append(bb)

                res = {"text": str(res_text), "preb": str(res_preb), "boxes": str(res_boxes), "im_type": self.rec_dict_list[i]['im_type'], "score": str(score), "raw_score": str(res_score)}
                print(str(res_text))
                if self.rec_dict_list[i]['im_type'] == 'extra':
                    res['ext_key'] = self.rec_dict_list[i]['ext_key']
                    print(str(score))
                else:
                    res['mask_list'] = str(self.rec_dict_list[i]['mask_list'])
                    image = self.raw_im[..., ::-1]
                    draw_img = draw_ocr(
                        image,
                        boxes)
                    draw_img_save = "/paddle/inference_results/"
                    if not os.path.exists(draw_img_save):
                        os.makedirs(draw_img_save)
                    cv2.imwrite(
                        os.path.join(draw_img_save, os.path.basename('result.jpg')),
                        draw_img[:, :, ::-1])
                    cv2.imwrite(
                        os.path.join(draw_img_save, os.path.basename('raw_image.jpg')),
                        image[:, :, ::-1])
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