import colorsys
import os
import time

import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from PIL import ImageDraw, ImageFont

from nets.MSSDet import MSSDet_model
from utils.utils_bbox import BBoxUtility
from utils.utils import get_classes, resize_image, cvtColor
from utils.anchors import get_anchors

'''
训练自己的数据集必看！
'''
class MSSDet(object):
    _defaults = {

        "model_path"        : 'model_data/MSSDet_weights.h5',
        "classes_path"      : 'model_data/voc_classes.txt',

        "input_shape"       : [300, 300],

        "confidence"        : 0.5,

        "nms_iou"           : 0.45,

        'anchors_size'      : [30, 60, 111, 162, 213, 264, 315],

        "letterbox_image"   : False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"


    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.anchors                        = get_anchors(self.input_shape, self.anchors_size)
        self.num_classes                    = self.num_classes + 1
        

        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.bbox_util = BBoxUtility(self.num_classes, nms_thresh=self.nms_iou)
        self.generate()


    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        

        self.MSSDet = MSSDet_model([self.input_shape[0], self.input_shape[1], 3], self.num_classes)
        self.MSSDet.load_weights(self.model_path, by_name=True)
        print('{} model, anchors, and classes loaded.'.format(model_path))


    def detect_image(self, image, crop = False):

        image_shape = np.array(np.shape(image)[0:2])

        image       = cvtColor(image)

        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        image_data = preprocess_input(np.expand_dims(np.array(image_data, dtype='float32'), 0))

        preds      = self.MSSDet.predict(image_data)

        results     = self.bbox_util.decode_box(preds, self.anchors, image_shape, 
                                                self.input_shape, self.letterbox_image, confidence=self.confidence)

        if len(results[0])<=0:
            return image

        top_label   = np.array(results[0][:, 4], dtype = 'int32')
        top_conf    = results[0][:, 5]
        top_boxes   = results[0][:, :4]

        font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 1)
        

        if crop:
            for i, c in list(enumerate(top_boxes)):
                top, left, bottom, right = top_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])

        image       = cvtColor(image)

        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        image_data = preprocess_input(np.expand_dims(np.array(image_data, dtype='float32'), 0))

        preds      = self.MSSDet.predict(image_data)

        results     = self.bbox_util.decode_box(preds, self.anchors, image_shape, 
                                                self.input_shape, self.letterbox_image, confidence=self.confidence)
        t1 = time.time()
        for _ in range(test_interval):
            preds      = self.MSSDet.predict(image_data)

            results     = self.bbox_util.decode_box(preds, self.anchors, image_shape, 
                                                    self.input_shape, self.letterbox_image, confidence=self.confidence)
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        image_shape = np.array(np.shape(image)[0:2])

        image       = cvtColor(image)

        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        image_data = preprocess_input(np.expand_dims(np.array(image_data, dtype='float32'), 0))

        preds      = self.MSSDet.predict(image_data)

        results     = self.bbox_util.decode_box(preds, self.anchors, image_shape, 
                                                self.input_shape, self.letterbox_image, confidence=self.confidence)

        if len(results[0])<=0:
            return 

        top_label   = results[0][:, 4]
        top_conf    = results[0][:, 5]
        top_boxes   = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])
            
            top, left, bottom, right = box

            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
