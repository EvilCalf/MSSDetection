import os
import random
import xml.etree.ElementTree as ET

import numpy as np

from utils.utils import get_classes

annotation_mode     = 0
classes_path        = 'model_data/SA-HP.txt'

trainval_percent    = 0.8
train_percent       = 0.8

data_path  = 'VOCdevkit'

data_sets  = [('SA-HP', 'train'), ('SA-HP', 'val')]
classes, _      = get_classes(classes_path)

photo_nums  = np.zeros(len(data_sets))
nums        = np.zeros(len(classes))
def convert_annotation(pathname, image_id, list_file):
    in_file = open(os.path.join(data_path, '%s/Annotations/%s.xml'%(pathname, image_id)), encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        
        nums[classes.index(cls)] = nums[classes.index(cls)] + 1
        
if __name__ == "__main__":
    random.seed(0)
    if " " in os.path.abspath(data_path):
        raise ValueError("数据集存放的文件夹路径与图片名称中不可以存在空格，否则会影响正常的模型训练，请注意修改。")

    if annotation_mode == 0 or annotation_mode == 1:
        print("Generate txt in ImageSets.")
        xmlfilepath     = os.path.join(data_path, 'SA-HP/Annotations')
        saveBasePath    = os.path.join(data_path, 'SA-HP/ImageSets')
        temp_xml        = os.listdir(xmlfilepath)
        total_xml       = []
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)

        num     = len(total_xml)  
        list    = range(num)  
        tv      = int(num*trainval_percent)  
        tr      = int(tv*train_percent)  
        trainval= random.sample(list,tv)  
        train   = random.sample(trainval,tr)  
        
        print("train and val size",tv)
        print("train size",tr)
        ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
        ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
        ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
        fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
        
        for i in list:  
            name=total_xml[i][:-4]+'\n'  
            if i in trainval:  
                ftrainval.write(name)  
                if i in train:  
                    ftrain.write(name)  
                else:  
                    fval.write(name)  
            else:  
                ftest.write(name)  
        
        ftrainval.close()  
        ftrain.close()  
        fval.close()  
        ftest.close()
        print("Generate txt in ImageSets done.")

    if annotation_mode == 0 or annotation_mode == 2:
        print("Generate train.txt and val.txt for train.")
        type_index = 0
        for pathname, image_set in data_sets:
            image_ids = open(os.path.join(data_path, '%s/ImageSets/%s.txt'%(pathname, image_set)), encoding='utf-8').read().strip().split('\n')
            list_file = open('%s_%s.txt'%(pathname, image_set), 'w', encoding='utf-8')
            for image_id in image_ids:
                list_file.write('%s/%s/JPEGImages/%s.jpg'%(os.path.abspath(data_path), pathname, image_id))

                convert_annotation(pathname, image_id, list_file)
                list_file.write('\n')
            photo_nums[type_index] = len(image_ids)
            type_index += 1
            list_file.close()
        print("Generate train.txt and val.txt for train done.")
        
        def printTable(List1, List2):
            for i in range(len(List1[0])):
                print("|", end=' ')
                for j in range(len(List1)):
                    print(List1[j][i].rjust(int(List2[j])), end=' ')
                    print("|", end=' ')
                print()

        str_nums = [str(int(x)) for x in nums]
        tableData = [
            classes, str_nums
        ]
        colWidths = [0]*len(tableData)
        len1 = 0
        for i in range(len(tableData)):
            for j in range(len(tableData[i])):
                if len(tableData[i][j]) > colWidths[i]:
                    colWidths[i] = len(tableData[i][j])
        printTable(tableData, colWidths)

        if photo_nums[0] <= 500:
            print("训练集数量小于500，属于较小的数据量，请注意设置较大Epoch以满足足够的梯度下降次数Step。")

        if np.sum(nums) == 0:
            print("在数据集中并未获得任何目标")