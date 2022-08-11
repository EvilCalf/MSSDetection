import math
min_dim = 224   #######维度  
# conv4_3 ==> 38 x 38  
# fc7 ==> 19 x 19  
# conv6_2 ==> 10 x 10  
# conv7_2 ==> 5 x 5  
# conv8_2 ==> 3 x 3  
# conv9_2 ==> 1 x 1  
mbox_source_layers = ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']   
# in percent %  
min_ratio = 20   
max_ratio = 90  

step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 2)))
min_sizes = []   
max_sizes = []  
for ratio in range(min_ratio, max_ratio + 1, step): 

  min_sizes.append(min_dim * ratio / 100.)  
  max_sizes.append(min_dim * (ratio + step) / 100.)  
min_sizes = [min_dim * 10 / 100.] + min_sizes  
max_sizes = [min_dim * 20 / 100.] + max_sizes  
steps = [8, 16, 32, 64, 100, 300]  #计算卷积层产生的prior_box距离原图的步长，先验框中心点的坐标会乘以step  
aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]  #横纵比


voc = {
    #'num_classes': 21,
    'num_classes': 2,
    #'max_iter': 10000,
    'feature_maps': [112, 56, 28, 14, 14, 14],
    'min_dim': 224,
    'steps': [2, 4, 8, 16, 16, 16, 16],
    'min_sizes': [35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.0],
    'max_sizes': [76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6],
    'aspect_ratios': [[2,3], [2, 3], [2, 3], [2, 3],[2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}