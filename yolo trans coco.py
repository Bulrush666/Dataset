###########################################################################################################################
import json
import os
import cv2

"""
Step1：需要先在root_path；路径下创建classes.txt！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
Step2：images（原图文件夹）也需要复制到root_path下！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
Step3：修改下面前五个变量的路径或者名称
"""

######################################第一步##################################################
#将yolo格式的标签：classId, xCenter, yCenter, w, h转换为coco格式：classId, xMin, yMim, xMax,     #
# yMax格式。coco的id编号从1开始计算，所以这里classId应该从1开始计算。最终annos.txt中每行为imageName,   #
# classId, xMin, yMim, xMax, yMax, 一个bbox对应一行                                           #
#############################################################################################
# 原始标签路径 txt cls 中心点的位置+宽高
originLabelsDir = r'/root/tf-logs/shangke/test/labels'
# 转换后的文件保存路径 输出结果
saveDir = r'/root/tf-logs/shangke/test2'
# 原始标签对应的图片路径
originImagesDir = r'/root/tf-logs/shangke/test/images'
# 以及annotations文件夹(如果没有则会自动创建，用于保存最后的json)
root_path = r'/root/tf-logs/shangke'
# 用于创建训练集或验证集
phase = 'train'  # 需要修正





txtFileList = os.listdir(originLabelsDir)
i = 0
with open(saveDir, 'w') as fw:
    for txtFile in txtFileList:
        with open(os.path.join(originLabelsDir, txtFile), 'r') as fr:
            labelList = fr.readlines()
            i += 1
            if i == len(txtFileList) -1:
                break
            for label in labelList:
                label = label.strip().split()
                x = float(label[1])
                y = float(label[2])
                w = float(label[3])
                h = float(label[4])

                # convert x,y,w,h to x1,y1,x2,y2
                imagePath = os.path.join(originImagesDir,
                                         txtFile.replace('txt', 'jpg'))
                image = cv2.imread(imagePath)
                H, W, _ = image.shape
                x1 = (x - w / 2) * W
                y1 = (y - h / 2) * H
                x2 = (x + w / 2) * W
                y2 = (y + h / 2) * H
                # 为了与coco标签方式对，标签序号从1开始计算
                fw.write(txtFile.replace('txt', 'jpg') + ' {} {} {} {} {}\n'.format(int(label[0]) + 1, x1, y1, x2, y2))

        print('{} done   {} {}'.format(txtFile, i, len(txtFileList)))


######################################第二步##################################################
#将标签转换为coco格式并以json格式保存，代码如下。根路径root_path中，包含images(图片文件夹)              #
# ，annos.txt(bbox标注)，classes.txt(一行对应一种类别名字), 以及annotations文件夹(如果没有则会自动创建 #
# 用于保存最后的json)                                                                          #
#############################################################################################
# ------------用os提取images文件夹中的图片名称，并且将BBox都读进去------------

# 根路径，里面包含images(图片文件夹)，annos.txt(bbox标注)，classes.txt(类别标签),


# dataset用于保存所有数据的图片信息和标注信息
dataset = {'categories': [], 'annotations': [], 'images': []}

# 打开类别标签
with open(os.path.join(root_path, 'classes.txt')) as f:
    classes = f.read().strip().split()

# 建立类别标签和数字id的对应关系
for i, cls in enumerate(classes, 1):
    dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})

# 读取images文件夹的图片名称
indexes = os.listdir(os.path.join(root_path, 'images'))

# 统计处理图片的数量
global count
count = 0

# 读取Bbox信息
with open(os.path.join(root_path, 'annos.txt')) as tr:
    annos = tr.readlines()

    # ---------------接着将，以上数据转换为COCO所需要的格式---------------
    for k, index in enumerate(indexes):
        count += 1
        # 用opencv读取图片，得到图像的宽和高
        im = cv2.imread(os.path.join(root_path, 'images/') + index)
        height, width, _ = im.shape

        # 添加图像的信息到dataset中
        dataset['images'].append({'file_name': index,
                                  'id': k,
                                  'width': width,
                                  'height': height})

        for ii, anno in enumerate(annos):
            parts = anno.strip().split()

            # 如果图像的名称和标记的名称对上，则添加标记
            if parts[0] == index:
                # 类别
                cls_id = parts[1]
                # x_min
                x1 = float(parts[2])
                # y_min
                y1 = float(parts[3])
                # x_max
                x2 = float(parts[4])
                # y_max
                y2 = float(parts[5])
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                dataset['annotations'].append({
                    'area': width * height,
                    'bbox': [x1, y1, width, height],
                    'category_id': int(cls_id),
                    'id': i,
                    'image_id': k,
                    'iscrowd': 0,
                    # mask, 矩形是从左上角点按顺时针的四个顶点
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })

        print('{} images handled'.format(count))

# 保存结果的文件夹
folder = os.path.join(root_path, 'annotations')
if not os.path.exists(folder):
    os.makedirs(folder)
json_name = os.path.join(root_path, 'annotations/{}.json'.format(phase))
with open(json_name, 'w') as f:
    json.dump(dataset, f)

if __name__ == "__main__":
    # 原始标签路径
    originLabelsDir = '/root/tf-logs/shangke/train/labels'
    # 转换后的文件保存路径
    saveTempTxt = '/root/tf-logs/shangke/train2'
    # 原始标签对应的图片路径
    originImagesDir ='/root/tf-logs/shangke/train/images'
    yolo2txt(originLabelsDir, originImagesDir, saveTempTxt)

