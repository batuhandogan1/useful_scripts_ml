# annotations içinde gezicek 
#   image ı bulucak vehicle type detection modelini çalıştıracak
#   eğer araç bulursa bulduğu araç ve araçların içerisinde anno[bbox] var mı diye bakacak
#   varsa markayı ve category id ye göre modeli yazdıracak

from tools.divide2bb.divide2bb import Divide2BB

import os
import cv2
import json

class ExtractMMR(Divide2BB):
    def __init__(self, bbox_arr, path_arr, class_arr):
        self.bbox_arr = bbox_arr
        self.path_arr = path_arr
        self.class_arr = class_arr
        self.extract()


    def extract(self):

        Dict = {1: 'pickup', 2: 'hatchback', 3: 'sedan', 4: 'light-truck', 5: 'heavy-truck', 6: 'bus', 7: 'van', 8: 'minivan', 9: 'sport', 10: 'suv'}


        f = open(f'/home/bdogan/Desktop/annotations/mmr_set6.json')

        destination = '/home/bdogan/Desktop/dataset/'


        data = json.load(f)

        for image in data['images']:

            for anno in data['annotations']:

                if anno['attributes'].get('mark') != None and anno.get('bbox') != None and anno.get('category_id') != None:
                    if image['id'] == anno['image_id']:

                        for i, bbox in enumerate(self.bbox_arr):
                            divided_name = (str(self.path_arr[i])).split('/')
                            if image['file_name'] == f'{divided_name[-2]}/{divided_name[-1]}':
                                
                                img = cv2.imread(str(self.path_arr[i]))
                                height, width, channels = img.shape

                                new_bbox = super().xywh2xyxy((width, height), bbox)
                                class_name = anno['attributes']['mark'] + ' - ' + Dict[anno['category_id']]

                                if super().is_contain(list(new_bbox), list(anno['bbox'])) == True:

                                    cropped_image = img[int(new_bbox[1]) : int(new_bbox[3]),
                                    int(new_bbox[0]) : int(new_bbox[2])]

                                    if not os.path.exists(os.path.join(os.getcwd(), f'{destination}/{class_name}')):
                                        os.makedirs(os.path.join(os.getcwd(), f'{destination}/{class_name}'))
 
                                    cv2.imwrite(destination + '/' + class_name + '/' + str(divided_name[-1]), cropped_image)

                        # class_name = anno['attributes']['mark'] + ' - ' + anno['attributes']['type']

                        # if not os.path.exists(destination + class_name):
                        #     os.makedirs(destination + class_name)

                        # img_name = image['file_name'].split('/')
                        # img = cv2.imread(source + '/' + set_name + '/' + img_name[1])

                        # x, y, w, h = anno['bbox']
                        # crop_img = img[int(y):int(y)+int(h), int(x):int(x)+int(w)]
                        # cv2.imwrite(destination + '/' + class_name + '/' + img_name[1], crop_img)
                
        f.close()