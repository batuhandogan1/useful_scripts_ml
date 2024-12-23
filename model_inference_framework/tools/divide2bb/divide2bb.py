import cv2
import numpy as np
import os

class Divide2BB():
    def __init__(self, bbox_arr, path_arr, class_arr):
        self.bboxes = []
        self.paths = []
        self.classes = []

        self.bboxes = bbox_arr.copy()
        self.bboxes = np.array(self.bboxes)

        self.paths = path_arr.copy()
        self.paths = np.array(self.paths)

        self.classes = class_arr.copy()
        self.classes = np.array(self.classes)
    


    def xywh2xyxy(self, size, box):
        """
        Param:  self
                +size: (Tuple)
                +box: (Numpy array)

        Using:  +size: (Tuple)
                +box: (Numpy array)

        Does:   Gets a numpy xywh Yolo format
                array and transforms to x1y1x2y2  format.
        
        Return: (x1, y1, x2, y2)
        """
        width, height = size
        x_center, y_center, w, h = box
        x1 = (x_center - w / 2) * width
        y1 = (y_center - h / 2) * height
        x2 = (x_center + w / 2) * width
        y2 = (y_center + h / 2) * height
        return (x1, y1, x2, y2)



    def xyxyx2xywh(self, size, box):
        """
        Param:  self
                +size: (Tuple)
                +box: (Numpy array)

        Using:  +size: (Tuple)
                +box: (Numpy array)

        Does:   Gets a numpy x1y1x2y2 array and
                transforms to xywh YOLO format.
        
        Return: x, y, w, h
        """
        img_width, img_height = size
        x1, y1, x2, y2 = box

        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        
        width = x2 - x1
        height = y2 - y1
        
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        
        return x_center, y_center, width, height



    def get_bboxes(self):
        return self.bboxes
    


    def get_pathes(self):
        return self.paths
    


    def get_classes(self):
        return self.classes
    


    def is_contain(self, big_box, small_box):
        """
        Param:  self
                +big_box: (Numpy array)
                +small_box: (Numpy array)

        Using:  +big_box: (Numpy array)
                +small_box: (Numpy array)

        Does:   If big box contains small box returns True.
                Else False.
        
        Return: True or False
        """
        x1_box_big, y1_box_big, x2_box_big, y2_box_big = big_box
        x1_box_small, y1_box_small, x2_box_small, y2_box_small = small_box

        if (x1_box_big < x1_box_small) and (y1_box_big < y1_box_small) and (x2_box_big > x2_box_small) and (y2_box_big > y2_box_small):
            return True
        else:
            return False



    def calculate_new_bounding_box(self, big_box, small_box):
        """
        Param:  self
                +big_box: (Numpy array)
                +small_box: (Numpy array)

        Using:  +big_box: (Numpy array)
                +small_box: (Numpy array)

        Does:   For old_big_box and old_small_box coordinates. Assume old_big_box
                x1y1 coordinate is 0,0 and calculate new_small_box coordinate.
        
        Return: True or False
        """
        x1_box_big, y1_box_big, x2_box_big, y2_box_big = big_box
        x1_box_small, y1_box_small, x2_box_small, y2_box_small = small_box

        new_small_box = [x1_box_small-x1_box_big , y1_box_small-y1_box_big , x2_box_small-x1_box_big , y2_box_small-y1_box_big]
        
        return np.array(new_small_box)



    def divide_images(self):
        """
        Param:  self

        Using:  +self.bboxes: (Numpy array)
                +self.paths: (Numpy array)
                **self.xywh2xyxy

        Does:   Read images from +self.paths, for every image switch bboxes yolo format to x1y1x2y2 format.
                For corresponding bbox crop image and save to current path/output images.
                Then print remaining number of images.
        
        Return: None
        """
        if self.bboxes.any() != None:
             
            count = 1
            while True:
                
                if not os.path.exists(os.path.join(os.getcwd(), f'divide2bb_runs/exp{count}')):
                    os.makedirs(os.path.join(os.getcwd(), f'divide2bb_runs/exp{count}'))
                    break

                elif count == 10_000:
                    break
                
                else:
                    count += 1


        for index in range(len(self.paths)):

            image = cv2.imread(str(self.paths[index]))
            height, width, channels = image.shape

            self.bboxes[index] = self.xywh2xyxy((width, height), self.bboxes[index])

            # self.bboxes[index][0] = x1
            # self.bboxes[index][1] = y1
            # self.bboxes[index][2] = x2
            # self.bboxes[index][3] = y2

            cropped_image = image[int(self.bboxes[index][1]) : int(self.bboxes[index][3]),
                                    int(self.bboxes[index][0]) : int(self.bboxes[index][2])]

            image_name = os.path.split(self.paths[index])[-1]

            cv2.imwrite(f'./divide2bb_runs/exp{count}/' + str(index) + image_name, cropped_image)
            print(f'({index + 1}/{len(self.paths)}) image wroted to {os.getcwd()}/divide2bb_runs/exp{count}')


        
    def divide_images_classification(self):
        """
        Param:  self

        Using:  +self.bboxes: (Numpy array)
                +self.paths: (Numpy array)
                +self.classes: (Numpy array)
                **self.xywh2xyxy

        Does:   Read images from +self.paths, for every image switch bboxes yolo format to x1y1x2y2 format.
                For corresponding bbox crop image and save to current 
                path/divide2bb_runs/exp{count}/{self.classes[index]} images.
                In other words, it puts all data in the appropriate folders according to the class name.
                Then print remaining number of images.
        
        Return: None
        """
        if self.bboxes.any() != None:
             
            count = 1
            while True:
                
                if not os.path.exists(os.path.join(os.getcwd(), f'divide2bb_runs/exp{count}')):
                    os.makedirs(os.path.join(os.getcwd(), f'divide2bb_runs/exp{count}'))
                    break

                elif count == 10_000:
                    break
                
                else:
                    count += 1


        for index in range(len(self.paths)):

            image = cv2.imread(str(self.paths[index]))
            height, width, channels = image.shape

            self.bboxes[index] = self.xywh2xyxy((width, height), self.bboxes[index])

            # self.bboxes[index][0] = x1
            # self.bboxes[index][1] = y1
            # self.bboxes[index][2] = x2
            # self.bboxes[index][3] = y2

            cropped_image = image[int(self.bboxes[index][1]) : int(self.bboxes[index][3]),
                                    int(self.bboxes[index][0]) : int(self.bboxes[index][2])]

            image_name = os.path.split(self.paths[index])[-1]

            if not os.path.exists(os.path.join(os.getcwd(), f'divide2bb_runs/exp{count}/{self.classes[index]}')):
                os.makedirs(os.path.join(os.getcwd(), f'divide2bb_runs/exp{count}/{self.classes[index]}'))

            cv2.imwrite(f'./divide2bb_runs/exp{count}/{self.classes[index]}/' + str(index) + image_name, cropped_image)
            print(f'({index + 1}/{len(self.paths)}) image wroted to {os.getcwd()}/divide2bb_runs/exp{count}/{self.classes[index]}')



    def divide_images_detection(self):
        """
        Param:  self

        Using:  +self.bboxes: (Numpy array)
                +self.paths: (Numpy array)
                +self.classes: (Numpy array)
                **self.xywh2xyxy

        Does:   Read images from +self.paths, for every image switch bboxes yolo format to x1y1x2y2 format.
                Inference image on given model. Crop for detected bboxes. If the bounding box contains at least
                one annotation, write this annotation to a new txt file named after the newly cropped image
                by editing its coordinates. Save image and corresponding annotation to path/divide2bb_runs/exp{count} images.
        
        Return: None
        """
        if self.bboxes.any() != None:
             
            count = 1
            while True:
                
                if not os.path.exists(os.path.join(os.getcwd(), f'divide2bb_runs/exp{count}')):
                    os.makedirs(os.path.join(os.getcwd(), f'divide2bb_runs/exp{count}'))
                    break

                elif count == 10_000:
                    break
                
                else:
                    count += 1

        labeled_image_count = 1
        
        for index in range(len(self.paths)):

            image = cv2.imread(str(self.paths[index]))
            height, width, channels = image.shape

            self.bboxes[index] = self.xywh2xyxy((width, height), self.bboxes[index])

            # self.bboxes[index][0] = x1
            # self.bboxes[index][1] = y1
            # self.bboxes[index][2] = x2
            # self.bboxes[index][3] = y2

            cropped_image = image[int(self.bboxes[index][1]) : int(self.bboxes[index][3]),
                                    int(self.bboxes[index][0]) : int(self.bboxes[index][2])]

            image_name = os.path.split(self.paths[index])[-1]
            before_extension = (image_name.split('.jpg'))[0]

            temp_path = ((str(self.paths[index])).split('.jpg'))[0]
            txt_name = temp_path + '.txt'
            

            if os.path.exists(txt_name):

                if os.path.getsize(txt_name) > 0:

                    reading_file = open(txt_name,"r")
                    for line in reading_file:

                        list_line = line.strip().split(' ')
                        small_box = [float(list_line[1]), float(list_line[2]), float(list_line[3]), float(list_line[4])]
                        small_box = self.xywh2xyxy((width, height), small_box)
                        small_box = np.array(small_box)

                        if (self.is_contain(self.bboxes[index], small_box)):

                            new_small_box = self.calculate_new_bounding_box(self.bboxes[index], small_box)
                            new_img_width = int(self.bboxes[index][2]) - int(self.bboxes[index][0])
                            new_img_height = int(self.bboxes[index][3]) - int(self.bboxes[index][1])
                            new_small_box_xywh = self.xyxyx2xywh((new_img_width, new_img_height), new_small_box)

                            file = open(f'./divide2bb_runs/exp{count}/' + str(index) + before_extension + '.txt', 'a')
                            file.write(f"{list_line[0]} {new_small_box_xywh[0]} {new_small_box_xywh[1]} {new_small_box_xywh[2]} {new_small_box_xywh[3]}\n")
                            file.close()

                            cv2.imwrite(f'./divide2bb_runs/exp{count}/' + str(index) + image_name, cropped_image)
                            print(f'({labeled_image_count}) image wroted to {os.getcwd()}/divide2bb_runs/exp{count}')
                            labeled_image_count += 1

                else:
                    print(txt_name + ' is contain nothing.')
            
            else:
                print('There is no such file -> ' + temp_path + '.txt')