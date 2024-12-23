import cv2
import os
from yolo_show_annos import get_anno
import argparse

parser = argparse.ArgumentParser(description='A tool for cleaning a dataset quickly (YOLO format only)')
parser.add_argument('--path', type=str, required=True, help='Path for dataset containing images and annotations')

args = parser.parse_args()

path = args.path

files = os.listdir(path)
paths = []
bbox = []

# Dosya adlarını benzersiz olarak topluyoruz
for file in files:
    arr = file.split('.')
    if arr[0] not in paths:
        paths.append(arr[0])

i = 0  # Başlangıçta ilk dosya

while 0 <= i < len(paths):
    plain_path = paths[i]
    image = get_anno(path + plain_path, i)
    cv2.imshow('image', image)

    k = cv2.waitKey(0)
    
    if k == 115:  # 's' tuşu
        os.remove(path + plain_path + '.jpg')
        os.remove(path + plain_path + '.txt')
        print(f"{plain_path}.jpg ve {plain_path}.txt silindi.")
        paths.pop(i)  # Silinen görüntü ve anotasyonu listeden kaldır
        if i >= len(paths):  # Silinen son elemandan sonra liste sınırını aşmamak için kontrol
            i -= 1  # Eğer son eleman silinmişse bir önceki elemana dön
    elif k == 107:  # 'k' tuşu
        continue
    elif k == 81:  # Sol ok tuşu
        i -= 1  # Bir önceki görüntüye git
        if i < 0:
            i = 0  # Listenin başında kalmak için kontrol
    elif k == 83:  # Sağ ok tuşu
        i += 1  # Bir sonraki görüntüye git

    cv2.destroyAllWindows()
