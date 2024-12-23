import os
import shutil
import argparse



parser = argparse.ArgumentParser(description='This script divides dataset to train and val set')

parser.add_argument('--source_path', type=str, required=True, help='Dataset path needs to contain images and annotations')
parser.add_argument('--export_path', type=str, required=True, help='Export folder path will contain train and val files')
parser.add_argument('--split_ratio', type=int, default=10, help='Train val split ratio (EX: 10 for 9:1)')

args = parser.parse_args()


path = os.path.join(args.export_path + '/train')
os.mkdir(path)
path = os.path.join(args.export_path + '/train/' + 'images')
os.mkdir(path)
path = os.path.join(args.export_path + '/train/' + 'labels')
os.mkdir(path)

path = os.path.join(args.export_path + '/val')
os.mkdir(path)
path = os.path.join(args.export_path + '/val/' + 'images')
os.mkdir(path)
path = os.path.join(args.export_path + '/val/' + 'labels')
os.mkdir(path)

dataset = os.listdir(args.source_path)
data_count = len(dataset)
val_count = int(data_count / args.split_ratio)

if (val_count % 2 != 0):
    val_count -= 1

val_image_count = val_count / 2

for file in dataset:
    if file.endswith('.jpg'):
        temp = file.split('.jpg')

        shutil.move(args.source_path + '/' + temp[0] + '.jpg', args.export_path + '/val/' + 'images/' + temp[0] + '.jpg')
        shutil.move(args.source_path + '/' + temp[0] + '.txt', args.export_path + '/val/' + 'labels/' + temp[0] + '.txt')

        val_image_count -= 1

        if (val_image_count == 0):
            break

dataset = os.listdir(args.source_path)

for file in dataset:
    if file.endswith('.jpg'):
        shutil.move(args.source_path + '/' + file, args.export_path + '/train/' + 'images/' + file)
    
    if file.endswith('.txt'):
        shutil.move(args.source_path + '/' + file, args.export_path + '/train/' + 'labels/' + file)