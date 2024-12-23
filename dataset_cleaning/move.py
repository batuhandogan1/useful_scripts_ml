import os
import shutil
import argparse

# Argümanları tanımlama
parser = argparse.ArgumentParser(description='Taşınacak dosya sayısını ve dizin yollarını belirleyin.')
parser.add_argument('--src_path', type=str, required=True, help='Kaynak dizin yolu (fotoğrafların olduğu yer)')
parser.add_argument('--dest_path', type=str, required=True, help='Hedef dizin yolu (dosyaların taşınacağı yer)')
parser.add_argument('--count', type=int, required=True, help='Taşınacak fotoğraf ve label dosya sayısı')

args = parser.parse_args()

src_path = args.src_path
dest_path = args.dest_path
count = args.count

# Kaynak dizindeki dosyaları listeleme
files = os.listdir(src_path)

# Dosya adlarını benzersiz olarak toplama
paths = []
for file in files:
    arr = file.split('.')
    if arr[0] not in paths:
        paths.append(arr[0])

# Hedef dizin yoksa oluştur
if not os.path.exists(dest_path):
    os.makedirs(dest_path)

# Dosyaları taşıma
for i in range(min(count, len(paths))):  # Belirtilen sayıda dosyayı işleme
    plain_path = paths[i]
    
    # Fotoğraf dosyasını taşıma
    image_src = os.path.join(src_path, plain_path + '.jpg')
    image_dest = os.path.join(dest_path, plain_path + '.jpg')
    if os.path.exists(image_src):
        shutil.copy(image_src, image_dest)
        print(f"{image_src} taşındı.")

    # Label dosyasını taşıma
    label_src = os.path.join(src_path, plain_path + '.txt')
    label_dest = os.path.join(dest_path, plain_path + '.txt')
    if os.path.exists(label_src):
        shutil.copy(label_src, label_dest)
        print(f"{label_src} taşındı.")
