import cv2
import os


def get_anno(path, i):

    image = cv2.imread(path + '.jpg')

    h, w, c = image.shape
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    bboxes = []

    color_dict = {0: (0, 0, 255),
                  1: (0, 128, 255),
                  2: (0, 255, 255),
                  3: (0, 255, 0),
                  4: (255, 255, 0),
                  5: (255, 0, 0),
                  6: (255, 0, 127),
                  7: (255, 0, 255),
                  8: (127, 0, 255)}

    f = open(path + ".txt", "r")
    for line in f:
        arr = line.split()
        bboxes.append([arr[0], arr[1], arr[2], arr[3], arr[4]])

    for bbox in bboxes:
        class_name = int(bbox[0])

        # bbox[0] = class
        # bbox[1] = x
        # bbox[2] = y
        # bbox[3] = width
        # bbox[4] = height

        width = float(bbox[3])
        height = float(bbox[4])

        xc = float(bbox[1]) * w
        yc = float(bbox[2]) * h

        box_width = width * w
        box_height = height * h

        x1 = xc - (box_width / 2)
        x2 = xc + (box_width / 2)

        y1 = yc - (box_height / 2)
        y2 = yc + (box_height / 2)

        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)

        start = (x1, y1)
        end = (x2, y2)

        color = (color_dict[class_name])

        image = cv2.rectangle(image, start, end, color, thickness)
        image = cv2.putText(image, str(class_name), (x1, y1 - 5), font, fontScale, color, 1, cv2.LINE_AA)
        image = cv2.putText(image, str(i), (5, 20), font, fontScale, (255, 255, 255), 1, cv2.LINE_AA)

    return image



if __name__ == "__main__":

    path = 'C:/Users/batuhan/Desktop/File/'

    files = os.listdir(path)
    paths = []
    bbox = []

    for file in files:
        arr = file.split('.')

        if arr[0] not in paths:
            paths.append(arr[0])

    for plain_path in paths:
        image = get_anno(path + plain_path)
        cv2.imshow('resim', image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()