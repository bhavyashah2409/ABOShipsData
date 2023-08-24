import os
import cv2 as cv
import shutil as s
import pandas as pd
import imagesize as im

# CREATE FOLDERS AND SUBFOLDERS
DESTINATION_FOLDER = 'Dataset'
AREA_THRESHOLD = 1000
if not os.path.exists(DESTINATION_FOLDER):
    os.mkdir(DESTINATION_FOLDER)
if not os.path.exists(os.path.join(DESTINATION_FOLDER, 'train')):
    os.mkdir(os.path.join(DESTINATION_FOLDER, 'train'))
if not os.path.exists(os.path.join(DESTINATION_FOLDER, 'train', 'images')):
    os.mkdir(os.path.join(DESTINATION_FOLDER, 'train', 'images'))
if not os.path.exists(os.path.join(DESTINATION_FOLDER, 'train', 'labels')):
    os.mkdir(os.path.join(DESTINATION_FOLDER, 'train', 'labels'))
if not os.path.exists(os.path.join(DESTINATION_FOLDER, 'test')):
    os.mkdir(os.path.join(DESTINATION_FOLDER, 'test'))

# READ LABELS FILE
df = pd.read_csv('Vesibussi_Labels.csv')
df = pd.DataFrame(df)

# READ CLASSES FILE
classes = open('classes.txt', 'r').read().split('\n')
classes.remove('')

# CONVERT LABELS TO YOLO FORMAT
def convert_to_yolo(a):
    filename = a['filename']
    xmin = a['xmin']
    ymin = a['ymin']
    xmax = a['xmax']
    ymax = a['ymax']
    c = a['class']
    img_w, img_h = im.get(os.path.join('Seaships', filename[:8], filename + '.png'))
    x = (xmin + xmax) / (2.0 * img_w)
    y = (ymin + ymax) / (2.0 * img_h)
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    c = classes.index(c)
    return c, x, y, w, h

df['bbox'] = df.apply(lambda a: convert_to_yolo(a), axis=1)
df = df.drop(columns=['width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
df = df.groupby('filename').agg({'bbox': list})
df['filename'] = df.index
df.reset_index(drop=True, inplace=True)

# WRITE IMAGES AND LABELS TO TXT FILE
def write_images_and_labels_above_threshold(a, threshold):
    img_path = os.path.join('Seaships', a['filename'][:8] , a['filename'] + '.png')
    img = cv.imread(img_path)
    img_h, img_w, _ = img.shape
    train_img = os.path.join(DESTINATION_FOLDER, 'train', 'images', a['filename'] + '.png')
    cv.imwrite(train_img, img)
    train_label = os.path.join(DESTINATION_FOLDER, 'train', 'labels', a['filename'] + '.txt')
    with open(train_label, 'w') as f:
        for c, x, y, w, h in a['bbox']:
            if w * h > threshold / (img_h * img_w):
                f.write(f'{c} {x} {y} {w} {h}\n')
        f.close()
    return train_img, train_label

train_df = df.apply(lambda a: write_images_and_labels_above_threshold(a, AREA_THRESHOLD), axis=1, result_type='expand')
train_df.columns = ['image', 'label']

# GET NUMBER OF IMAGES
print(train_df.head())
print('TRAIN IMAGES:', train_df.shape[0])

all_images = []
for folder in os.listdir('Seaships'):
    all_images = all_images + os.listdir(os.path.join('Seaships', folder))
all_images = set([os.path.splitext(image)[0] for image in all_images])
train_images = set(df['filename'].to_list())
test_images = all_images.difference(all_images.intersection(train_images))
print('TEST IMAGES:', len(test_images))

s.copyfile('classes.txt', os.path.join(DESTINATION_FOLDER, 'classes.txt'))

# SAVE DATAFRAMES
print(df.head())
train_df.to_csv(os.path.join(DESTINATION_FOLDER, 'train_df.csv'), index=False)
test_df = pd.DataFrame({'image': list(test_images)})
test_df.to_csv(os.path.join(DESTINATION_FOLDER, 'test_df.csv'), index=False)
