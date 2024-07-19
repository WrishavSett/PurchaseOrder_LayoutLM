import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm


import ast
import glob
from PIL import Image
from lxml import etree
from pytesseract import pytesseract
from shapely.geometry import Polygon
from sklearn.model_selection import train_test_split


pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


f = open(r'annotated_dataset\\json_format.json')
label_studio_data = json.load(f)


def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    # print(poly_1,poly_2)
    # iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    iou = poly_1.intersection(poly_2).area
    min_area = min(poly_1.area,poly_2.area)
    return iou/min_area


def hocr_to_dataframe(fp):
    doc = etree.parse(fp)
    words = []
    wordConf = []
    coords_list = []
    for path in doc.xpath('//*'):
        if 'ocrx_word' in path.values():
            coord_text = path.values()[2].split(';')[0].split(' ')[1:] 
            word_coord = list(map(int, coord_text)) #x1, y1, x2, y2
            conf = [x for x in path.values() if 'x_wconf' in x][0]
            wordConf.append(int(conf.split('x_wconf ')[1]))
            words.append(path.text)
            coords_list.append(word_coord)

    dfReturn = pd.DataFrame({'word' : words,
                             'coords': coords_list,
                             'confidence' : wordConf})

    return(dfReturn)


document_data = dict()
document_data['file_name'] = []
document_data['labelled_bbox']= []

for i in range(len(label_studio_data)):
    row = label_studio_data[i]
    file_name = os.path.basename(row['data']['image'])
    label_list, labels, bboxes = [], [], []

    for label_ in row['annotations'][0]['result']:
        label_value = label_['value']
        x, y, w, h = label_value['x'], label_value['y'], label_value['width'], label_value['height']
        original_w , original_h = label_['original_width'], label_['original_height']

        x1 = int((x * original_w) / 100)
        y1 = int((y * original_h) / 100)
        x2 = x1 + int(original_w*w / 100)
        y2 = y1 + int(original_h*h / 100)
        
        label = label_value['rectanglelabels']
        label_list.append((label, (x1,y1,x2,y2), original_h, original_w))
        
    document_data['file_name'].append(file_name)    
    document_data['labelled_bbox'].append(label_list)        

custom_dataset = pd.DataFrame(document_data)


import pandas as pd

# Assuming df is your DataFrame
custom_dataset.to_csv('data.txt', sep='\t', index=False)


label2id = {"Order ID": 0, "Order Date": 1, "Customer Name": 2}
print(label2id)
id2label = {v:k for k, v in label2id.items()}
print(id2label)


final_list = []
    
for i in tqdm(custom_dataset.iterrows(), total=custom_dataset.shape[0]):
    custom_label_text = {}
    word_list = []
    ner_tags_list  = []
    bboxes_list = []
    
    file_name = i[1]['file_name']
    for image in glob.glob('annotated_dataset/coco_format/images/*.jpg'): #Make sure you add your extension or change it based on your needs 
        frame_file_name = os.path.basename(image)
        if frame_file_name == file_name:
            custom_label_text['id'] = i[0]
            image_basename = os.path.basename(image)
            custom_label_text['file_name'] = image_basename
            annotations = []
            label_coord_list = i[1]['labelled_bbox']
            for label_coord in label_coord_list:
                (x1,y1,x2,y2) = label_coord[1]
                box1 = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]] 
                label = label_coord[0][0]
                base_name = os.path.join(os.path.basename(image).split('.')[0])
                pytesseract.run_tesseract(image, base_name, extension='box', lang=None, config="hocr")
                hocr_file = os.path.join(base_name+'.hocr')
                hocr_df = hocr_to_dataframe(hocr_file)
                for word in hocr_df.iterrows():
                    coords = word[1]['coords']
                    (x1df,y1df,x2df,y2df) = coords
                    box2 = [[x1df, y1df], [x2df, y1df], [x2df, y2df], [x1df, y2df]]
                    words = word[1]['word']
                    overlap_perc = calculate_iou(box1,box2)
                    temp_dic = {}
                    if overlap_perc > 0.80:
                        if words != '-':
                            word_list.append(words)
                            bboxes_list.append(coords)
                            label_id = label2id[label]                              
                            ner_tags_list.append(label_id)
                        
                        custom_label_text['tokens'] = word_list
                        custom_label_text['bboxes'] = bboxes_list
                        custom_label_text['ner_tags'] = ner_tags_list

    final_list.append(custom_label_text)


train, test = train_test_split(final_list, random_state=21, test_size=0.3)

for detail  in final_list:
    with open('final_list.txt', 'a') as f:
        f.write(str(detail))
        f.write('\n')
        
for detail  in train:
    with open('train.txt', 'a') as f:
        f.write(str(detail))
        f.write('\n')
        
for detail  in test:
    with open('test.txt', 'a') as f:
        f.write(str(detail))
        f.write('\n')