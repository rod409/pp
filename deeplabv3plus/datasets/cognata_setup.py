import csv
import ast
import numpy as np
import os
import pathlib
from PIL import Image
from argparse import ArgumentParser

def get_color_map(cognata_path, scenario):
    color_map = {}
    segmentation_file = os.path.join(cognata_path, scenario, 'semantic_segmentation.csv')
    with open(segmentation_file) as csv_file:
        reader = csv.DictReader(csv_file)
        data = list(reader)
        for row in data:
            color = ast.literal_eval(row['semanticType.Color'])
            color = tuple(color)
            id = ast.literal_eval(row['semanticType.Number'])
            color_map[color] = id
    return color_map

def named_tuples(cognata_path, scenario):
    tuples = []
    segmentation_file = os.path.join(cognata_path, scenario, 'semantic_segmentation.csv')
    with open(segmentation_file) as csv_file:
        reader = csv.DictReader(csv_file)
        data = list(reader)
        for row in data:
            color = ast.literal_eval(row['semanticType.Color'])
            color = tuple(color)
            id = ast.literal_eval(row['semanticType.Number'])
            train_id = id
            label = "'" + row['semanticType.Label'] + "'"
            tuples.append('CognataClass({}, {}, {}, {})'.format(label, id, train_id, color))
    return tuples

def convert_to_label(cognata_path, scenario, camera, label_folder, file_name, color_map):
    image_file = os.path.join(cognata_path, scenario, camera, file_name)
    img = Image.open(image_file).convert('RGB')
    #img = Image.open('/home/rshojaei/large_assets/data/cityscapes/gtFine/train/aachen/aachen_000042_000019_gtFine_labelIds.png')
    #img = Image.open('/home/rshojaei/pptest/pp/0000000033_labelids.png')
    img = np.array(img)
    labels = np.empty(shape=(img.shape[0], img.shape[1]), dtype=np.uint8)
    for c in color_map:
        labels[(img == c).all(axis=2)] = color_map[c]

    label_image = Image.fromarray(labels)
    label_path = os.path.join(label_folder, file_name)
    label_image.save(label_path)

def convert_all(cognata_path, scenarios, cameras, color_map):
    for scene in scenarios:
        for cam in cameras:
            label_folder = os.path.join(cognata_path, scene, cam + '_label_id')
            img_folder = os.path.join(cognata_path, scene, cam)
            pathlib.Path(label_folder).mkdir(exist_ok=True)
            img_files = sorted([os.path.join(img_folder, f) for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f))])
            for img_f in img_files:
                convert_to_label(cognata_path, scene, cam, label_folder, os.path.basename(img_f), color_map)

def get_args():
    parser = ArgumentParser(description="scan objects in dataset")
    parser.add_argument("--cognata-path", default='/cognata', type=str)
    parser.add_argument("--named-tuples", action='store_true')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    opt = get_args()
    scenarios = ['10001_Urban_Clear_Noon']
    cameras = ['Cognata_Camera_01_8M_sl_png']
    if opt.named_tuples:
        tuple_list = named_tuples(opt.cognata_path, scenarios[0])
        print(*tuple_list, sep=',\n')
    else:
        cm = get_color_map(opt.cognata_path, scenarios[0])
        convert_all(opt.cognata_path, scenarios, cameras, cm)
