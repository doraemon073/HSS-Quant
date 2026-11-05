#=========================================================
#
#  
#
#=========================================================
import os
import argparse
import random
import re
from os.path import join

def _indexed_files(root_path, relative_path):
    data_path = join(root_path, relative_path)
    if not os.path.isdir(data_path):
        raise FileNotFoundError('DRIVE data directory not found: {}'.format(data_path))
    indexed = {}
    for filename in os.listdir(data_path):
        match = re.match(r'^(\d+)_', filename)
        if match is None:
            continue
        image_id = int(match.group(1))
        if image_id in indexed:
            raise ValueError('Duplicate DRIVE image ID {} in {}'.format(image_id, data_path))
        indexed[image_id] = join(data_path, filename)
    return indexed

def get_path_list(root_path,img_path,label_path,fov_path):
    indexed = [_indexed_files(root_path, path) for path in (img_path, label_path, fov_path)]
    expected_ids = set(indexed[0])
    if any(set(files) != expected_ids for files in indexed[1:]):
        raise ValueError('DRIVE image, label, and FOV IDs do not match for {}'.format(root_path))
    return [[files[image_id] for image_id in sorted(expected_ids)] for files in indexed]

def write_path_list(name_list, save_path, file_name):
    with open(join(save_path, file_name), 'w') as path_file:
        for i in range(len(name_list[0])):
            path_file.write('{} {} {}\n'.format(name_list[0][i], name_list[1][i], name_list[2][i]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default=os.environ.get('DRIVE_DATA_ROOT', '/home/xpp/data/DRIVE'))
    parser.add_argument('--split-seed', default=2026, type=int)
    parser.add_argument('--validation-images', default=4, type=int)
    args = parser.parse_args()

    if not os.path.isdir(args.data_root):
        raise FileNotFoundError('DRIVE data root not found: {}'.format(args.data_root))

    save_path = './prepare_dataset/data_path_list/DRIVE/'
    os.makedirs(save_path, exist_ok=True)

    train_records = get_path_list(args.data_root, 'training/images', 'training/1st_manual', 'training/mask')
    print('Number of train imgs:',len(train_records[0]))
    write_path_list(train_records, save_path, 'train.txt')

    test_paths = ('test/images', 'test/1st_manual', 'test/mask')
    if all(os.path.isdir(join(args.data_root, path)) for path in test_paths):
        test_records = get_path_list(args.data_root, *test_paths)
        print('Number of test imgs:',len(test_records[0]))
        write_path_list(test_records, save_path, 'test.txt')
    else:
        print('Test manifest not generated: DRIVE test ground-truth labels are missing.')

    image_ids = [int(re.match(r'^(\d+)_', os.path.basename(path)).group(1)) for path in train_records[0]]
    if args.validation_images <= 0 or args.validation_images >= len(image_ids):
        raise ValueError('--validation-images must be between 1 and the number of training images - 1')
    validation_ids = set(random.Random(args.split_seed).sample(image_ids, args.validation_images))
    train_ids = set(image_ids) - validation_ids
    lists_path = './lists/DRIVE'
    os.makedirs(lists_path, exist_ok=True)
    train_split = [[path for path, image_id in zip(paths, image_ids) if image_id in train_ids]
                   for paths in train_records]
    val_split = [[path for path, image_id in zip(paths, image_ids) if image_id in validation_ids]
                 for paths in train_records]
    write_path_list(train_split, lists_path, 'train_images_seed{}.txt'.format(args.split_seed))
    write_path_list(val_split, lists_path, 'val_images_seed{}.txt'.format(args.split_seed))
    print('Train image IDs:', sorted(train_ids))
    print('Validation image IDs:', sorted(validation_ids))

    print("Finish!")
    
