import io
import os
import re
import copy
import yaml
import shutil

from sklearn.model_selection import KFold


def img2weight_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}weights{os.sep}"  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


def read_data_folder(path):
    files = []

    image_path = os.path.join(path, "images")
    label_path = os.path.join(path, "labels")
    for img_fn in os.listdir(image_path):
        lbl_fn = re.sub(".jpg$", ".txt", img_fn)

        current_image_path = os.path.join(image_path, img_fn)
        current_label_path = os.path.join(label_path, lbl_fn)
        files.append([current_image_path, current_label_path])
    return files


def write_data_folder(path, name, files):
    image_path = os.path.join(path, name, "images")
    label_path = os.path.join(path, name, "labels")
    os.makedirs(image_path)
    os.makedirs(label_path)
    for img_pth, lbl_pth in files:
        shutil.copy2(img_pth, image_path)
        shutil.copy2(lbl_pth, label_path)


def read_yaml_file(path):
    with open(path, "r") as f:
        yaml_data = yaml.safe_load(f)
    return yaml_data


def write_sub_directory_yaml_file(path, data, idx):
    data["train"] = re.sub("/train", f"-{idx}/train", data["train"])
    data["val"] = re.sub("/valid", f"-{idx}/valid", data["val"])
    with io.open(path, 'w', encoding='utf8') as outfile:
        yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)


def write_next_directory_yaml_file(path):
    dataset_name_regex = "([A-Za-z\s-]+)-(\d+)-?(\d+)?"
    data = read_yaml_file(path)

    matched_groups = re.search(dataset_name_regex, data["train"])
    dataset_name, version, sub_version = matched_groups.group(1), matched_groups.group(2), matched_groups.group(3)

    new_name = f"{dataset_name}-{int(version)+1}"
    new_path = re.sub(dataset_name_regex, new_name, path)
    data["train"] = f"{new_name}/train/images"
    data["val"] = f"{new_name}/valid/images"
    data["test"] = f"{new_name}/test/images"

    with io.open(new_path, 'w', encoding='utf8') as outfile:
        yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)



def generate_cv_batch(source_path, dest_path, source_folder):
    train_files = read_data_folder(os.path.join(source_path, source_folder, "train"))
    valid_files = read_data_folder(os.path.join(source_path, source_folder, "valid"))
    test_files = read_data_folder(os.path.join(source_path, source_folder, "test"))
    yaml_file = read_yaml_file(os.path.join(source_path, source_folder, "data.yaml"))

    all_files = train_files #+ valid_files

    kf = KFold(n_splits=5, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(all_files)):
        new_path = os.path.join(dest_path, f"{source_folder}-{i}")
        new_train_files = [all_files[idx] for idx in train_index]
        new_valid_files = [all_files[idx] for idx in test_index]
        write_data_folder(new_path, "train", new_train_files)
        write_data_folder(new_path, "valid", valid_files)
        write_data_folder(new_path, "test", new_valid_files)
        write_sub_directory_yaml_file(os.path.join(new_path, "data.yaml"), copy.deepcopy(yaml_file), i)

