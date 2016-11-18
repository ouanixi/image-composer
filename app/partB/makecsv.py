import cv2
from app.utils import feature_extractor
import csv

DATAPATH = "/home/ouanixi/Work/image-composer/dataset/raw"


def load_files(fname):
    image_path = []
    with open(fname) as f:
        content = f.readlines()
        for line in content:
            image_path.append(DATAPATH + line.split('.',1)[1].rstrip('\n'))
    return image_path


def create_training_set(man_path, nat_path):
    # get manmade images
    man_image_names = load_files(DATAPATH + '/' + man_path)
    nat_image_names = load_files(DATAPATH + '/' + nat_path)
    feature_list = []
    for name in man_image_names:
        img = cv2.imread(name)
        edges = feature_extractor.getHog(img)
        img_dict = dict()
        for i in xrange(len(edges)):
            pix = 'pix_' + str(i)
            img_dict[pix] = edges[i]
        img_dict["class"] = 0
        feature_list.append(img_dict)

    for name in nat_image_names:
        img = cv2.imread(name)
        edges = feature_extractor.getHog(img)
        img_dict = dict()
        for i in xrange(len(edges)):
            pix = 'pix_' + str(i)
            img_dict[pix] = edges[i]
        img_dict["class"] = 1
        feature_list.append(img_dict)

    return feature_list


def get_dataset(man_path, nat_path):
    feature_list = create_training_set(man_path, nat_path)
    X = []
    Y = []
    for element in feature_list:
        row = []
        for key, value in element.iteritems():
            if key != "class":
                row.append(value)
        X.append(row)
        Y.append(element["class"])
    return X, Y


def make_training_csv(man, nat, name):
    toCSV = create_training_set(man, nat)
    keys = toCSV[0].keys()
    with open(name, 'wb') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(toCSV)


if __name__ == '__main__':
    make_training_csv("manmade_test.txt", "natural_test.txt", "hog_only_test.csv")
    make_training_csv("manmade_training.txt", "natural_training.txt", "hog_only_train.csv")
