import os


def load_data(path_to_data):
    data = []
    with open(path_to_data) as raw_data:
        for line in raw_data:
            data.append((line.strip().split(',')))
    return data


def main():
    # load data
    DATA_DIR = '/baobab/kiyomaru/2018-shinjin/jumanpp.midasi'
    PATH_TO_TRAIN = os.path.join(DATA_DIR, 'train.csv')
    PATH_TO_TEST = os.path.join(DATA_DIR, 'test.csv')
    train = load_data(PATH_TO_TRAIN)
    test = load_data(PATH_TO_TEST)

    # convert document to ids

    # define a model

    # get results

    return 0


if __name__ == '__main__':
    main()
