from entities import DataSet
import train2

TRAIN_SIZE = 100000
VALID_SIZE = 10000
TEST_SIZE = 10000


def main():
    train_set = DataSet('train', TRAIN_SIZE)
    # scaler = train_set.scaler

    validation_set = DataSet('valid', VALID_SIZE, scaler=None)
    test_set = DataSet('test', TEST_SIZE, include_test=False, scaler=None)

    train2.train(train_set, validation_set, test_set)

if __name__ == '__main__':
    main()
