import pytest


def test_split_data_inflamatory():
    train_data = open('train_data/data_inflamatory.txt', 'r')
    train_data = train_data.read().strip().split('\n')

    test_data = open('test_data/data_inflamatory.txt', 'r')
    test_data = test_data.read().strip().split('\n')

    valid_data = open('validation_data/data_inflamatory.txt', 'r')
    valid_data = valid_data.read().strip().split('\n')


    #test train and test
    for file in train_data:
        assert file not in test_data, 'Error - training data is mixed with testing data - inflamatory'
        assert file not in valid_data, 'Error - training data is mixed with valid data - inflamatory'

    for file in valid_data:
        assert file not in train_data, 'Error - validation data is mixed with training data - inflamatory'
        assert file not in test_data, 'Error - validation data is mixed with test data - inflamatory'
            
    for file in test_data:
        assert file not in train_data, 'Error - test data is mixed with training data - inflamatory'
        assert file not in valid_data, 'Error - test data is mixed with valid data - inflamatory'


def test_split_data_normal():
    train_data = open('train_data/data_normal.txt', 'r')
    train_data = train_data.read().strip().split('\n')

    test_data = open('test_data/data_normal.txt', 'r')
    test_data = test_data.read().strip().split('\n')

    valid_data = open('validation_data/data_normal.txt', 'r')
    valid_data = valid_data.read().strip().split('\n')


    #test train and test
    for file in train_data:
        assert file not in test_data, 'Error - training data is mixed with testing data - normal'
        assert file not in valid_data, 'Error - training data is mixed with valid data - normal'

    for file in valid_data:
        assert file not in train_data, 'Error - validation data is mixed with training data - normal'
        assert file not in test_data, 'Error - validation data is mixed with test data - normal'
            
    for file in test_data:
        assert file not in train_data, 'Error - test data is mixed with training data - normal'
        assert file not in valid_data, 'Error - test data is mixed with valid data - normal'



def test_split_data_other():
    train_data = open('train_data/data_other.txt', 'r')
    train_data = train_data.read().strip().split('\n')

    test_data = open('test_data/data_other.txt', 'r')
    test_data = test_data.read().strip().split('\n')

    valid_data = open('validation_data/data_other.txt', 'r')
    valid_data = valid_data.read().strip().split('\n')


    #test train and test
    for file in train_data:
        assert file not in test_data, 'Error - training data is mixed with testing data - other'
        assert file not in valid_data, 'Error - training data is mixed with valid data - other'

    for file in valid_data:
        assert file not in train_data, 'Error - validation data is mixed with training data - other'
        assert file not in test_data, 'Error - validation data is mixed with test data - other'
            
    for file in test_data:
        assert file not in train_data, 'Error - test data is mixed with training data - other'
        assert file not in valid_data, 'Error - test data is mixed with valid data - other'



def test_split_data_tumor():
    train_data = open('train_data/data_tumor.txt', 'r')
    train_data = train_data.read().strip().split('\n')

    test_data = open('test_data/data_tumor.txt', 'r')
    test_data = test_data.read().strip().split('\n')

    valid_data = open('validation_data/data_tumor.txt', 'r')
    valid_data = valid_data.read().strip().split('\n')


    #test train and test
    for file in train_data:
        assert file not in test_data, 'Error - training data is mixed with testing data - tumor'
        assert file not in valid_data, 'Error - training data is mixed with valid data - tumor'

    for file in valid_data:
        assert file not in train_data, 'Error - validation data is mixed with training data - tumor'
        assert file not in test_data, 'Error - validation data is mixed with test data - tumor'
            
    for file in test_data:
        assert file not in train_data, 'Error - test data is mixed with training data - tumor'
        assert file not in valid_data, 'Error - test data is mixed with valid data - tumor'


