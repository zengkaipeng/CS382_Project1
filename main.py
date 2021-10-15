from Models import InterpolationModel


def File_Read(train_dir, val_dir, test_dir):
    with open(train_dir) as Fin:
        train_set = Fin.read()
    with open(val_dir) as Fin:
        val_set = Fin.read()
    with open(test_dir) as Fin:
        test_set = Fin.read()
    return train_set, val_set, test_set


if __name__ == '__main__':
    
    train_dir = '../data/train_set.txt'
    val_dir = '../data/dev_set.txt'
    test_dir = '../data/test_set.txt'
    train_set, val_set, test_set = File_Read(train_dir, val_dir, test_dir)
    
    model = InterpolationModel(degree=2, context=train_set)
    print('[INFO] model init done')
    model.train(val_set, verbose=True, eps=1e-3)
    print(model.get_PPL(val_set))
    '''
    train_dir = '../data/toy_train.txt'
    dev_dir = '../data/toy_dev.txt'
    train_set, dev_set, _ = File_Read(train_dir, dev_dir, dev_dir)
    model = InterpolationModel(degree=2, context=train_set)
    model.train(dev_set, verbose=True, eps=1e-3)
    print(model.get_PPL(dev_set))
    
    while True: 
        text = input().strip()
        print(model.get_p(text))
	'''