from Models import InterpolationModel
from Models import AddkModel

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
    
    model = AddkModel(degree=3, context=train_set + val_set)
    print('[INFO] model init done')
    model.get_PPL(test_set)

    model = InterpolationModel(degree=3, context=train_set)
    print('[INFO] model init done')
    model.train(val_set, verbose=True, eps=1e-3)
    print(model.get_PPL(test_set))

    