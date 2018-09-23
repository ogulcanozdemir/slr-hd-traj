import os


datapath = '/raid/users/oozdemir/data/BosphorusSign/ToyDataset'
classdir = sorted(list(map(int, os.listdir(datapath))))

train_split_file = open('/raid/users/oozdemir/code/tm-shd-slr/data/splits/toydata/train.txt', 'w')
test_split_file = open('/raid/users/oozdemir/code/tm-shd-slr/data/splits/toydata/test.txt', 'w')

for _clazz in classdir:
    _clazzpath = os.path.join(datapath, str(_clazz))

    userdir = sorted(os.listdir(_clazzpath))
    for _user in userdir:
        _user_split = _user.split('_')

        print(str(_clazz) + '-' + _user)
        if _user_split[0] != '053':
            train_split_file.write(str(_clazz) + '/' + _user + ' ' + str(_clazz) + '\n')
        else:
            test_split_file.write(str(_clazz) + '/' + _user + ' ' + str(_clazz) + '\n')


class_indices = [(i, item) for i,item in enumerate(classdir, 1)]
class_indices_file = open('/raid/users/oozdemir/code/tm-shd-slr/data/splits/toydata/class_indices.txt', 'w')
for idx, item in class_indices:
    class_indices_file.write(str(item) + ':' + str(idx) + '\n')

train_split_file.close()
test_split_file.close()
class_indices_file.close()