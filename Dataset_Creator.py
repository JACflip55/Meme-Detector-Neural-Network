from tflearn.data_utils import build_hdf5_image_dataset
from tflearn.data_utils import image_preloader
from tflearn.data_utils import shuffle
import h5py

folder = 'Data'

def load_image_dataset():
    print('Loading Images...')

    X,Y = image_preloader(folder, image_shape=(128,128),mode = 'folder',categorical_labels= True,filter_channel=True)

    return X,Y

X,Y = load_image_dataset()
print('Shuffling data...')
X,Y = shuffle(X,Y)

h5f = h5py.File('dataset.h5', 'w')

print('Creating Dataset...')
h5f.create_dataset('X', data=X)
h5f.create_dataset('Y', data=Y)

h5f.close()
print('Dataset Created')