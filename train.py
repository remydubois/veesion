from utils import *
from models import *
import argparse
import pandas
from keras.optimizers import Adam, SGD
from sklearn.utils import class_weight
from callbacks import *
from keras.models import Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
import sys


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if '/Users/remydubois/anaconda3/lib/python3.6' in sys.path:
    TARGET = '/Users/remydubois/Dropbox/Remy/results/'
    LOCAL = '/Users/remydubois/Downloads/'
else:
    TARGET = '/cbio/donnees/rdubois/results/'
    LOCAL = '/mnt/data40T_v2/rdubois/data/veesion/'

parser = argparse.ArgumentParser(
    description='Train model for video gesture classification')
parser.add_argument('--epochs',
                    type=int,
                    default=100,
                    help='Number of epochs for which to train.'
                    )
parser.add_argument('--gpu',
                    default='0',
                    help='Which gpu to use.'
                    )
parser.add_argument('--batchsize',
                    type=int,
                    default=32,
                    help='batch size.'
                    )
parser.add_argument('--length',
                    type=int,
                    default=40,
                    help='video length.'
                    )
parser.add_argument('--bypass',
                    default='simple',
                    help='Bypass to use in the squuezenet, ie simple or nothing.'
                    )
parser.add_argument('--width',
                    type=int,
                    default=150,
                    help='Width of the frames'
                    )
parser.add_argument('--model',
                    default='squeezenet',
                    help='model to use')


def main(args_):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # read in
    y_train, y_test, y_valid = read_labels(LOCAL)
    train_paths = path_parser(
        LOCAL + '20bn-jester-v1', y_train[0])
    train_generator = data_generator(train_paths, y_train[2])

    test_paths = path_parser(
        LOCAL + '20bn-jester-v1', y_test[0])
    test_generator = data_generator(test_paths, y_test[2])

    # Define model
    input_ = Input(shape=(100, 150, 40, 3))
    if args.model == 'squeezenet':
        output_ = SqueezeNetOutput3D(
            input_, num_classes=len(y_train[2].unique()), bypass=args.bypass)
    elif args.model == 'naive':
        output_ = NaiveModelOutput(input_, len(y_train[2].unique()))
    elif args.model == 'vgg16':
        output_ = VGG16Output(input_, len(y_train[2].unique()))
    else:
        raise ValueError('Unkown model')

    model = Model(input_, output_)

    # Define callbacks
    tb = MyTensorBoard(TARGET + 'veesion/' + args.model,
                       write_batch_performance=True)
    ck = ModelCheckpoint(filepath=TARGET + 'veesion/' + args.model + '/model-ckpt', verbose=0, save_best_only=True)

    # Handle unbalance data
    class_weights = class_weight.compute_class_weight('balanced',
                                                      y_train[2].unique(),
                                                      y_train[2])

    # Compile
    model.compile(optimizer=Adam(1.e-4),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    # Train
    history = model.fit_generator(
        train_generator,
        class_weight=class_weights,
        steps_per_epoch=y_train.shape[0] // args.batchsize,
        validation_data=test_generator,
        validation_steps=y_test.shape[0] // args.batchsize,
        epochs=args.epochs,
        workers=1,
        callbacks=[tb, ck],
        max_queue_size=5
    )

    with open(TARGET + 'veesion/' + args.model + '/hist', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

if __name__ == '__main__':
    args = parser.parse_args()

    main(args)
