import argparse

from keras.utils.data_utils import OrderedEnqueuer
from keras.utils.generic_utils import Progbar

from generators import DataGeneratorRF
from utils import get_all_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dirs', type=str, default='', nargs='+', help='Folders that contains test data.', required=True)
    parser.add_argument(
        '--model_sep', type=str, default='', help='Path of the model to load.', required=True)
    parser.add_argument(
        '--model_energy', type=str, default='', help='Path of the model to load.', required=True)
    parser.add_argument(
        '--model_azalt', type=str, default='', help='Path of the model to load.', required=True)

    FLAGS, unparsed = parser.parse_known_args()
    dirs = FLAGS.dirs

    h5files = get_all_files(dirs)
    batch_size = 64

    print('Building test generator...')
    # TODO: check that the three generators keep the same event ordering
    # test_generator_sep = DataGeneratorC(h5files, batch_size=batch_size, arrival_time=True, shuffle=False)
    # test_generator_eng = DataGeneratorR(h5files, batch_size=batch_size, arrival_time=True, shuffle=False,
    #                                    feature='energy')
    # test_generator_daa = DataGeneratorR(h5files, batch_size=batch_size, arrival_time=True, shuffle=False, feature='xy')

    test_generator = DataGeneratorRF()

    # retrieve ground truth
    print('Retrieving ground truth...')
    steps_done = 0
    steps = len(test_generator)
    # steps = 10

    enqueuer = OrderedEnqueuer(test_generator, use_multiprocessing=True)
    enqueuer.start(workers=4, max_queue_size=10)
    output_generator = enqueuer.get()

    progbar = Progbar(target=steps)

    while steps_done < steps:
        generator_output = next(output_generator)
        x, energy, altaz, y = generator_output

        steps_done += 1
        progbar.update(steps_done)
