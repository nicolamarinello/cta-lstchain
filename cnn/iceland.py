from classifier_training import classifier_training_main
from regressor_training import regressor_training_main

traind_class = ['/home/nmarinel/simulations/Paranal_gamma-diffuse_North_20deg_3HB9_DL1_ML1_interp',
                '/home/nmarinel/simulations/Paranal_proton_North_20deg_3HB9_DL1_ML1_interp']
testd_class = ['/mnt/simulations/Paranal_gamma_North_20deg_3HB9_DL1_ML1_interp',
               '/mnt/simulations/Paranal_proton_North_20deg_3HB9_DL1_ML1_interp_test']

traind_regr = ['/home/nmarinel/simulations/Paranal_gamma-diffuse_North_20deg_3HB9_DL1_ML1_interp']

testd_regr = ['/mnt/simulations/Paranal_gamma_North_20deg_3HB9_DL1_ML1_interp']


regressor_training_main(folders=traind_regr, model_name='DenseNet', time=True, epochs=150, batch_size=64, opt='adam',
                        val=True, red=1, lropf=False, sd=False, clr=False, es=True, feature='energy',
                        workers=4, test_dirs=testd_regr)

classifier_training_main(folders=traind_class, model_name='ResNetF', time=True, epochs=42, batch_size=64, opt='adam',
                         val=True, red=1, lropf=False, sd=False, clr=False, es=False, workers=4, test_dirs=testd_class)
