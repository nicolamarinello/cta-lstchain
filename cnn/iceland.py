from classifier_training import classifier_training_main
from keras import backend as K
from regressor_training import regressor_training_main

if __name__ == "__main__":

    which = 'paper-densenet'

    if which == 'resnets':

        traind_class = ['/ssdraptor/simulations/Paranal_gamma-diffuse_North_20deg_3HB9_DL1_ML1_interp',
                        '/ssdraptor/simulations/Paranal_proton_North_20deg_3HB9_DL1_ML1_interp']
        testd_class = ['/ssdraptor/simulations/Paranal_gamma_North_20deg_3HB9_DL1_ML1_interp',
                       '/ssdraptor/simulations/Paranal_proton_North_20deg_3HB9_DL1_ML1_interp_test']

        traind_regr = ['/ssdraptor/simulations/Paranal_gamma-diffuse_North_20deg_3HB9_DL1_ML1_interp']

        testd_regr = ['/ssdraptor/simulations/Paranal_gamma_North_20deg_3HB9_DL1_ML1_interp']

        # train and test classic ResNets architectures
        model_names = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']
        time = True
        epochs = 100
        batch_size = 256
        opt = 'sgd'
        validation = True
        reduction = 1
        lrop = True
        sd = False
        clr = False
        es = False
        workers = 4

        classifier_training_main(folders=traind_class,
                                 model_name=model_names[0],
                                 time=False,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 opt=opt,
                                 val=validation,
                                 red=reduction,
                                 lropf=lrop,
                                 sd=sd,
                                 clr=clr,
                                 es=es,
                                 workers=workers,
                                 test_dirs=testd_class
                                 )

        K.clear_session()

        for model in model_names:

            print(model)

            classifier_training_main(folders=traind_class,
                                     model_name=model,
                                     time=time,
                                     epochs=epochs,
                                     batch_size=batch_size,
                                     opt=opt,
                                     val=validation,
                                     red=reduction,
                                     lropf=lrop,
                                     sd=sd,
                                     clr=clr,
                                     es=es,
                                     workers=workers,
                                     test_dirs=testd_class
                                     )

            K.clear_session()

    elif which == 'paper-densenet':

        traind_class = ['/home/nmarinel/simulations/Paranal_gamma-diffuse_North_20deg_3HB9_DL1_ML1_interp',
                        '/home/nmarinel/simulations/Paranal_proton_North_20deg_3HB9_DL1_ML1_interp']
        testd_class = ['/mnt/simulations/Paranal_gamma_North_20deg_3HB9_DL1_ML1_interp',
                       '/mnt/simulations/Paranal_proton_North_20deg_3HB9_DL1_ML1_interp_test']

        traind_regr = ['/home/nmarinel/simulations/Paranal_gamma-diffuse_North_20deg_3HB9_DL1_ML1_interp']

        testd_regr = ['/mnt/simulations/Paranal_gamma_North_20deg_3HB9_DL1_ML1_interp']

        # densenet paper classifier
        model = 'DenseNet'
        time = True
        epochs = 50
        batch_size = 64
        opt = 'adam'
        validation = True
        reduction = 1
        lrop = False
        sd = False
        clr = False
        es = False
        workers = 4

        classifier_training_main(folders=traind_class,
                                 model_name=model,
                                 time=time,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 opt=opt,
                                 val=validation,
                                 red=reduction,
                                 lropf=lrop,
                                 sd=sd,
                                 clr=clr,
                                 es=es,
                                 workers=workers,
                                 test_dirs=testd_class
                                 )
        K.clear_session()

        # densenet paper energy
        feature = 'energy'

        regressor_training_main(folders=traind_regr,
                                model_name=model,
                                time=time,
                                epochs=epochs,
                                batch_size=batch_size,
                                opt=opt,
                                val=validation,
                                red=reduction,
                                lropf=lrop,
                                sd=sd,
                                clr=clr,
                                es=es,
                                feature=feature,
                                workers=workers,
                                test_dirs=testd_regr)

        K.clear_session()

        # densenet paper direction
        feature = 'xy'

        regressor_training_main(folders=traind_regr,
                                model_name=model,
                                time=time,
                                epochs=epochs,
                                batch_size=batch_size,
                                opt=opt,
                                val=validation,
                                red=reduction,
                                lropf=lrop,
                                sd=sd,
                                clr=clr,
                                es=es,
                                feature=feature,
                                workers=workers,
                                test_dirs=testd_regr)

        K.clear_session()

    elif which == 'paper-resnetf':

        traind_class = ['/ssdraptor/simulations/Paranal_gamma-diffuse_North_20deg_3HB9_DL1_ML1_interp',
                        '/ssdraptor/simulations/Paranal_proton_North_20deg_3HB9_DL1_ML1_interp']
        testd_class = ['/ssdraptor/simulations/Paranal_gamma_North_20deg_3HB9_DL1_ML1_interp',
                       '/ssdraptor/simulations/Paranal_proton_North_20deg_3HB9_DL1_ML1_interp_test']

        traind_regr = ['/ssdraptor/simulations/Paranal_gamma-diffuse_North_20deg_3HB9_DL1_ML1_interp']

        testd_regr = ['/ssdraptor/simulations/Paranal_gamma_North_20deg_3HB9_DL1_ML1_interp']

        # densenet paper classifier
        model = 'ResNetF'
        epochs = 50
        batch_size = 64
        opt = 'adam'
        validation = True
        reduction = 1
        lrop = False
        sd = False
        clr = False
        es = False
        workers = 4

        """

        classifier_training_main(folders=traind_class,
                                 model_name=model,
                                 time=False,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 opt=opt,
                                 val=validation,
                                 red=reduction,
                                 lropf=lrop,
                                 sd=sd,
                                 clr=clr,
                                 es=es,
                                 workers=workers,
                                 test_dirs=testd_class
                                 )
        K.clear_session()

        classifier_training_main(folders=traind_class,
                                 model_name=model,
                                 time=True,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 opt=opt,
                                 val=validation,
                                 red=reduction,
                                 lropf=lrop,
                                 sd=sd,
                                 clr=clr,
                                 es=es,
                                 workers=workers,
                                 test_dirs=testd_class
                                 )
        K.clear_session()
        
        """

        # densenet paper energy
        feature = 'energy'

        regressor_training_main(folders=traind_regr,
                                model_name=model,
                                time=False,
                                epochs=epochs,
                                batch_size=batch_size,
                                opt=opt,
                                val=validation,
                                red=reduction,
                                lropf=lrop,
                                sd=sd,
                                clr=clr,
                                es=es,
                                feature=feature,
                                workers=workers,
                                test_dirs=testd_regr)

        K.clear_session()

        regressor_training_main(folders=traind_regr,
                                model_name=model,
                                time=True,
                                epochs=epochs,
                                batch_size=batch_size,
                                opt=opt,
                                val=validation,
                                red=reduction,
                                lropf=lrop,
                                sd=sd,
                                clr=clr,
                                es=es,
                                feature=feature,
                                workers=workers,
                                test_dirs=testd_regr)

        K.clear_session()

        # densenet paper direction
        feature = 'xy'

        regressor_training_main(folders=traind_regr,
                                model_name=model,
                                time=False,
                                epochs=epochs,
                                batch_size=batch_size,
                                opt=opt,
                                val=validation,
                                red=reduction,
                                lropf=lrop,
                                sd=sd,
                                clr=clr,
                                es=es,
                                feature=feature,
                                workers=workers,
                                test_dirs=testd_regr)

        K.clear_session()

        regressor_training_main(folders=traind_regr,
                                model_name=model,
                                time=True,
                                epochs=epochs,
                                batch_size=batch_size,
                                opt=opt,
                                val=validation,
                                red=reduction,
                                lropf=lrop,
                                sd=sd,
                                clr=clr,
                                es=es,
                                feature=feature,
                                workers=workers,
                                test_dirs=testd_regr)

        K.clear_session()

    # learning rate comparison on densenet
    elif which == 'lrcmpdense':

        traind_class = ['/home/nmarinel/simulations/Paranal_gamma-diffuse_North_20deg_3HB9_DL1_ML1_interp',
                        '/home/nmarinel/simulations/Paranal_proton_North_20deg_3HB9_DL1_ML1_interp']
        testd_class = ['/mnt/simulations/Paranal_gamma_North_20deg_3HB9_DL1_ML1_interp',
                       '/mnt/simulations/Paranal_proton_North_20deg_3HB9_DL1_ML1_interp_test']

        # only train to evaluate validation lr
        lrs = [1e-4, 1e-3, 1e-2]
        model_names = 'DenseNet'
        time = True
        epochs = 30
        batch_size = 64
        opt = 'sgd'
        validation = True
        reduction = 1
        lrop = True
        sd = False
        clr = False
        es = False
        workers = 4