import datetime
import os

import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard)
from keras.layers import Conv2D, Dense, DepthwiseConv2D
from tensorflow.keras.optimizers import SGD, Adam
from keras.regularizers import l2

from nets.MSSDet import MSSDet_model
from nets.MSSDet_training import MultiboxLoss, get_lr_scheduler
from utils.anchors import get_anchors
from utils.callbacks import ExponentDecayScheduler, LossHistory
from utils.dataloader import MSSDetDatasets
from utils.utils import get_classes

if __name__ == "__main__":

    tf.compat.v1.disable_eager_execution()
    classes_path = 'model_data/voc_classes.txt'

    model_path = 'model_data/MSSDet_weights.h5'

    input_shape = [224, 224]

    anchors_size = [22.4, 44.8, 82.88, 120.96, 159.04, 197.12,235.2]

    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 16

    UnFreeze_Epoch = 2000 # epoch
    Unfreeze_batch_size = 8

    Freeze_Train = False


    Init_lr = 2e-3
    Min_lr = Init_lr * 0.01

    optimizer_type = "sgd"
    momentum = 0.937
    weight_decay = 1e-4

    lr_decay_type = 'cos'

    save_period = 1

    save_dir = 'logs'

    num_workers = 1


    train_annotation_path   = 'train/SA-HP_train.txt'
    val_annotation_path     = 'train/SA-HP_val.txt'


    class_names, num_classes = get_classes(classes_path)
    num_classes += 1
    anchors = get_anchors(input_shape, anchors_size)

    K.clear_session()
    model = MSSDet_model((input_shape[0], input_shape[1], 3), num_classes)
    # if model_path != '':
    #     print('Load weights {}.'.format(model_path))
    #     model.load_weights(model_path, by_name=True, skip_mismatch=True)


    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    for layer in model.layers:
        if isinstance(layer, DepthwiseConv2D):
            layer.add_loss(l2(weight_decay)(layer.depthwise_kernel))
        elif isinstance(layer, Conv2D) or isinstance(layer, Dense):
            layer.add_loss(l2(weight_decay)(layer.kernel))


    if True:
        if Freeze_Train:
            freeze_layers = 17
            for i in range(freeze_layers):
                model.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(
                freeze_layers, len(model.layers)))


        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        start_epoch = Init_Epoch
        end_epoch = Freeze_Epoch if Freeze_Train else UnFreeze_Epoch


        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr,
                              lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr,
                             lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam': Adam(lr=Init_lr_fit, beta_1=momentum),
            'sgd': SGD(lr=Init_lr_fit, momentum=momentum, nesterov=True)
        }[optimizer_type]
        model.compile(optimizer=optimizer, loss=MultiboxLoss(
            num_classes, neg_pos_ratio=3.0).compute_loss)


        lr_scheduler_func = get_lr_scheduler(
            lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

        train_dataloader = MSSDetDatasets(
            train_lines, input_shape, anchors, batch_size, num_classes, train=True)
        val_dataloader = MSSDetDatasets(
            val_lines, input_shape, anchors, batch_size, num_classes, train=False)

        time_str = datetime.datetime.strftime(
            datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        logging = TensorBoard(log_dir)
        loss_history = LossHistory(log_dir)
        checkpoint = ModelCheckpoint(os.path.join(save_dir, "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"),
                                     monitor='val_loss', save_weights_only=True, save_best_only=False, period=save_period)
        early_stopping = EarlyStopping(
            monitor='val_loss', min_delta=0, patience=10, verbose=1)
        lr_scheduler = LearningRateScheduler(lr_scheduler_func, verbose=1)
        callbacks = [logging, loss_history, checkpoint, lr_scheduler]

        if start_epoch < end_epoch:
            print('Train on {} samples, val on {} samples, with batch size {}.'.format(
                num_train, num_val, batch_size))
            model.fit_generator(
                generator=train_dataloader,
                steps_per_epoch=epoch_step,
                validation_data=val_dataloader,
                validation_steps=epoch_step_val,
                epochs=end_epoch,
                initial_epoch=start_epoch,
                use_multiprocessing=True if num_workers > 1 else False,
                workers=num_workers,
                callbacks=callbacks
            )
        #---------------------------------------#
        #   冻结学习
        #---------------------------------------#
        if Freeze_Train:
            batch_size = Unfreeze_batch_size
            start_epoch = Freeze_Epoch if start_epoch < Freeze_Epoch else start_epoch
            end_epoch = UnFreeze_Epoch

            nbs = 64
            lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
            lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
            Init_lr_fit = min(max(batch_size / nbs * Init_lr,
                                  lr_limit_min), lr_limit_max)
            Min_lr_fit = min(max(batch_size / nbs * Min_lr,
                                 lr_limit_min * 1e-2), lr_limit_max * 1e-2)

            lr_scheduler_func = get_lr_scheduler(
                lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
            lr_scheduler = LearningRateScheduler(lr_scheduler_func, verbose=1)
            callbacks = [logging, loss_history, checkpoint, lr_scheduler]

            for i in range(len(model.layers)):
                model.layers[i].trainable = True
            model.compile(optimizer=optimizer, loss=MultiboxLoss(
                num_classes, neg_pos_ratio=3.0).compute_loss)

            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size

            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

            train_dataloader.batch_size = Unfreeze_batch_size
            val_dataloader.batch_size = Unfreeze_batch_size

            print('Train on {} samples, val on {} samples, with batch size {}.'.format(
                num_train, num_val, batch_size))
            model.fit_generator(
                generator=train_dataloader,
                steps_per_epoch=epoch_step,
                validation_data=val_dataloader,
                validation_steps=epoch_step_val,
                epochs=end_epoch,
                initial_epoch=start_epoch,
                use_multiprocessing=True if num_workers > 1 else False,
                workers=num_workers,
                callbacks=callbacks
            )
