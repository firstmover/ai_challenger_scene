""" scene classification resnet
    reference code see https://gist.github.com/omoindrot

    use random crop and random scale, 10crop eval.
    momentum-sgd
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import time

import tensorflow as tf
import numpy as np
import math

import utility_tensorflow as utils
import tqdm

import scene_config as c

root = c.root

data_dir = c.data_dir
tmp_dir = c.tmp_dir
log_dir = c.log_dir
checkpoint_dir = c.checkpoint_dir
model_dir = c.model_dir
acc_loss_path = c.acc_loss_path

batch_size = c.batch_size

num_epochs = c.num_epochs

# Constants describing the training process.
MOVING_AVERAGE_DECAY = c.MOVING_AVERAGE_DECAY  # The decay to use for the moving average.

# learning rate decay constant
NUM_EPOCHS_PER_DECAY = c.NUM_EPOCHS_PER_DECAY  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = c.LEARNING_RATE_DECAY_FACTOR  # Learning rate decay factor.

# momentum SGD optimizer
initial_learning_rate_mSGD = c.initial_learning_rate_mSGD
momentum = c.momentum

num_examples_per_epoch_for_train = c.num_examples_per_epoch_for_train
num_examples_per_epoch_for_val = c.num_examples_per_epoch_for_val
steps_per_epoch = int(math.ceil(num_examples_per_epoch_for_train / float(batch_size)))

# write tensorboard info every freq steps
tensorboard_write_frq = c.tensorboard_write_frq

# checkpoint every freq epoch
checkpoint_freq = c.checkpoint_freq

use_minimal_summary = c.use_minimal_summary

# bn updata op name, must be grouped with training op
resnet_layer = c.resnet_layer

keep_prob = c.dropout_keep_prob

import scene_resnet as resnet
UPDATE_OPS_COLLECTION = resnet.UPDATE_OPS_COLLECTION

import scene_input


def check_train(sess, correct_prediction, is_training, dataset_init_op, 
    n_batch, loss, dropout_keep_prob):
    """
    sampled accuracy and loss for training set.
    """
    # Initialize the correct dataset
    sess.run(dataset_init_op)
    num_correct, num_samples, sum_loss = 0, 0, 0
    for _ in tqdm.tqdm(range(n_batch)):
        try:
            feed_dict = {is_training: False, dropout_keep_prob: 1.0}
            corr_pred, l = sess.run([correct_prediction, loss], feed_dict)
            num_correct += corr_pred.sum()
            num_samples += corr_pred.shape[0]
            sum_loss += l
        except tf.errors.OutOfRangeError:
            break

    # return accuracy and loss.
    acc = float(num_correct) / num_samples
    sum_loss = float(sum_loss) / num_samples
    return acc, sum_loss


def check_val(sess, correct_prediction, is_training, dataset_init_op, 
    correct_prediction_top3, dropout_keep_prob):
    """
    Check the accuracy(top1 and top3) of the model on val.
    """
    # Initialize the correct dataset
    sess.run(dataset_init_op)
    num_correct, num_correct_top3, num_samples = 0, 0, 0
    num_steps_per_epoch = int(math.ceil(num_examples_per_epoch_for_val / float(batch_size)))
    for i in tqdm.tqdm(range(num_steps_per_epoch)):
        try:
            feed_dict = {is_training: False, dropout_keep_prob:1.0}
            correct_pred, correct_pred_top3 = sess.run(
                [correct_prediction, correct_prediction_top3], feed_dict)
            num_correct += correct_pred.sum()
            num_correct_top3 += correct_pred_top3.sum()
            num_samples += correct_pred.shape[0]
        except tf.errors.OutOfRangeError:
            break

    # Return the fraction of datapoints that were correctly classified
    acc = float(num_correct) / num_samples
    acc_top3 = float(num_correct_top3) / num_samples
    return acc, acc_top3


def check_time(curr_time, n_iter, interval):
    """ print time info """
    def _format_interval(t):
        mins, s = divmod(int(t), 60)
        h, m = divmod(mins, 60)
        if h:
            return '%d:%02d:%02d' % (h, m, s)
        else:
            return '%02d:%02d' % (m, s)

    rate = '%5.2f' % (n_iter / interval) if interval else '?'
    return "time: {} interval: {} {} iter/sec".format(_format_interval(curr_time),
                                                      _format_interval(interval), rate)


def train_resnet(total_loss, global_step):
    """ momentum sgd with lr exponentially decay """

    # Variables that affect learning rate.
    num_batches_per_epoch = num_examples_per_epoch_for_train / batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate_mSGD,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = utils.add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.MomentumOptimizer(lr, momentum, use_nesterov=True)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    if not use_minimal_summary:
        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    if c.use_moving_average:
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
    else:
        variables_averages_op = tf.no_op()

    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    apply_gradient_op_with_bn_update = tf.group(apply_gradient_op, batchnorm_updates_op)

    with tf.control_dependencies([apply_gradient_op_with_bn_update, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def train_resnet_fc(total_loss, global_step):
    raise NotImplementedError("dropout, freeze bn not, etc. Need modifying.")
    """ momentum sgd with lr exponentially decay """

    # Variables that affect learning rate.
    num_batches_per_epoch = num_examples_per_epoch_for_train / batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate_mSGD,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = utils.add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.MomentumOptimizer(lr, momentum)
        grads = opt.compute_gradients(
            total_loss, 
            var_list=tf.get_collection(resnet.RESNET_FC_VARIABLES))

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    if not use_minimal_summary:
        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(
        tf.get_collection(resnet.RESNET_FC_VARIABLES))

    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    apply_gradient_op_with_bn_update = tf.group(apply_gradient_op, batchnorm_updates_op)

    with tf.control_dependencies([apply_gradient_op_with_bn_update, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def inference_resnet(images, is_training, layer, dropout_keep_prob, freeze_bn=False):
    if layer == 50:
        num_blocks = [3, 4, 6, 3]
    elif layer == 101:
        num_blocks = [3, 4, 23, 3]
    elif layer == 152:
        num_blocks = [3, 8, 36, 3]
    else:
        raise ValueError("Invalid resnet layer number.")
    return resnet.inference(images,
                            num_classes=c.num_classes,
                            is_training=is_training,
                            bottleneck=True,
                            num_blocks=num_blocks,
                            freeze_bn=freeze_bn,
                            dropout_keep_prob=dropout_keep_prob)


def loss_resnet(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_loss_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                        name='cross_entropy_loss')

    # l2 regularizer
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)

    return loss_


def main_resnet():
    # Get the list of filenames and corresponding list of labels for training et validation
    train_filenames, train_labels = scene_input.list_images('train')
    val_filenames, val_labels = scene_input.list_images('validation')

    # --------------------------------------------------------------------------
    # In TensorFlow, you first want to define the computation graph with all the
    # necessary operations: loss, training op, accuracy...
    # Any tensor created in the `graph.as_default()` scope will be part of `graph`
    graph = tf.Graph()
    with graph.as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        is_training = tf.placeholder(tf.bool, [], "is_training")
        dropout_keep_prob = tf.placeholder(tf.float32, [], "dropout_keep_prob")

        images, labels, train_data_init_op, val_data_init_op \
            = scene_input.get_dataset_with_random_scale(train_filenames, train_labels,
                                                        val_filenames, val_labels, batch_size)

        if not use_minimal_summary:
            tf.image_summary('images', images)

        with tf.name_scope('inference'):
            logits= inference_resnet(
                images, is_training, resnet_layer, dropout_keep_prob)

        with tf.name_scope('loss'):
            loss_ = loss_resnet(logits, labels)

        with tf.name_scope('train'):
            full_train_op = train_resnet(loss_, global_step)

        with tf.name_scope('evaluation'):
            # Evaluation metrics
            prediction = tf.to_int32(tf.argmax(logits, 1))
            labels = tf.to_int32(labels)
            correct_prediction = tf.equal(prediction, labels)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            correct_prediction_top3 = tf.nn.in_top_k(logits, labels, 3)
            accuracy_top3 = tf.reduce_mean(tf.cast(correct_prediction_top3, tf.float32))

        init_var_op = tf.global_variables_initializer()

        merged_summary = tf.summary.merge_all()

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

        # tf.get_default_graph().finalize()

    # --------------------------------------------------------------------------
    # Now that we have built the graph and finalized it, we define the session.
    # The session is the interface to *run* the computational graph.
    # We can call our training operations with `sess.run(train_op)` for instance
    print("writing acc&loss info to {}".format(acc_loss_path))
    with tf.Session(graph=graph) as sess, open(acc_loss_path, "w") as f:
        sess.run(init_var_op)

        # tensorboard writer.
        writer = tf.summary.FileWriter(log_dir, graph)

        # restore pretrained model (only conv layers).
        variables_to_restore = resnet.get_variables_except_fc()
        # variables_to_restore = tf.get_collection(resnet.RESNET_VARIABLES)
        pretrained_model_saver = tf.train.Saver(variables_to_restore)
        pretrained_model_path = os.path.join(model_dir, resnet.checkpoint_fn(resnet_layer))
        if os.path.exists(pretrained_model_path):
            print("Loading pretrained model from :{}".format(pretrained_model_path))
            pretrained_model_saver.restore(sess, pretrained_model_path)
        else:
            raise ValueError(
                "Cannot find pretrained model: {}".format(pretrained_model_path))

        # check model_dir for checkpoint file.
        restore_epoch = 13 # designate a specific epoch to restore or None.
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if restore_epoch == None and ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            restore_epoch = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[1])
            restore_step = (restore_epoch - 1) * steps_per_epoch
            sess.run(global_step.assign(restore_step))
            print("restore check point from: {}".format(ckpt.model_checkpoint_path))
            print("get epoch: {} step: {}".format(restore_epoch, restore_step))
        elif restore_epoch != None:
            restore_model_path = os.path.join(checkpoint_dir, "scene-{}".format(restore_epoch))
            saver.restore(sess, restore_model_path)
            restore_step = (restore_epoch - 1) * steps_per_epoch
            sess.run(global_step.assign(restore_step))
            print("restore check point from: {}".format(restore_model_path))
            print("get epoch: {} step: {}".format(restore_epoch, restore_step))    
        else:
            print('No checkpoint found.')

        # Train the entire model for a few more epochs, continuing with the *same* weights.
        start_time = time.time()
        
        f.write("sampled train acc, sampled train loss, val acc top1, val acc top3\n")
        for epoch in range(num_epochs):
            if restore_epoch is not None and epoch < restore_epoch:
                continue
            else:
                print('epoch {} / {}'.format(epoch + 1, num_epochs))
            tick = time.time()
            sess.run(train_data_init_op)
            for i in tqdm.tqdm(range(steps_per_epoch)):
                try:
                    if tensorboard_write_frq > 0 and i % tensorboard_write_frq == 0:
                        feed_dict = {is_training: True, dropout_keep_prob: keep_prob}
                        _, summary = sess.run([full_train_op, merged_summary], feed_dict)
                        writer.add_summary(summary, epoch * steps_per_epoch + i)
                    else:
                        feed_dict = {is_training: True, dropout_keep_prob: keep_prob}
                        _ = sess.run(full_train_op, feed_dict)
                except tf.errors.OutOfRangeError:
                    break
            tock = time.time()
            # print(check_time(tock - start_time, steps_per_epoch, tock - tick))

            # check point
            if (epoch + 1) % checkpoint_freq == 0:
                saver.save(sess, os.path.join(checkpoint_dir, 'scene'), global_step=epoch + 1)

            # Check on the train and val sets every epoch.
            train_acc, train_loss = check_train(sess, correct_prediction, is_training, train_data_init_op, 
                dropout_keep_prob=dropout_keep_prob, n_batch=int(5000/batch_size), loss=loss_)
            print('Train: accuracy {0:.4f} loss {1:.4f}'.format(train_acc, train_loss))
            val_acc, val_acc_top3 = check_val(sess, correct_prediction, is_training, val_data_init_op, 
                dropout_keep_prob=dropout_keep_prob, correct_prediction_top3=correct_prediction_top3)
            print('Val: accuracy (top1){0:.4f} (top3){1:.4f}'.format(val_acc, val_acc_top3))
            f.write("{epoch} {0:.4f} {1:.4f} {2:.4f} {3:.4f}\n".format(
                train_acc, train_loss, val_acc, val_acc_top3, epoch=epoch + 1))


if __name__ == '__main__':
    main_resnet()
