""" scene classification version2
    reference code see https://gist.github.com/omoindrot
    
    experiment result:
    1.  Add random rotation theta to data augmentation, truncated normal mean=0, 
        stddev=theta/2, range[-theta, theta]. There is always a increase in val 
        acc every time lr decay. From loss curve, first decay interval can be 
        longer, but from val acc, there is overfit. Best val acc all occur after
        two lr decay. (Weird 30 is always less good than 10 and 45)
        5: best val acc 0.9031
        10: best val acc 0.9024. 
        30: beas val acc 0.9003.
        45: best val acc 0.9018.
        90: best val acc 0.8934
        with out any rotation: 0.9038 stable. Rotation seems to have little good.
        Seems not because the overfitting, train error didn't saturate.
    2.  random scale. match images smaller side to [256, 480] and resize them.
        impressive improvement: val acc 0.9181 ~ 0.9187 stable.
    3.  color augmentation. best val acc 0.9013 0.9001~0.9006 stable. Not helpful
        to overfitting and acc. Wrong implementation?(different from AlexNet color
        augmentation)

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

import scene_input
import scene_vgg as vgg
import utility_tensorflow as utils
import tqdm

data_dir = "/scratch/lyc/ai_challenger_scene/data"
log_dir = "/scratch/lyc/ai_challenger_scene/tmp/log"
checkpoint_dir = '/scratch/lyc/ai_challenger_scene/tmp/checkpoint'
model_dir = "scratch/lyc/ai_challenger_scene/model"

dropout_keep_prob = 0.5
batch_size = 64

num_epochs = 60 # 1000

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.

# learning rate decay constant
NUM_EPOCHS_PER_DECAY = 25 # 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

# momentum SGD optimizer
initial_learning_rate_mSGD = 0.001
momentum = 0.9

num_examples_per_epoch_for_train = scene_input.num_examples_per_epoch_for_train
num_examples_per_epoch_for_val = scene_input.num_examples_per_epoch_for_val
steps_per_epoch = int(num_examples_per_epoch_for_train / batch_size) + 1

# write tensorboard info every freq steps
tensorboard_write_frq = 20
checkpoint_freq = 3


def check_train(sess, correct_prediction, keep_prob, dataset_init_op, n_batch, loss):
    """
    sampled accuracy and loss for training set.
    """
    # Initialize the correct dataset
    sess.run(dataset_init_op)
    num_correct, num_samples, sum_loss = 0, 0, 0
    for _ in tqdm.tqdm(range(n_batch)):
        try:
            corr_pred, l = sess.run([correct_prediction, loss], {keep_prob: 1.0})
            num_correct += corr_pred.sum()
            num_samples += corr_pred.shape[0]
            sum_loss += l
        except tf.errors.OutOfRangeError:
            break

    # return accuracy and loss.
    acc = float(num_correct) / num_samples
    sum_loss = float(sum_loss) / num_samples
    return acc, sum_loss


def check_val(sess, correct_prediction, keep_prob,
                       dataset_init_op, correct_prediction_top3):
    """
    Check the accuracy(top1 and top3) of the model on val.
    """
    # Initialize the correct dataset
    sess.run(dataset_init_op)
    num_correct, num_correct_top3, num_samples = 0, 0, 0
    num_steps_per_epoch = int(num_examples_per_epoch_for_val / batch_size) + 1
    for i in tqdm.tqdm(range(num_steps_per_epoch)):
        try:
            correct_pred, correct_pred_top3 = sess.run(
                [correct_prediction, correct_prediction_top3], {keep_prob: 1.0})
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


def train_momentum_sgd(total_loss, global_step):
    """Train scene model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps
            processed.
    Returns:
        train_op: op for training.
    """
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
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    # for var in tf.trainable_variables():
    #     tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    # for grad, var in grads:
    #     if grad is not None:
    #         tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def main():
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

        keep_prob = tf.placeholder(tf.float32)

        images, labels, train_data_init_op, val_data_init_op \
            = scene_input.get_dataset_with_color_augmentation(train_filenames, train_labels,
                                                              val_filenames, val_labels, batch_size)

        with tf.name_scope('inference'):
            conv_net = vgg.Vgg16()
            conv_net.build(images, keep_prob, scene_input.num_classes)
            logits = conv_net.get_softmax_linear()

        with tf.name_scope('loss'):
            # Calculate the average cross entropy loss across the batch.
            labels = tf.cast(labels, tf.int64)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits, name='cross_entropy_per_example')
            loss = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')

        with tf.name_scope('train'):
            full_train_op = train_momentum_sgd(loss, global_step)

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

        saver = tf.train.Saver()

        tf.get_default_graph().finalize()

    # --------------------------------------------------------------------------
    # Now that we have built the graph and finalized it, we define the session.
    # The session is the interface to *run* the computational graph.
    # We can call our training operations with `sess.run(train_op)` for instance
    with tf.Session(graph=graph) as sess:
        sess.run(init_var_op)

        # tensorboard writer.
        writer = tf.summary.FileWriter(log_dir, graph)

        # check model_dir for checkpoint file.
        restore_epoch = None
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/model/model-10.xxx,
            # extract epoch from it.
            restore_epoch = ckpt.model_checkpoint_path.split('/')[-1].split('-')[2]
            sess.run(global_step.assign((restore_epoch - 1) * steps_per_epoch))
            print("restore check point from: {}".format(ckpt.model_checkpoint_path))
            print("get epoch: {} step: {}".format(restore_epoch, (restore_epoch - 1) * steps_per_epoch))
        else:
            print('training whole conv net from scratch.')
        start_time = time.time()

        # Train the entire model for a few more epochs, continuing with the *same* weights.
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
                        _, summary = sess.run([full_train_op, merged_summary], {keep_prob: dropout_keep_prob})
                        writer.add_summary(summary, epoch * steps_per_epoch + i)
                    else:
                        _ = sess.run(full_train_op, {keep_prob: dropout_keep_prob})
                except tf.errors.OutOfRangeError:
                    break
            tock = time.time()
            print(check_time(tock - start_time, steps_per_epoch, tock - tick))

            # check point
            if (epoch + 1) % checkpoint_freq == 0:
                saver.save(sess, os.path.join(checkpoint_dir, 'scene'), global_step=epoch + 1)

            # Check on the train and val sets every epoch.
            train_acc, train_loss = check_train(sess, correct_prediction, keep_prob,
                                                train_data_init_op, n_batch=100, loss=loss)
            print('Train: accuracy {0:.4f} loss {1:.4f}'.format(train_acc, train_loss))
            val_acc, val_acc_top3 = check_val(sess, correct_prediction, keep_prob, val_data_init_op,
                                              correct_prediction_top3=correct_prediction_top3)
            print('Val: accuracy (top1){0:.4f} (top3){1:.4f}'.format(val_acc, val_acc_top3))


if __name__ == '__main__':
    main()
