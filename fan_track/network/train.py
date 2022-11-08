import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from fan_track.data_generation.batch_sampler import BatchSampler
import os
import matplotlib.pyplot as plt
from functools import reduce
from fan_track.utils.generic_utils import TrainingType
from fan_track.config.config import *
from fan_track.network.tracker import build_global_maps
import progressbar
import datetime
import tqdm
from fan_track.utils import generic_utils

class Trainer(object):
    """
    Trains a AssocNet instance
    :param model: the model instance to train
    :param args: parsed arguments
    :param optimizer: (optional) name of the optimizer to use (momentum or adam)
    """

    def __init__(self, model, args):
        self.args = args
        self.model = model
        self.batch_size = self.args.batch_size
        self.working_dir = os.path.join(generic_utils.get_project_root(), 'data')

        if args.training_type != TrainingType.Simnet:
            self.batch_sampler = BatchSampler()
            self.batch_sampler.load_dataset_files()

    def get_optimizer(self):

        self.learning_rate_node = tf.train.exponential_decay(AssocnetConfig.LEARNING_RATE,
                                                                 self.model.in_global_step_assoc_ph,
                                                                 AssocnetConfig.DECAY_STEP,
                                                                 AssocnetConfig.DECAY_RATE,
                                                                 staircase = True
                                                            )

        if GlobalConfig.OPTIMIZER == "Momentum":

            optimizer = tf.train.MomentumOptimizer(self.learning_rate_node,
                                                   AssocnetConfig.MOMENTUM,
                                                   use_nesterov=True)
        else:
            optimizer = tf.train.AdamOptimizer(self.learning_rate_node)


        # create a train op: compute the loos, apply the gradient and return the loss value
        train_op = slim.learning.create_train_op(self.model.loss, optimizer)

        return train_op

    def _initialize(self, output_path):

        self.asoocNet_train_op = self.get_optimizer()

        tf.summary.scalar('averege_loss', self.model.avg_loss)
        tf.summary.scalar('average_accuracy', self.model.avg_accuracy)

        tf.summary.scalar('learning_rate', self.learning_rate_node)

        self.summary_op = tf.summary.merge_all()

        # global variables are saved/restored to checkpoint like weights
        global_init = tf.global_variables_initializer()

        self.model.sess.run(global_init)

        # local variables are not save to checkpoint like counters for number of epochs
        local_init = tf.local_variables_initializer()

        self.model.sess.run(local_init)

        self._create_directories(output_path)

    def _create_directories(self,path):
        # obtain the complete path of the file which is located in the current working directory
        path = os.path.abspath(path)

        if not os.path.exists(path):
            os.makedirs(path)

    def train_simnet(self, mb_sampler):
        '''
           Train the simnet network.
        '''

        checkpoint_path = os.path.join(self.working_dir,'simnet','checkpoints')
        datetime_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        train_log_path = os.path.join(self.working_dir,'simnet','logs','training',datetime_str)
        val_log_path = os.path.join(self.working_dir,'simnet','logs','validation',datetime_str)

        self._create_directories(train_log_path)
        self._create_directories(val_log_path)

        # tensorflow event file which contains summaries for training
        train_writer = tf.summary.FileWriter(train_log_path,self.model.graph)
        valid_writer = tf.summary.FileWriter(val_log_path,self.model.graph)

        # initialize variables
        global_init = tf.global_variables_initializer()
        self.model.sess.run(global_init)
        local_init = tf.local_variables_initializer()
        self.model.sess.run(local_init)

        # add operation to save weights of the network
        saver = tf.train.Saver(max_to_keep=None)

        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        if latest_checkpoint is not None:
            print('Restoring Checkpoint')
            saver.restore(self.model.sess, latest_checkpoint)

        epoch_bar = tqdm.tqdm(total=SimnetConfig.EPOCHS, desc='Epoch', position=0)

        for epoch in range(SimnetConfig.EPOCHS):
            batch_accuracies = []

            # mini batch index
            mb_sampler.shuffle_minibatches(dataset_type = 'training')

            step_bar = tqdm.tqdm(total=mb_sampler.training_mbatch_count, desc='Batch', position=1)
            step_bar_val = tqdm.tqdm(total=mb_sampler.validation_mbatch_count, desc='Val', position=2)

            mb_idx = 0
            while (True):

                mb_idx, mb = mb_sampler.next_mb(mb_idx = mb_idx,
												operation = 'training',
												mb_dir = GlobalConfig.SIMNET_MINIBATCH_PATH)
                # end of each epoch
                if (mb_idx == -1):
                    break

                num_pairs = mb.item()['labels'].shape[0]

                if mb.item()['bbox'].shape[-1] !=1:
                    bbox_features = np.expand_dims(mb.item()['bbox'],axis = 3)
                else:
                    bbox_features = mb.item()['bbox']

                summary,simnet_accuracy, _, = self.model.sess.run([self.model.simnet_summary_op, self.model.simnet_accuracy, self.model.simnet_train_op],
                                       feed_dict={self.model.in_bbox_ph: bbox_features,
                                                  self.model.in_appear_ph: mb.item()['feat'],
                                                  self.model.simnet_labels_ph: mb.item()['labels'],
                                                  self.model.num_target_ph: [num_pairs],
                                                  self.model.global_step: epoch,
                                                  self.model.training_ph: True,
                                                  }
                                       )
                step_bar.update(1)
                batch_accuracies.append(simnet_accuracy)

            # write the summary data
            train_writer.add_summary(summary, epoch)

            epoch_accuracy = np.mean(batch_accuracies)
            #print('Epoch:{0} Training Accuracy:{1}'.format(epoch, epoch_accuracy))

            # Save Checkpoints
            if ((epoch + 1) % SimnetConfig.SAVE_RATE == 0):
                saver.save(self.model.sess, os.path.join(checkpoint_path,'simnet.ckpt'), epoch)

            # Validation
            if ((epoch + 1) % SimnetConfig.VALIDATION_RATE == 0):

                #tqdm.tqdm.write('\nValidation')
                total_val_loss = 0
                total_val_true_positives = 0
                total_val_true_negatives = 0
                total_val_false_positives = 0
                total_val_false_negatives = 0
                val_cosine_similarities = []
                total_val_comparisons = []

                val_idx = 0

                while(True):

                    val_idx, val_mb = mb_sampler.next_mb(mb_idx=val_idx,
														 operation='validation',
														 mb_dir=GlobalConfig.SIMNET_MINIBATCH_PATH)

                    if (val_idx == -1):
                        break

                    val_num_pairs = val_mb.item()['labels'].shape[0]

                    if val_mb.item()['bbox'].shape[-1] != 1:
                        bbox_features = np.expand_dims(val_mb.item()['bbox'], axis=3)
                    else:
                        bbox_features = val_mb.item()['bbox']

                    val_loss, true_positives, true_negatives, false_positives, false_negatives, val_comparisons, cosine_sim = self.model.sess.run([
                                                                                       self.model.simnet_loss,
                                                                                       self.model.TP,
                                                                                       self.model.TN,
                                                                                       self.model.FP,
                                                                                       self.model.FN,
                                                                                       self.model.comparison,
                                                                                       self.model.cosine_sim
                                                                                       ],
                                                                                      feed_dict={
                                                                                          self.model.in_bbox_ph: bbox_features,
                                                                                          self.model.in_appear_ph: val_mb.item()['feat'],
                                                                                          self.model.simnet_labels_ph: val_mb.item()['labels'],
                                                                                          self.model.num_target_ph: [val_num_pairs],
                                                                                          self.model.training_ph: False
                                                                                          }
                                                                                      )

                    total_val_loss += val_loss
                    total_val_true_positives += true_positives
                    total_val_true_negatives += true_negatives
                    total_val_false_positives += false_positives
                    total_val_false_negatives += false_negatives
                    val_cosine_similarities.extend(cosine_sim)
                    total_val_comparisons.extend(val_comparisons)

                    step_bar_val.update(1)

                average_val_loss = total_val_loss/mb_sampler.validation_mbatch_count
                val_accuracy = np.mean(total_val_comparisons)
                val_precision = total_val_true_positives / (total_val_true_positives + total_val_false_positives)
                val_recall = total_val_true_positives / (total_val_true_positives + total_val_false_negatives)

                val_summary = tf.Summary()
                val_summary.value.add(tag="loss_opt_pred/accuracy_1", simple_value=val_accuracy)
                val_summary.value.add(tag='loss_opt_pred/loss', simple_value=average_val_loss)

                # write the summary data
                valid_writer.add_summary(val_summary, epoch)
                # neg_idx, _ = np.nonzero(np.multiply(val_cosine_similarities, (mb_sampler.val_labels - 1)))
                #
                # hist, _ = np.histogram(val_cosine_similarities[neg_idx], bins=np.arange(-1, 1.1, 0.25), range=(-1.0, 1.0))
                # print('histogram of negatives:{0}'.format(hist))

                tqdm.tqdm.write('\nEpoch {0:d}'.format(epoch))
                tqdm.tqdm.write('\nValidation Accuracy:{0:.4f}, Loss:{1:.5f}'.format(val_accuracy, total_val_loss))
                tqdm.tqdm.write('\nValidation Precision:{0:.3f}, Validation Recall:{1:0.3f}'.format(val_precision, val_recall))
                tqdm.tqdm.write('\n------------------------------------------------------')

            epoch_bar.update(1)

    def train_assocnet(self):
        '''
            Launches the training process
        :Arguments:
                output_path: path where to store checkpoints
                restore: Flag if previous model should be restored
                prediction_path: path where to save predictions on each epoch
        '''

        checkpoint_path = os.path.join(self.working_dir,'assocnet','checkpoints')
        save_path = os.path.join(checkpoint_path, "assocnet.ckpt")

        output_path_train = os.path.join(self.working_dir,'tensorboard/assocNet_trained')

        output_path_valid = os.path.join(self.working_dir,'tensorboard/assocNet_validation')

        # Initialize all variables and parameters
        self._initialize(output_path_train)

        # tensorflow event file which contains summaries for training
        summary_writer = tf.summary.FileWriter(output_path_train, graph=self.model.graph)

        # tensorflow event file which contains summaries for validation
        self.summary_writer_val = tf.summary.FileWriter(output_path_valid, graph=self.model.graph)

        # add operation to save weights of the network
        saver = tf.train.Saver(max_to_keep=None)

        # restore the weights from the checkpoint
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        if latest_checkpoint is not None:
            print('Restoring Checkpoint')
            saver.restore(self.model.sess, latest_checkpoint)

        print('Start optimization')

        sim_maps = np.load(os.path.join(self.working_dir, 'sim_maps_train.npy'), allow_pickle=True)
        #print("sim maps:", sim_maps)

        epoch_bar = tqdm.tqdm(total=self.model.args.epochs, desc='Epoch', position=0)

        for epoch in range(self.model.args.epochs):

            # shuffle the examples
            np.random.shuffle(sim_maps)

            # average training loss
            avg_loss = 0.0

            # average accuracy of predictions
            summary = None
            avg_accuracy = 0.

            step_bar = tqdm.tqdm(total=len(sim_maps), desc='Batch', position=1)

            for exp_idx,example in enumerate(sim_maps):

                # TODO : Replace the train op
                # train the assoc_net
                _, summary, avg_loss, avg_accuracy, maps, m_pred_x, m_pred_y = self.model.sess.run(
                                                                                [   self.asoocNet_train_op,
                                                                                    self.summary_op,
                                                                                    self.model.avg_loss,
                                                                                    self.model.avg_accuracy,
                                                                                    self.model.sliced_maps,
                                                                                    self.model.x_pred,
                                                                                    self.model.y_pred],
                                                                                    feed_dict={
                                                                                            self.model.in_local_corr_map_ph: example['local_corr_map'],
                                                                                            self.model.in_num_target_array_ph: [example['num_targets']],
                                                                                            self.model.in_assoc_labels_ph: example['labels'],
                                                                                            self.model.in_exp_index_ph: exp_idx,
                                                                                            self.model.in_avg_loss_ph: avg_loss,
                                                                                            self.model.in_avg_accuracy_ph: avg_accuracy,
                                                                                            self.model.in_global_step_assoc_ph: epoch,
                                                                                            #self.model.in_training_deConvNet_ph: True
                                                                                            self.model.in_training_assocnet_ph: True
                                                                                            }
                                                                            )
                associations = self.association_results(example['targets_xy'],
                                                        example['num_targets'],
                                                        example['measurements_xy'],
                                                        example['local_corr_map'],
                                                        m_pred_x,
                                                        m_pred_y)
                step_bar.update(1)
                # if (exp_idx == 100 and example['num_targets'] > 0):
                #
                #     fig = plt.figure()
                #
                #     ax = fig.add_subplot(131)
                #     ax.set_title('Prediction')
                #     ax.imshow(maps[0][:,:,0])
                #
                #
                #     ax = fig.add_subplot(132)
                #     ax.set_title('Local Map')
                #     ax.imshow(example['local_corr_map'][0][:,:,0])
                #
                #     ax = fig.add_subplot(133)
                #     ax.set_title('Label Map')
                #     ax.imshow(example['labels'][0][:,:,0])
                #
                #     plt.savefig( './data/assocnet/predictions/' + str(epoch)+'.png')
                #
                #     plt.close()

            # write the summary data
            if (summary):
                summary_writer.add_summary(summary, epoch)

            tqdm.tqdm.write('Training epoch {:}, Average loss: {:.4e}, Average accuracy: {:.4e}'.format(epoch,avg_loss, avg_accuracy))

            self.validate_network(epoch)

            # save network variables
            saver.save(self.model.sess, save_path, epoch)
            epoch_bar.update(1)

        print("Optimization Finished!")

        return save_path

    def validate_network(self, epoch):

        # average validation loss
        avg_loss = 0

        # average accuracy of predictions
        avg_accuracy = 0

        sim_maps = np.load(os.path.join(self.working_dir, 'sim_maps_val.npy'), allow_pickle=True)
        summary = None

        accuracies = []
        for num_exp, example in enumerate(sim_maps):

            summary, avg_loss, maps, accuracy,correct_predictions, avg_accuracy = self.model.sess.run([self.summary_op,
                                                                         self.model.avg_loss,
                                                                         self.model.sliced_maps,
                                                                         self.model.accuracy,
                                                                         self.model.correct_pred,
                                                                         self.model.avg_accuracy],
                                                                         feed_dict={
                                                                                      self.model.in_local_corr_map_ph: example['local_corr_map'],
                                                                                      self.model.in_num_target_array_ph: [example['num_targets']],
                                                                                      self.model.in_assoc_labels_ph: example['labels'],
                                                                                      self.model.avg_loss: avg_loss,
                                                                                      self.model.avg_accuracy: avg_accuracy,
                                                                                      self.model.in_exp_index_ph: num_exp,
                                                                                      self.model.in_global_step_assoc_ph: epoch,
                                                                                      self.model.in_training_assocnet_ph: False
                                                                                    }

                                                                        )
            accuracies.append(accuracy)

        print('Validation Accuracy:{0}'.format(np.mean(accuracies)))
        # write the summary data
        if (summary):
            self.summary_writer_val.add_summary(summary, epoch)

        print('Validation epoch {:}, Average loss: {:.4e}, Average accuracy: {:.4e}\n'.format(epoch,avg_loss, avg_accuracy))

    def run_simnet(self):
        ''' run simnet and save simnet ouputs for the assocnet'''
        # training data
        self.create_similarity_maps(is_training_flag = True, filename = os.path.join(self.working_dir, 'sim_maps_train.npy'))

        # validation data
        self.create_similarity_maps(is_training_flag = False, filename = os.path.join(self.working_dir, 'sim_maps_val.npy'))

    def create_similarity_maps(self, is_training_flag, filename):

        if os.path.exists(filename):
            return

        if (is_training_flag):
            print('Running simnet to save training dataset for assocnet')
        else:
            print('Running simnet to save validation dataset for assocnet')

        # index of the example from the dataset
        index = 0

        # simnet output collection for assocnet
        simnet_output_collection = []


        while(True):

            index,frame_pair_data = self.batch_sampler.get_frame_pairs(is_training = is_training_flag, count = 1, startIndex = index)

            if (index == -1 or frame_pair_data is None):
                break

            frame_pair_data = frame_pair_data[0]

            #frame_pair_data['labels'] = np.expand_dims(frame_pair_data['labels'], axis = 0)
            measurement_bboxes = frame_pair_data['measurement_bboxes']
            target_bboxes = frame_pair_data['target_bboxes']
            measurement_bboxes = np.expand_dims(measurement_bboxes, axis=1)
            measurement_bboxes = np.expand_dims(measurement_bboxes, axis=3)
            target_bboxes = np.expand_dims(target_bboxes, axis=1)
            target_bboxes = np.expand_dims(target_bboxes, axis=3)
            # run the first network to save intermediate maps
            m_bbox_feat, m_appear_feat, m_bbox_weights, m_appear_weights, \
            t_bbox_feat, t_appear_feat, t_bbox_weights, t_appear_weights = self.model.sess.run(
                                    [self.model.m_feat1,
                                    self.model.m_feat2,
                                    self.model.w_bbox_m,
                                    self.model.w_appear_m,
                                    self.model.t_feat1,
                                    self.model.t_feat2,
                                    self.model.w_bbox_t,
                                    self.model.w_appear_t],
                                    feed_dict={
                                                #self.model.in_bbox_ph: frame_pair_data['bbox'],
                                                #self.model.in_appear_ph: frame_pair_data['feat'],
                                                self.model.in_bbox_tar_ph: target_bboxes,
                                                self.model.in_bbox_meas_ph: measurement_bboxes,
                                                self.model.in_appear_tar_ph: frame_pair_data['target_avod_feat'],
                                                self.model.in_appear_meas_ph: frame_pair_data['measurement_avod_feat'],
                                                self.model.num_target_ph: [frame_pair_data['num_targets']],
                                                self.model.training_ph: False,
                                            }
                                    )

            local_corr_map = build_global_maps(frame_pair_data['meas_locations'], m_bbox_feat, m_appear_feat, m_bbox_weights, m_appear_weights,
                            t_bbox_feat, t_appear_feat, t_bbox_weights, t_appear_weights, frame_pair_data['targets_xy'],
                            frame_pair_data['num_targets'])

            simnet_output = {
                                'local_corr_map': local_corr_map,
                                'labels': frame_pair_data['labels'],
                                'num_targets': frame_pair_data['num_targets'],
                                'targets_xy': frame_pair_data['targets_xy'],
                                'measurements_xy': frame_pair_data['meas_locations']
                            }

            simnet_output_collection.append(simnet_output)

            print('Index:{0}'.format(index))

        np.save(filename, simnet_output_collection)

        print('sim_maps saved')

    def association_results(self, t_centers, num_targets, m_centers, corr_maps, m_pred_x, m_pred_y):
        '''
            Obtain the assignment of measurement labels to target labels.
            :Arguments:
                t_centers: array of target indices in the global map.
                num_targets: the number of targets being tracked.
                m_centers: array of measurement indices in the global map.
                corr_maps: local correlation maps for each targets.
                m_pred_x: the row indices of the predicted pixels in local maps.
                m_pred_y: the column indices of the predicted pixels in local maps.
            :Returns:
                A list of 2-tuples showing target and measurement indices like (i,j) where
                i and j are orders of targets and measurements in the inputs to the network.

        '''

        # target to measurement association scores
        assoc_results  = []

        # list of correlation scores for associations
        corr_scores = []

        for t_idx in range(num_targets):

            # local row and column indices of the measurement associated with target with the index  t_idx
            z_x = m_pred_x[t_idx]
            z_y = m_pred_y[t_idx]

            # targets are centered at local maps
            tar_pos_idx = int((GlobalConfig.CROP_SIZE - 1)/2 + 1)

            # compute the global indices of the measurement
            delta_x = z_x - tar_pos_idx
            delta_y = z_y - tar_pos_idx

            z_global_x = t_centers[t_idx][0,0] + delta_x
            z_global_y = t_centers[t_idx][0,1] + delta_y

            for m_idx,z_xy in enumerate(m_centers,1):

                flag = reduce(lambda x,y: x and y, z_xy == (z_global_x,z_global_y) )

                if (flag):
                    break

            if not(flag):
                m_idx = None

            # find the measurement index
            assoc_results.append((t_idx+1,m_idx))


        return assoc_results

    def train_combined_network(self):

        save_path = os.path.join(generic_utils.get_project_root(),'data','assocnet','checkpoints', "assocnet.ckpt")
        fig_path = os.path.join(generic_utils.get_project_root(),'data','assocnet','prediction')

        output_path_train = os.path.join(generic_utils.get_project_root(),'data','tensorboard/assocNet_trained')

        output_path_valid = os.path.join(generic_utils.get_project_root(),'data','tensorboard/assocNet_validation')

        # Initialize all variables and parameters
        self._initialize(output_path_train)

        # tensorflow event file which contains summaries for training
        summary_writer = tf.summary.FileWriter(output_path_train, graph=self.model.sess.graph)

        # tensorflow event file which contains summaries for validation
        self.summary_writer_val = tf.summary.FileWriter(output_path_valid, graph=self.model.sess.graph)

        # add operation to save weights of the network
        saver = tf.train.Saver(max_to_keep=None)

        # restore the weights from the checkpoint
        # saver.restore(self.model.sess, '/home/ebaser/Kitti/assocnet/checkpoint/assocnet.ckpt')

        print('Start optimization')

        for epoch in range(self.model.args.epochs):

            # shuffle the examples


            # average training loss
            avg_loss = 0.0

            # average accuracy of predictions
            avg_accuracy = 0.0

            exp_idx = 0
            index = 0

            # for exp_idx, example in enumerate(sim_maps):
            while(True):

                # swap dimensions to place target index to the last dimension
                # exp_labels = np.swapaxes(example['labels'], 1, 3)

                # swap width by height
                # exp_labels = np.swapaxes(exp_labels, 1, 2)

                index, frame_pair_data = self.batch_sampler.get_frame_pairs(is_training=True, count=1,
                                                                            startIndex=index)
                if (index == -1 or frame_pair_data is None):
                    break

                frame_pair_data = frame_pair_data[0]

                # frame_pair_data['labels'] = np.expand_dims(frame_pair_data['labels'], axis = 0)


                # train the assoc_net
                _, summary, avg_loss, avg_accuracy, maps, m_pred_x, m_pred_y, local_corr_map = self.model.sess.run(
                    [self.asoocNet_train_op,
                     self.summary_op,
                     self.model.avg_loss,
                     self.model.avg_accuracy,
                     self.model.sliced_maps,
                     self.model.x_pred,
                     self.model.y_pred,
                     self.model.local_corr_map,
                     ],
                    feed_dict={
                        self.model.in_bbox_ph: frame_pair_data['bbox'],
                        self.model.in_appear_ph: frame_pair_data['feat'],
                        self.model.num_target_ph: frame_pair_data['num_targets'],
                        self.model.in_meas_loc_ph: frame_pair_data['meas_locations'],
                        self.model.in_tar_cntr_xy: frame_pair_data['targets_xy'],
                        self.model.training_ph: True,
                        self.model.in_assoc_labels_ph: frame_pair_data['labels'],
                        self.model.in_exp_index_ph: exp_idx,
                        self.model.in_avg_loss_ph: avg_loss,
                        self.model.in_avg_accuracy_ph: avg_accuracy,
                        self.model.in_global_step_assoc_ph: epoch,
                        # self.model.in_training_deConvNet_ph: True
                        self.model.in_training_assocnet_ph: True
                    }
                )

                associations = self.association_results(frame_pair_data['targets_xy'],
                                                        frame_pair_data['num_targets'],
                                                        frame_pair_data['measurements_xy'],
                                                        m_pred_x,
                                                        m_pred_y)

                if (exp_idx == 100 and frame_pair_data['num_targets'] > 0):
                    fig = plt.figure()

                    ax = fig.add_subplot(131)
                    ax.set_title('Prediction')
                    ax.imshow(maps[0][:, :, 0])

                    ax = fig.add_subplot(132)
                    ax.set_title('Local Map')
                    ax.imshow(local_corr_map[0][:, :, 0])

                    ax = fig.add_subplot(133)
                    ax.set_title('Label Map')
                    ax.imshow(frame_pair_data['labels'][0][:, :, 0])

                    plt.savefig(fig_path + '/' + str(epoch) + '.png')

                    plt.close()

                    # write the summary data
            summary_writer.add_summary(summary, epoch)

            print('Training epoch {:}, Average loss: {:.4e}, Average accuracy: {:.4e}'.format(epoch, avg_loss,
                                                                                              avg_accuracy))
            self.validate_network(epoch)

            # save network variables
            saver.save(self.model.sess, save_path)

        print("Optimization Finished!")
