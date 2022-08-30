"""AVOD VGG Network."""
import os
import tensorflow as tf
import fan_track.avod_feature_extractor.avod_img_vgg_graph as avod_img_vgg
slim = tf.contrib.slim
framework = tf.contrib.framework
from avod import root_dir as avod_root_dir
from avod.core import anchor_projector
from avod.core import constants

import numpy as np

from avod.core import box_3d_encoder

class ImgVGGNet():
    
    def __init__(self,avod_model_path):
        
        # img_vgg graph
        self.g = tf.Graph()
        
        # feature maps from the img_vgg
        self.feat_maps = None
        
        # specify where the avod model was saved.
        self.avod_model_path = avod_model_path
        
        root_dir = os.path.dirname(os.path.abspath(__file__))
    
        # specify where the new model will live: 
        vgg_checkpoint_path = root_dir  + "/checkpoints/img_vgg"
        
        self.avod_file = tf.train.latest_checkpoint(self.avod_model_path) 
        
        self.vgg_ckpt_file = vgg_checkpoint_path + "/vgg_16.ckpt"
        
        if not tf.gfile.Exists(vgg_checkpoint_path):
            tf.gfile.MakeDirs(vgg_checkpoint_path)

    def setup(self,input_pixel_size,img_depth,model_config,roi_crop_size):
        '''construction phase, that assembles a graph,'''
        
        img_feature_extractor = avod_img_vgg.ImgVgg(model_config)
        
        # Create the graph
        with self.g.as_default():
            
            with tf.variable_scope('img_input'):
                # dummy nodes that provide entry points for input image to the graph
                img_input_placeholder = tf.placeholder(tf.float32,[None, None, img_depth],"in")
                img_input_batches = tf.expand_dims(img_input_placeholder, axis=0)
        
                # the preprocessed tensor object
                img_preprocessed = img_feature_extractor.preprocess_input(img_input_batches,input_pixel_size)
    
            # define the img_vgg model 
            self.end_points = img_feature_extractor.build(img_preprocessed,input_pixel_size,False) 
              
            # Extract ROIs
            img_rois, img_pred_boxes = self.RoIPool(roi_crop_size,input_pixel_size)
            
            self.end_points['roi_pooling/crop_resize/img_rois'] = img_rois
            self.end_points['roi_pooling/crop_resize/img_pred_boxes'] = img_pred_boxes
            
    def get_box_indices(self,boxes):
                        proposals_shape = boxes.get_shape().as_list()
                        if any(dim is None for dim in proposals_shape):
                            proposals_shape = tf.shape(boxes)
                        ones_mat = tf.ones(proposals_shape[:2], dtype=tf.int32)
                        multiplier = tf.expand_dims(
                            tf.range(start=0, limit=proposals_shape[0]), 1)
                        return tf.reshape(ones_mat * multiplier, [-1])
       
    def RoIPool(self,roi_crop_size,input_pixel_size):
        
        with tf.variable_scope('roi_pooling'):
            # entry points for predicted bboxes in the in the format [x, y, z, dim_x, dim_y, dim_z]
            pred_placeholder = tf.placeholder(tf.float32,[None,6],'predictions')
            
            calib_placeholder = tf.placeholder(tf.float32, [3, 4], 'stereo_calib')
            
            img_size = tf.constant(input_pixel_size, dtype=tf.float32)
                    
            with tf.variable_scope('crop_resize'):
                    
                    # project 3d predicted bboxes onto the image place
                    img_pred_boxes, img_pred_boxes_norm = \
                                 anchor_projector.tf_project_to_image_space(pred_placeholder,calib_placeholder,img_size)  
                                                                  
                                                                  
                    # reorder the projected bboxes for crop and resizing
                    img_pred_boxes_norm_tf_order = anchor_projector.reorder_projected_boxes(img_pred_boxes_norm)
                    
                    img_feature_maps = self.end_points['img_vgg/upsampling/maps_out']

                    img_boxes_norm_batches = tf.expand_dims(img_pred_boxes_norm, axis=0)

                    # These should be all 0's since there is only 1 image since box_indices[i] specifies image
                    box_indices = self.get_box_indices(img_boxes_norm_batches)
                    
                    # Do ROI Pooling on image
                    img_rois = tf.image.crop_and_resize(img_feature_maps,img_pred_boxes_norm_tf_order,box_indices,roi_crop_size,name='img_roi')
                    
                    
                    return img_rois, img_pred_boxes   
                  
    def restore_save(self):
        '''restore img_vgg from the avod's checkpoint'''
        
        with tf.Session(graph = self.g) as sess:
            # Get all variables,i.e. only those created in the bev_vgg to restore
            variables_to_restore = slim.get_variables_to_restore(exclude=['roi_pooling/crop_resize/img_shape'])
            
            # Restore only the convolutional layers from the avod checkpoint
            init_fn = framework.assign_from_checkpoint_fn(self.avod_file, variables_to_restore)
            
            # create the saver object to restore the variables of the bev_vgg
            saver = tf.train.Saver()
            
            # restore the bev_vgg's variables from the avod's checkpoint file
            if not(init_fn is None):
                with tf.Session(graph=self.g) as sess:
                    # perform the assignment, i.e.,restore
                    init_fn(sess)                       
                    saver.save(sess, self.vgg_ckpt_file)
                    print("img_vgg checkpoint file was saved.")
                    
    def restore_graph(self):

        # sess will launch/handle img_vgg graph
        self.sess = tf.Session(graph=self.g)
        
        # make the img_vgg the default graph    
        with self.g.as_default():
            # create a saver object to restore the variables
            saver = tf.train.Saver()
            # Restore variables from disk
            saver.restore(self.sess, self.vgg_ckpt_file)
            
    def close_session(self):
        '''close the session running the img_vgg graph'''
        self.sess.close()
            
    def extract_features(self,input_img,predictions,stereo_calib):
        '''launch the img_vgg graph to extract img feature maps'''
        
        # access the input image placeholder variable
        img_input_placeholder = self.g.get_tensor_by_name("img_input/in:0")
        
        pred_placeholder = self.g.get_tensor_by_name("roi_pooling/predictions:0")
        
        calib_placeholder = self.g.get_tensor_by_name("roi_pooling/stereo_calib:0")
        
        # access the op that you want to run. 
        feature_maps_out = self.end_points['img_vgg/upsampling/maps_out']
        
        img_rois = self.end_points['roi_pooling/crop_resize/img_rois']
        
        img_pred_boxes = self.end_points['roi_pooling/crop_resize/img_pred_boxes']
        
        # return outputs from the img_vgg
        feat_maps, obj_rois, obj_pred_boxes = self.sess.run([feature_maps_out,img_rois,img_pred_boxes],
                                                             feed_dict = {img_input_placeholder:input_img,
                                                                          pred_placeholder:predictions,
                                                                          calib_placeholder:stereo_calib})
            
        # print("img outputs were obtained.") 
        return feat_maps, obj_rois, obj_pred_boxes
           
        
    def checkRestore(self):  
        """Check variables are correctly restored to img_vgg"""
               
        w_img_vgg = []

        # launch the img_vgg graph 
        with tf.Session(graph=self.g) as sess:
            saver = tf.train.Saver()
            
            # Restore variables from disk.
            saver.restore(sess, self.vgg_ckpt_file)

            for v in tf.trainable_variables():
                if (v.name.find("img_vgg") != - 1):
                    if (v.name.find("weights")  != -1):
                        w_img_vgg.append(sess.run(v))
                    elif (v.name.find("bias")  != -1):
                        w_img_vgg.append(sess.run(v))
                    elif (v.name.find("beta")  != -1):
                        w_img_vgg.append(sess.run(v))
                    elif (v.name.find("gamma")  != -1):
                        w_img_vgg.append(sess.run(v))
                        
        w_avod_img_vgg = []

        # launch the avod's graph
        avod_graph = tf.Graph() 
        with tf.Session(graph=avod_graph) as sess:
            # append the network defined in meta file to the current graph    
            saver = tf.train.import_meta_graph(self.avod_file + ".meta")
            
            # restore the values of the trained parameters 
            saver.restore(sess, self.avod_file)
            
            for v in tf.trainable_variables():
                if (v.name.find("img_vgg") != - 1):
                    if (v.name.find("weights")  != -1):
                        w_avod_img_vgg.append(sess.run(v))
                    elif (v.name.find("bias")  != -1):
                        w_avod_img_vgg.append(sess.run(v))
                    elif (v.name.find("beta")  != -1):
                        w_avod_img_vgg.append(sess.run(v))
                    elif (v.name.find("gamma")  != -1):
                        w_avod_img_vgg.append(sess.run(v))
                        
        
        for i,w in enumerate(w_img_vgg):
            if not(np.array_equal(w,w_avod_img_vgg[i])):
                print("resotoring img_vgg failed.")
                return False;
            
        # print("img_vgg restored successfully.")
        return True