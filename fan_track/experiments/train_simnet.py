import os
import time
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import glob
import fan_track.network.model as assoc_model
# from utils.generic_utils import prepare_args
from fan_track.data_generation.simnet_batch_sampler import SimnetBatchSampler
from fan_track.network.train import Trainer
from fan_track.utils.generic_utils import prepare_args
from fan_track.utils.generic_utils import TrainingType
from fan_track.config.config import GlobalConfig
from fan_track.config.config import SimnetConfig

def main():
    start_time = time.time()
    args = prepare_args()
    args.training_type = TrainingType.Simnet
    args.epochs = 500

    if not os.path.exists(AssocnetConfig.SIMNET_CKPT_DIR):
        os.makedirs(AssocnetConfig.SIMNET_CKPT_DIR)

    model = assoc_model.AssocModel(args)
    batch_sampler = SimnetBatchSampler(mb_dir = GlobalConfig.SIMNET_MINIBATCH_PATH)
    batch_sampler.construct()

    # prepare mini-batches for training
    if len(glob.glob(os.path.join(batch_sampler.mb_dir,'training', '*.npy'))) == 0:
        batch_sampler.generate_and_save_mini_batch(file_mappings=batch_sampler.training,
                                                   dataset_type='training')

    # prepare the validation set
    if len(glob.glob(os.path.join(batch_sampler.mb_dir,'validation', '*.npy'))) == 0:
        batch_sampler.generate_and_save_mini_batch(file_mappings=batch_sampler.validation,
                                               dataset_type='validation')
    batch_sampler.load_mb_filenames(mb_dir = GlobalConfig.SIMNET_MINIBATCH_PATH)

    with model.graph.as_default():
        model.build_simnet()
        trainer = Trainer(model, args)
        trainer.train_simnet(batch_sampler)

    elapsed_time = time.time() - start_time
    print('Time taken:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

if __name__ == '__main__':
    main()
