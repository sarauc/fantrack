
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import argparse
import fan_track.network.model as assoc_model
from fan_track.network.model import TrainingType
from fan_track.network.train import Trainer
from fan_track.utils.generic_utils import prepare_args
import os
from fan_track.config.config import AssocnetConfig


def main():

    if not os.path.exists(AssocnetConfig.ASSOCNET_CKPT_DIR):
        os.makedirs(AssocnetConfig.ASSOCNET_CKPT_DIR)

    args = prepare_args()

    args.training_type = TrainingType.SimnetRun

    # create an instance of the model
    model = assoc_model.AssocModel(args)

    # create trainer and pass all the previous components to it
    trainer = Trainer(model, args)

    with model.graph.as_default():
        model.build_simnet()

        trainer.run_simnet()

    args.training_type = TrainingType.Assocnet

    # create an instance of the model
    model = assoc_model.AssocModel(args)

    # create trainer and pass all the previous components to it
    trainer = Trainer(model, args)

    with model.graph.as_default():
        model.build_assoc_model()

        trainer.train_assocnet()


if __name__ == '__main__':
    main()
