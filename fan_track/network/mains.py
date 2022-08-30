import argparse
import model as assoc_model
from model import TrainingType
from train import Trainer
from fan_track.utils.generic_utils import prepare_args

def main():
    args = prepare_args()
    # create an instance of the model
    model = assoc_model.AssocModel(args)
    
    # create trainer and pass all the previous components to it
    trainer = Trainer(model, args)

    with model.graph.as_default():

        model.build_simnet()
        
        trainer.run_simnet()
        
    model.reset_graph()
    
    model.args.training_type = TrainingType.Assocnet
        
    with model.graph.as_default():
   
        model.build_assoc_model()

        trainer.train_assocnet()
        

if __name__ == '__main__':
    main()
