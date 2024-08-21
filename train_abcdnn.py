# updated 10/25 by Daniel Li
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
from json import loads as load_json
from json import dumps as write_json
from argparse import ArgumentParser

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# import custom methods
import config
import abcdnn

parser = ArgumentParser()
parser.add_argument( "-s", "--source", required = True )
parser.add_argument( "-t", "--target", required = True )
parser.add_argument( "--hpo", action = "store_true" )
parser.add_argument( "--randomize", action = "store_true" )
parser.add_argument( "--verbose", action = "store_true" )
parser.add_argument( "--closure", action = "store_true" )
parser.add_argument( "-p", "--postfix", default = "best_model", help = "Name of model saved to Results directory" )
parser.add_argument( "-d", "--disc_tag", default = "ABCDnn", help = "Postfix appended to original branch names of transformed variables" )
args = parser.parse_args()

if args.randomize:
  seed_rng = np.random.randint( 10000 )
else:
  seed_rng = config.params[ "MODEL" ][ "SEED" ]

hp = { key: config.params["MODEL"][key] for key in config.params[ "MODEL" ]  }

print( "[START] Training ABCDnn model {} iwth discriminator: {}".format( args.postfix, args.disc_tag ) )
if config.params[ "MODEL" ][ "RETRAIN" ]:
  print( "[INFO] Deleting existing model with name: {}".format( args.postfix ) )
  os.system( "rm -rvf Results/{}*".format( args.postfix ) )
if args.hpo:
  print( "  [INFO] Running on optimized parameters" )
  with open( os.path.join( config.results_path, "opt_params.json" ), "r" ) as jsf:
    hpo_cfg = load_json( jsf.read() )
    for key in hpo_cfg["PARAMS"]:
      hp[key] = hpo_cfg["PARAMS"][key]
else:
  print( "  [INFO] Running on fixed parameters" )
for key in hp: print( "   - {}: {}".format( key, hp[key] ) )
               
abcdnn_ = abcdnn.ABCDnn_training()
abcdnn_.setup_events(
  rSource = args.source, 
  rTarget = args.target,
  selection = config.selection,
  variables = config.variables,
  regions = config.regions,
  closure = args.closure
)

abcdnn_.setup_model(
  nodes_cond = hp[ "NODES_COND" ],
  hidden_cond = hp[ "HIDDEN_COND" ],
  nodes_trans = hp[ "NODES_TRANS" ],
  lr = hp[ "LRATE" ],
  decay = hp[ "DECAY" ],
  gap = hp[ "GAP" ],
  depth = hp[ "DEPTH" ],
  regularizer = hp[ "REGULARIZER" ],
  initializer = hp[ "INITIALIZER" ],
  activation = hp[ "ACTIVATION" ],
  beta1 = hp[ "BETA1" ],
  beta2 = hp[ "BETA2" ],
  mmd_sigmas = hp[ "MMD SIGMAS" ],
  mmd_weights = hp[ "MMD WEIGHTS" ],
  minibatch = config.params[ "MODEL" ][ "MINIBATCH" ],
  savedir = config.params[ "MODEL" ][ "SAVEDIR" ],
  disc_tag = args.disc_tag,
  seed = seed_rng,
  verbose = config.params[ "MODEL" ][ "VERBOSE" ],
  model_tag = args.postfix,
  closure = config.params[ "MODEL" ][ "CLOSURE" ],
  permute = config.params[ "MODEL" ][ "PERMUTE" ],
  retrain = config.params[ "MODEL" ][ "RETRAIN" ]
)

abcdnn_.train(
  steps = config.params[ "TRAIN" ][ "EPOCHS" ],
  patience = config.params[ "TRAIN" ][ "PATIENCE" ],
  monitor = config.params[ "TRAIN" ][ "MONITOR" ],
  display_loss = config.params[ "TRAIN" ][ "SHOWLOSS" ],
  early_stopping = config.params[ "TRAIN" ][ "EARLY STOP" ],
  monitor_threshold = config.params[ "TRAIN" ][ "MONITOR THRESHOLD" ],
  periodic_save = config.params[ "TRAIN" ][ "PERIODIC SAVE" ],
)

#abcdnn_.evaluate_regions()
#abcdnn_.extended_ABCD()
abcdnn_.save_hyperparameters()
