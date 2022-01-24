# updated 10/23 by Daniel Li
import os
from scipy import stats
import numpy as np
import json
import tensorflow as tf
from argparse import ArgumentParser
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

# import custom methods
import config
import abcdnn

parser = ArgumentParser()
parser.add_argument( "-s", "--source", default = "", required = False )
parser.add_argument( "-t", "--target", default = "", required = False )
parser.add_argument( "-v", "--verbose", action = "store_true" )
args = parser.parse_args()

if args.source != "": config.params[ "EVENTS" ][ "SOURCE" ] = os.path.join( config.data_path, args.source )
if args.target != "": config.params[ "EVENTS" ][ "TARGET" ] = os.path.join( config.data_path, args.target )

#nTrans = len( [ var for var in config.variables if config.variables[ var ][ "TRANSFORM" ] == True ] )

if not os.path.exists( "Results" ): os.mkdir( "Results" )
logfile = open( os.path.join( config.results_path, "hpo_log.txt" ), "w" )
logfile.write( "{:<10}, {:<11}, {:<11}, {:<7}, {:<5}, {:<7}, {:<7}, {:<11}, {:<10}, {:<7}, {:<7}, {:<7}\n".format(
  "NODES_COND", "HIDDEN_COND", "NODES_TRANS", "LRATE", "DECAY", "GAP", "DEPTH", "REGULARIZER", "ACTIVATION", "BETA1", "BETA2", "METRIC"
  )
)

space = []
for hp in config.hyper[ "OPTIMIZE" ]:
  if config.hyper[ "OPTIMIZE" ][ hp ][ 1 ] == "CAT": 
    space.append( Categorical( config.hyper[ "OPTIMIZE" ][ hp ][0], name = str(hp) ) )
  elif config.hyper[ "OPTIMIZE" ][ hp ][ 1 ] == "INT":
    space.append( Integer( config.hyper[ "OPTIMIZE" ][ hp ][0][0], config.hyper[ "OPTIMIZE" ][ hp ][0][1], name = str(hp) ) )
  elif config.hyper[ "OPTIMIZE" ][ hp ][ 1 ] == "REAL":
    space.append( Real( config.hyper[ "OPTIMIZE" ][ hp ][0][0], config.hyper[ "OPTIMIZE" ][ hp ][0][1], name = str(hp) ) )

min_opt_metric = 1e10
min_loss = 1e10

@use_named_args(space)
def objective(**X):
  print( ">> Configuration:\n{}\n".format(X) )

  abcdnn_ = abcdnn.ABCDnn_training()
  abcdnn_.setup_events( 
    rSource = config.params[ "EVENTS" ][ "SOURCE" ], 
    rTarget = config.params[ "EVENTS" ][ "TARGET" ], 
    selection = config.selection, 
    variables = config.variables, 
    regions = config.regions, 
    mc_weight = config.params[ "EVENTS" ][ "MCWEIGHT" ] 
    )
  
  vars = [ var for var in config.variables if config.variables[ var ][ "TRANSFORM" ] == True ]
  
  abcdnn_.setup_model( 
    nodes_cond = X["NODES_COND"], 
    hidden_cond = X["HIDDEN_COND"],
    nodes_trans = X["NODES_TRANS"],
    minibatch = config.hyper["PARAMS"]["MINIBATCH"],
    lr = X["LRATE"],
    decay = X["DECAY"],
    gap = X["GAP"],
    beta1 = X["BETA1"],
    beta2 = X["BETA2"],
    depth = X["DEPTH"],
    activation = X["ACTIVATION"],
    regularizer = X["REGULARIZER"],
    savedir = config.params[ "MODEL" ][ "SAVEDIR" ], 
    seed = config.params[ "MODEL" ][ "SEED" ],
    verbose = False, 
    retrain = config.params[ "MODEL" ][ "RETRAIN" ]
    )

  abcdnn_.train( 
    steps = config.hyper["PARAMS"][ "EPOCHS" ], 
    patience = config.hyper["PARAMS"][ "PATIENCE" ],
    monitor = config.params[ "TRAIN" ][ "MONITOR" ],
    early_stopping = False,
    split = config.params[ "TRAIN" ][ "SPLIT" ],
    display_loss = False, 
    save_hp = False,
    hpo = True
    )

  abcdnn_.evaluate_regions( hpo = True, verbose = True )

  D = []
  p = []
  for var in vars:
    D_, p_ = stats.ks_2samp( abcdnn_.rawdata[-1][ var ], abcdnn_.fakedata[-1][ :, vars.index( var ) ] )
    D.append( D_ )
    p.append( p_ )

  opt_metric = np.mean( D )  #abcdnn_.model.minloss.numpy()

  del abcdnn_
  tf.keras.backend.clear_session()
  print( ">> Optimization metric: {:.3f}".format( opt_metric ) )
  for i, var in enumerate( vars ):
    print( ">>   - {}: D = {:.3f}".format( var, D[i] ) )
  logfile.write(
    "{:<10}, {:<11}, {:<11}, {:<7}, {:<7}, {:<7}, {:<7}, {:<11}, {:<10}, {:<7}, {:<7}, {:<7.2e}\n".format(
      X["NODES_COND"], X["HIDDEN_COND"], X["NODES_TRANS"], X["LRATE"], X["DECAY"], X["GAP"], X["DEPTH"], X["REGULARIZER"], X["ACTIVATION"], X["BETA1"], X["BETA2"], opt_metric
    )
  )
  
  return opt_metric

opt_params = gp_minimize(
  func = objective,
  dimensions = space,
  verbose = config.hyper[ "PARAMS" ][ "VERBOSE" ],
  n_calls = config.hyper[ "PARAMS" ][ "N_CALLS" ],
  n_random_starts = config.hyper[ "PARAMS" ][ "N_RANDOM" ]
)

logfile.close()

print( ">> Writing optimized parameters to: opt_params.json" )
with open( os.path.join( config.results_path, "opt_params.json" ), "w" ) as jsf:
  json_dict = {}
  for i, param in enumerate( config.hyper[ "OPTIMIZE" ] ):
    json_dict[ param ] = opt_params.x[i]
  json_dict[ "METRIC" ] = opt_params.fun
  json_dict[ "EPOCHS" ] = config.hyper[ "PARAMS" ][ "EPOCHS" ]
  json_dict[ "PATIENCE" ] = config.hyper[ "PARAMS" ][ "PATIENCE" ]
  json_dict[ "N_CALLS" ] = config.hyper[ "PARAMS" ][ "N_CALLS" ]
  json_dict[ "N_RANDOM" ] = config.hyper[ "PARAMS" ][ "N_RANDOM" ]
    
  jsf.write( json.dumps( json_dict, indent = 2 ) )

