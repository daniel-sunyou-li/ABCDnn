'''
this script has four modes:
  1. retrieve the MMD loss in the signal region for a given network and source/target dataset(s) (default)
  2. calculate the systematic uncertainty of the model using dropout in inference
  3. calculate the non-closure uncertainty for extended ABCD in region D
  4. perform regression fit for input vs output 
this script should be run in mode 2 prior to the apply_abcdnn.py script such that the systematic uncertainties
are added as branches to the ABCDnn ntuple

last updated by daniel_li2@brown.edu on 11-05-2024
'''

import os
import numpy as np
import uproot
import tqdm
from argparse import ArgumentParser
from json import loads as load_json
from json import dumps as dump_json
from array import array
import ROOT
from scipy.optimize import curve_fit

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import config
import abcdnn

parser = ArgumentParser()
parser.add_argument( "-s", "--source", required = True )
parser.add_argument( "-t", "--target", required = True )
parser.add_argument( "-m", "--minor", required = True )
parser.add_argument( "-p", "--postfix", required = True )
parser.add_argument( "--batch", default = "10", help = "Number of batches to compute over" )
parser.add_argument( "--size", default = "1028", help = "Size of each batch for computing MMD loss" )
parser.add_argument( "-r", "--region", default = "D", help = "Region to evaluate (X,Y,A,B,C,D)" )
parser.add_argument( "--bayesian", action = "store_true", help = "Run Bayesian approximation to estimate model uncertainty" )
parser.add_argument( "--loss", action = "store_true", help = "Calculate the MMD loss" )
parser.add_argument( "--closure", action = "store_true", help = "Get the closure uncertainty for Regions B and C and add to the .json" )
parser.add_argument( "--yields", action = "store_true", help = "Get the statistical uncertainty of predicted yield and add to the .json" )
parser.add_argument( "--stats", action = "store_true", help = "Get mean and RMS of ABCDnn output and add to .json" )
parser.add_argument( "--transfer", action = "store_true", help = "Calculate transfer factors and add to .json" )
parser.add_argument( "--regression", action = "store_true", help = "Calculate a polynomial fit between input and output" )
parser.add_argument( "--verbose", action = "store_true" )
args = parser.parse_args()

def prepare_data( fSource, fTarget, fMinor, cVariables, cRegions, params ):
  print( "[START] Formatting data events into regions and performing pre-processing for ABCDnn evaluation" )
  rFile = {
    "SOURCE": uproot.open( fSource ),
    "TARGET": uproot.open( fTarget ),
    "MINOR":  uproot.open( fMinor )
  }

  rTree = {}

  for key_ in rFile:
    rTree[ key_ ] = rFile[ key_ ][ "Events" ]

  variables = [ str( key_ ) for key_ in sorted( cVariables.keys() ) if cVariables[ key_ ][ "TRANSFORM" ] ]
  variables_transform = [ str( key_ ) for key_ in sorted( cVariables.keys() ) if cVariables[ key_ ][ "TRANSFORM" ] ]
  if cRegions["Y"]["VARIABLE"] in cVariables and cRegions["X"]["VARIABLE"] in cVariables:
    variables.append( cRegions["Y"]["VARIABLE"] )
    variables.append( cRegions["X"]["VARIABLE"] )
  else:
    sys.exit( "[ERROR] Control variables not listed in config.variables, please add. Exiting..." )
  categorical = [ cVariables[ vName ][ "CATEGORICAL" ] for vName in variables ]
  lowerlimit =  [ cVariables[ vName ][ "LIMIT" ][0] for vName in variables ]
  upperlimit =  [ cVariables[ vName ][ "LIMIT" ][1] for vName in variables ]

  x_region = np.linspace( cRegions[ "X" ][ "MIN" ], cRegions[ "X" ][ "MAX" ], cRegions[ "X" ][ "MAX" ] - cRegions[ "X" ][ "MIN" ] + 1 )
  y_region = np.linspace( cRegions[ "Y" ][ "MIN" ], cRegions[ "Y" ][ "MAX" ], cRegions[ "Y" ][ "MAX" ] - cRegions[ "Y" ][ "MIN" ] + 1 )

  inputs = {}
  inputs_region = {}
  
  for key_ in rTree:
    if key_ == "MINOR":
      input_variables = variables + [ "xsecWeight" ]
    else:
      input_variables = variables.copy()
    inputs[ key_ ] = rTree[ key_ ].arrays( input_variables, library = "pd" )
    inputs_region[ key_ ] = { region: None for region in [ "X", "Y", "A", "B", "C", "D" ] }
    print( "[INFO] Found {} total {} entries".format( inputs[ key_ ].shape[0], key_ ) )

    inputs_region[ key_ ][ "X" ] = inputs[ key_ ].loc[ ( inputs[ key_ ][ cRegions[ "X" ][ "VARIABLE" ] ] == x_region[0] ) & ( inputs[ key_ ][ cRegions[ "Y" ][ "VARIABLE" ] ] == y_region[0] ) ]
    inputs_region[ key_ ][ "A" ] = inputs[ key_ ].loc[ ( inputs[ key_ ][ cRegions[ "X" ][ "VARIABLE" ] ] == x_region[1] ) & ( inputs[ key_ ][ cRegions[ "Y" ][ "VARIABLE" ] ] == y_region[0] ) ]

    if cRegions[ "Y" ][ "INCLUSIVE" ]:
      inputs_region[ key_ ][ "Y" ] = inputs[ key_ ].loc[ ( inputs[ key_ ][ cRegions[ "X" ][ "VARIABLE" ] ] == x_region[0] ) & ( inputs[ key_ ][ cRegions[ "Y" ][ "VARIABLE" ] ] >= y_region[1] ) ]
      inputs_region[ key_ ][ "C" ] = inputs[ key_ ].loc[ ( inputs[ key_ ][ cRegions[ "X" ][ "VARIABLE" ] ] == x_region[1] ) & ( inputs[ key_ ][ cRegions[ "Y" ][ "VARIABLE" ] ] >= y_region[1] ) ]
    else:
      inputs_region[ key_ ][ "Y" ] = inputs[ key_ ].loc[ ( inputs[ key_ ][ cRegions[ "X" ][ "VARIABLE" ] ] == x_region[0] ) & ( inputs[ key_ ][ cRegions[ "Y" ][ "VARIABLE" ] ] == y_region[1] ) ]
      inputs_region[ key_ ][ "C" ] = inputs[ key_ ].loc[ ( inputs[ key_ ][ cRegions[ "X" ][ "VARIABLE" ] ] == x_region[1] ) & ( inputs[ key_ ][ cRegions[ "Y" ][ "VARIABLE" ] ] == y_region[1] ) ]

    if cRegions[ "X" ][ "INCLUSIVE" ]:
      inputs_region[ key_ ][ "B" ] = inputs[ key_ ].loc[ ( inputs[ key_ ][ cRegions[ "X" ][ "VARIABLE" ] ] >= x_region[2] ) & ( inputs[ key_ ][ cRegions[ "Y" ][ "VARIABLE" ] ] == y_region[0] ) ]
    else:
      inputs_region[ key_ ][ "B" ] = inputs[ key_ ].loc[ ( inputs[ key_ ][ cRegions[ "X" ][ "VARIABLE" ] ] == x_region[2] ) & ( inputs[ key_ ][ cRegions[ "Y" ][ "VARIABLE" ] ] == y_region[0] ) ]

    if cRegions[ "X" ][ "INCLUSIVE" ] and cRegions[ "Y" ][ "INCLUSIVE" ]:
      inputs_region[ key_ ][ "D" ] = inputs[ key_ ].loc[ ( inputs[ key_ ][ cRegions[ "X" ][ "VARIABLE" ] ] >= x_region[2] ) & ( inputs[ key_ ][ cRegions[ "Y" ][ "VARIABLE" ] ] >= y_region[1] ) ]
    elif cRegions[ "X" ][ "INCLUSIVE" ] and not cRegions[ "Y" ][ "INCLUSIVE" ]:
      inputs_region[ key_ ][ "D" ] = inputs[ key_ ].loc[ ( inputs[ key_ ][ cRegions[ "X" ][ "VARIABLE" ] ] >= x_region[2] ) & ( inputs[ key_ ][ cRegions[ "Y" ][ "VARIABLE" ] ] == y_region[1] ) ]
    elif not cRegions[ "X" ][ "INCLUSIVE" ] and cRegions[ "Y" ][ "INCLUSIVE" ]:
      inputs_region[ key_ ][ "D" ] = inputs[ key_ ].loc[ ( inputs[ key_ ][ cRegions[ "X" ][ "VARIABLE" ] ] == x_region[2] ) & ( inputs[ key_ ][ cRegions[ "Y" ][ "VARIABLE" ] ] >= y_region[1] ) ]
    else:
      inputs_region[ key_ ][ "D" ] = inputs[ key_ ].loc[ ( inputs[ key_ ][ cRegions[ "X" ][ "VARIABLE" ] ] == x_region[2] ) & ( inputs[ key_ ][ cRegions[ "Y" ][ "VARIABLE" ] ] == y_region[1] ) ]

  print( "[INFO] Yields in each region:" )
  for region_ in [ "X", "Y", "A", "B", "C", "D" ]:
    print( "  + Region {}: Source = {}, Target = {}, Minor = {}".format( region_, inputs_region[ "SOURCE" ][ region_ ].shape[0], inputs_region[ "TARGET" ][ region_ ].shape[0], inputs_region[ "MINOR" ][ region_ ].shape[0] ) )
  print( "[DONE]\n" ) 

  print( "[START] Encoding and normalizing source inputs" )
  inputs_enc = {}
  inputs_norm = {}
  encoder = {}

  for key_ in [ "SOURCE", "TARGET" ]:
    inputs_enc[ key_ ] = {}
    inputs_norm[ key_ ] = {}

    input_mean =  np.hstack( [ float( mean ) for mean in params[ "INPUTMEANS" ] ] )
    input_sigma = np.hstack( [ float( sigma ) for sigma in params[ "INPUTSIGMAS" ] ] )

    for region_ in [ "X", "Y", "A", "B", "C", "D" ]:
      encoder[ region_ ] = abcdnn.OneHotEncoder_int( categorical, lowerlimit = lowerlimit, upperlimit = upperlimit )
      inputs_enc[ key_ ][ region_ ]  = encoder[ region_ ].encode( inputs_region[ key_ ][ region_ ].to_numpy( dtype = np.float32 ) )
      inputs_norm[ key_ ][ region_ ] = ( inputs_enc[ key_ ][ region_ ] - input_mean ) / input_sigma

  return inputs_norm[ "SOURCE" ], inputs_norm[ "TARGET" ], inputs_region
  
def prepare_model( checkpoint, params ):
  model = abcdnn.NAF(
    inputdim    = params[ "INPUTDIM" ],
    conddim     = params[ "CONDDIM" ],
    activation  = params[ "ACTIVATION" ],
    regularizer = params[ "REGULARIZER" ],
    initializer = params[ "INITIALIZER" ],
    nodes_cond  = params[ "NODES_COND" ],
    hidden_cond = params[ "HIDDEN_COND" ],
    nodes_trans = params[ "NODES_TRANS" ],
    depth       = params[ "DEPTH" ],
    permute     = True
  )
  model.load_weights( "Results/" + checkpoint )
  return model

def get_batch( X, Y, size, region ):
  Xmask = np.random.choice( np.shape( X[region] )[0], size = size, replace = False )
  Ymask = np.random.choice( np.shape( Y[region] )[0], size = size, replace = False )
  xBatch = X[region][Xmask]
  yBatch = Y[region][Ymask]
  return xBatch, yBatch

def get_loss( model, source, target, region, bSize, nBatches, bayesian = False, closure = False ):
  if closure:
    print( "[START] Evaluating MMD loss closure for {} batches of size {}".format( nBatches, bSize ) )
  else:
    print( "[START] Evaluating MMD loss for {} batches of size {}".format( nBatches, bSize ) )
  loss = []
  for i in range( nBatches ):
    sBatch, tBatch = get_batch( source, target, bSize, region )
    sPred = model( sBatch.astype( "float32" ), training = bayesian ) # applying prediction when getting the loss
    loss_i = abcdnn.mix_rbf_mmd2( tf.constant( sPred[:,0], shape = [ bSize, 1 ] ), tf.constant( tBatch[:,0].astype( "float32" ), shape = [ bSize, 1 ] ), sigmas = config.params[ "MODEL" ][ "MMD SIGMAS" ], wts = config.params[ "MODEL" ][ "MMD WEIGHTS" ] )
    loss.append(loss_i)
    print("  + Batch {} loss: {:.4f}".format(i,loss_i))

  print("[INFO] Average loss across {} batches: {:.4f} pm {:.4f}".format(nBatches,np.mean(loss),np.std(loss)))

  return np.mean(loss), np.std(loss)

def extended_ABCD( X, Y, A, B, C ):
  yield_pred = float( B * X * C**2 / ( A**2 * Y ) )
  stat_err = np.sqrt( yield_pred )
  syst_err = np.sqrt( 1. / B + 1. / X + 1. / Y + 4. / C + 4. / A ) * yield_pred

  print( "[INFO] Predicted Yield in Signal Region D: {:.2f} pm {:.2f} (stat) pm {:.2f} (syst)".format( yield_pred, stat_err, syst_err ) )
  return yield_pred, stat_err, syst_err 

def get_transfer( model, source, target, minor ):
  sf_source = 1. / ( float( args.source.split( "p" )[-1].split( ".root" )[0] ) / 100. )
  sf_target = 1. / ( float( args.target.split( "p" )[-1].split( ".root" )[0] ) / 100. )

  dX, dY, dA, dB, dC, dD = target[ "X" ].shape[0], target[ "Y" ].shape[0], target[ "A" ].shape[0], target[ "B" ].shape[0], target[ "C" ].shape[0], target[ "D" ].shape[0]
  mX, mY, mA, mB, mC, mD = np.sum( minor[ "X" ][ "xsecWeight" ] ), np.sum( minor[ "Y" ][ "xsecWeight" ] ), np.sum( minor[ "A" ][ "xsecWeight" ] ), np.sum( minor[ "B" ][ "xsecWeight" ] ), np.sum( minor[ "C" ][ "xsecWeight" ] ), np.sum( minor[ "D" ][ "xsecWeight" ] )

  cX = dX * sf_target - mX 
  cY = dY * sf_target - mY 
  cA = dA * sf_target - mA 
  cB = dB * sf_target - mB 
  cC = dC * sf_target - mC 

  mcD = source[ "D" ].shape[0] * sf_source

  pD, _, _ = extended_ABCD( cX, cY, cA, cB, cC )

  transfer_factor = float( pD ) / mcD

  with open( os.path.join( "Results/", args.postfix + ".json" ), "r+" ) as f:
    params = load_json( f.read() )
  params.update( { "TRANSFER": transfer_factor } )
  with open( "Results/{}.json".format( args.postfix ), "w" ) as f:
    f.write( dump_json( params, indent = 2 ) ) 

  print( "[INFO] Weighted minor background count: {:.2f}".format( mD  ) )
  print( "[INFO] Calculated a transfer factor of {:.3f}".format( transfer_factor ) )

  return transfer_factor

def regression_fit( model, source_norm, source, variables, tag ):
  variable_transform = ""
  with open( os.path.join( "Results/", tag+".json" ), "r+" ) as f:
    params = load_json( f.read() )
  means = np.hstack( [ float(mean) for mean in params["INPUTMEANS"] ] )
  sigmas = np.hstack( [ float(sigma) for sigma in params["INPUTSIGMAS"] ] )
  sPred = model( source_norm["D"] ) * sigmas[0] + means[0]
  for variable in variables:
    if variables[variable]["TRANSFORM"]:
      variable_transform = variable
  fit_params = np.polyfit(source["D"][variable_transform][::50],sPred[::50],2)
  print(fit_params)
  with open( os.path.join( "Results/", args.postfix+".json" ), "r+" ) as f:
    params = load_json( f.read() )
  params.update( { "FIT PARAMS": fit_params.tolist() } )
  with open( "Results/{}.json".format( args.postfix ), "w" ) as f:
    f.write( dump_json( params, indent = 2 ) )


def get_stats( model, source, region, bSize, tag ):
  sBatch, tbatch = get_batch( source, source, bSize, region ) 
  sPred = model( sBatch.astype( "float32" ) )
  with open( os.path.join( "Results/", tag + ".json" ), "r+" ) as f:
    params = load_json( f.read() )
  means = np.hstack( [ float( mean ) for mean in params[ "INPUTMEANS" ] ] )
  sigmas = np.hstack( [ float( sigma ) for sigma in params[ "INPUTSIGMAS" ] ] )
  mean_pred = np.mean( sPred * sigmas[0:2] + means[0:2], axis = 0 )
  std_pred = np.std( sPred * sigmas[0:2] + means[0:2], axis = 0 )
  params.update( { "SIGNAL_MEAN": [ str(mean) for mean in mean_pred ] } )
  params.update( { "SIGNAL_SIGMA":  [ str(std) for std in std_pred ] } )
  with open( "Results/{}.json".format( tag ), "w" ) as f:
    f.write( dump_json( params, indent = 2 ) ) 
  print( "[INFO] Mean of {} in region {}: {}".format( tag, region, mean_pred ) )
  print( "[INFO] RMS of {} in region {}: {}".format( tag, region, std_pred ) )

def main():
  print( "[START] Evaluating model {} on {} batches of size {}".format( args.postfix, args.batch, args.size ) )
  if args.bayesian: print( "[OPTION] Running with dropout on inference for Bayesian Approximation of model uncertainty" )
  with open( os.path.join( "Results/", args.postfix + ".json" ), "r" ) as f:
    params = load_json( f.read() )
  source_data, target_data, all_data = prepare_data( args.source, args.target, args.minor, config.variables, config.regions, params )
  NAF = prepare_model( args.postfix, params )
  if args.loss:
    mean, std = get_loss( NAF, source_data, target_data, args.region, int( args.size ), int( args.batch ), args.bayesian, False )
    print( "[DONE] MMD Loss in Region {}: {:.5f} pm {:.5f}".format( args.region, mean, std ) )
  if args.yields: 
    _, _, _ = extended_ABCD( target_data )
  if args.stats:
    get_stats( NAF, source_data, args.region, int( args.size ), args.postfix )
  if args.transfer:
    get_transfer( NAF, all_data[ "SOURCE" ], all_data[ "TARGET" ], all_data[ "MINOR" ] ) 
  if args.regression:
    regression_fit( NAF, source_data, all_data[ "SOURCE" ], config.variables, args.postfix ) 

main()
