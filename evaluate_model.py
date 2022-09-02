# this script has two modes:
#   1. retrieve the MMD loss in the signal region for a given network and source/target dataset(s) (default)
#   2. calculate the systematic uncertainty of the model using dropout in inference
#   3. calculate the non-closure uncertainty for extended ABCD in region D
# this script should be run in mode 2 prior to the apply_abcdnn.py script such that the systematic uncertainties
# are added as branches to the ABCDnn ntuple
#
# last updated by daniel_li2@brown.edu on 08-17-2022

import os
import numpy as np
import uproot
import tqdm
from argparse import ArgumentParser
from json import loads as load_json
from array import array
import ROOT

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import config
import abcdnn

parser = ArgumentParser()
parser.add_argument( "-s", "--source", required = True )
parser.add_argument( "-t", "--target", required = True )
parser.add_argument( "-m", "--tag", required = True )
parser.add_argument( "-b", "--batch", default = "10", help = "Number of batches to compute over" )
parser.add_argument( "-n", "--size", default = "1028", help = "Size of each batch for computing MMD loss" )
parser.add_argument( "-r", "--region", default = "D", help = "Region to evaluate (X,Y,A,B,C,D)" )
parser.add_argument( "--bayesian", action = "store_true", help = "Run Bayesian approximation to estimate model uncertainty" )
parser.add_argument( "--closure", action = "store_true", help = "Get the closure uncertainty (i.e. percent difference between predicted and true yield)" )
parser.add_argument( "--stat", action = "store_true", help = "Get the statistical uncertainty of predicted yield" )
parser.add_argument( "--verbose", action = "store_true" )
args = parser.parse_args()

def prepare_data( fSource, fTarget, cVariables, cRegions, params ):
  sFile = uproot.open( fSource )
  tFile = uproot.open( fTarget )
  sTree = sFile[ "Events" ]
  tTree = tFile[ "Events" ]
  
  variables = [ str( key ) for key in list( cVariables.keys() ) if cVariables[key]["TRANSFORM"] ]
  variables_transform = [ str( key ) for key in list( cVariables.keys() ) if cVariables[key]["TRANSFORM"] ]
  if cRegions["Y"]["VARIABLE"] in cVariables and cRegions["X"]["VARIABLE"] in cVariables:
    variables.append( cRegions["Y"]["VARIABLE"] )
    variables.append( cRegions["X"]["VARIABLE"] )
  else:
    sys.exit( "[ERROR] Control variables not listed in config.variables, please add. Exiting..." )
  categorical = [ cVariables[ vName ][ "CATEGORICAL" ] for vName in variables ]
  lowerlimit =  [ cVariables[ vName ][ "LIMIT" ][0] for vName in variables ]
  upperlimit =  [ cVariables[ vName ][ "LIMIT" ][1] for vName in variables ]
  
  inputs_src = sTree.pandas.df( variables )
  x_region = np.linspace( cRegions[ "X" ][ "MIN" ], cRegions[ "X" ][ "MAX" ], cRegions[ "X" ][ "MAX" ] - cRegions[ "X" ][ "MIN" ] + 1 )
  y_region = np.linspace( cRegions[ "Y" ][ "MIN" ], cRegions[ "Y" ][ "MAX" ], cRegions[ "Y" ][ "MAX" ] - cRegions[ "Y" ][ "MIN" ] + 1 )
  inputs_src_region = { region: None for region in [ "X", "Y", "A", "B", "C", "D" ] }
  print( ">> Found {} total source entries".format( inputs_src.shape[0] ) )
  inputs_tgt = tTree.pandas.df( variables ) 
  inputs_tgt_region = { region: None for region in [ "X", "Y", "A", "B", "C", "D" ] }
  print( ">> Found {} total target entries".format( inputs_tgt.shape[0] ) )

  inputs_src_region["X"] = inputs_src.loc[ ( inputs_src[ cRegions["X"]["VARIABLE"] ] == x_region[0] ) & ( inputs_src[ cRegions["Y"]["VARIABLE"] ] == y_region[0] ) ]
  inputs_tgt_region["X"] = inputs_tgt.loc[ ( inputs_tgt[ cRegions["X"]["VARIABLE"] ] == x_region[0] ) & ( inputs_tgt[ cRegions["Y"]["VARIABLE"] ] == y_region[0] ) ]
  inputs_src_region["A"] = inputs_src.loc[ ( inputs_src[ cRegions["X"]["VARIABLE"] ] == x_region[1] ) & ( inputs_src[ cRegions["Y"]["VARIABLE"] ] == y_region[0] ) ]
  inputs_tgt_region["A"] = inputs_tgt.loc[ ( inputs_tgt[ cRegions["X"]["VARIABLE"] ] == x_region[1] ) & ( inputs_tgt[ cRegions["Y"]["VARIABLE"] ] == y_region[0] ) ]

  if cRegions["Y"]["INCLUSIVE"]:
    inputs_src_region["Y"] = inputs_src.loc[ ( inputs_src[ cRegions["X"]["VARIABLE"] ] == x_region[0] ) & ( inputs_src[ cRegions["Y"]["VARIABLE"] ] >= y_region[1] ) ]
    inputs_tgt_region["Y"] = inputs_tgt.loc[ ( inputs_tgt[ cRegions["X"]["VARIABLE"] ] == x_region[0] ) & ( inputs_tgt[ cRegions["Y"]["VARIABLE"] ] >= y_region[1] ) ]
    inputs_src_region["C"] = inputs_src.loc[ ( inputs_src[ cRegions["X"]["VARIABLE"] ] == x_region[1] ) & ( inputs_src[ cRegions["Y"]["VARIABLE"] ] >= y_region[1] ) ]
    inputs_tgt_region["C"] = inputs_tgt.loc[ ( inputs_tgt[ cRegions["X"]["VARIABLE"] ] == x_region[1] ) & ( inputs_tgt[ cRegions["Y"]["VARIABLE"] ] >= y_region[1] ) ]
  else:
    inputs_src_region["Y"] = inputs_src.loc[ ( inputs_src[ cRegions["X"]["VARIABLE"] ] == x_region[0] ) & ( inputs_src[ cRegions["Y"]["VARIABLE"] ] == y_region[1] ) ]
    inputs_tgt_region["Y"] = inputs_tgt.loc[ ( inputs_tgt[ cRegions["X"]["VARIABLE"] ] == x_region[0] ) & ( inputs_tgt[ cRegions["Y"]["VARIABLE"] ] == y_region[1] ) ]
    inputs_src_region["C"] = inputs_src.loc[ ( inputs_src[ cRegions["X"]["VARIABLE"] ] == x_region[1] ) & ( inputs_src[ cRegions["Y"]["VARIABLE"] ] == y_region[1] ) ]
    inputs_tgt_region["C"] = inputs_tgt.loc[ ( inputs_tgt[ cRegions["X"]["VARIABLE"] ] == x_region[1] ) & ( inputs_tgt[ cRegions["Y"]["VARIABLE"] ] == y_region[1] ) ]

  if cRegions["X"]["INCLUSIVE"]:
    inputs_src_region["B"] = inputs_src.loc[ ( inputs_src[ cRegions["X"]["VARIABLE"] ] >= x_region[2] ) & ( inputs_src[ cRegions["Y"]["VARIABLE"] ] == y_region[0] ) ]
    inputs_tgt_region["B"] = inputs_tgt.loc[ ( inputs_tgt[ cRegions["X"]["VARIABLE"] ] >= x_region[2] ) & ( inputs_tgt[ cRegions["Y"]["VARIABLE"] ] == y_region[0] ) ]
  else:
    inputs_src_region["B"] = inputs_src.loc[ ( inputs_src[ cRegions["X"]["VARIABLE"] ] == x_region[2] ) & ( inputs_src[ cRegions["Y"]["VARIABLE"] ] == y_region[0] ) ]
    inputs_tgt_region["B"] = inputs_tgt.loc[ ( inputs_tgt[ cRegions["X"]["VARIABLE"] ] == x_region[2] ) & ( inputs_tgt[ cRegions["Y"]["VARIABLE"] ] == y_region[0] ) ]

  if cRegions["X"]["INCLUSIVE"] and cRegions["Y"]["INCLUSIVE"]:
    inputs_src_region["D"] = inputs_src.loc[ ( inputs_src[ cRegions["X"]["VARIABLE"] ] >= x_region[2] ) & ( inputs_src[ cRegions["Y"]["VARIABLE"] ] >= y_region[1] ) ]
    inputs_tgt_region["D"] = inputs_tgt.loc[ ( inputs_tgt[ cRegions["X"]["VARIABLE"] ] >= x_region[2] ) & ( inputs_tgt[ cRegions["Y"]["VARIABLE"] ] >= y_region[1] ) ]
  elif config.regions["X"]["INCLUSIVE"] and not config.regions["Y"]["INCLUSIVE"]:
    inputs_src_region["D"] = inputs_src.loc[ ( inputs_src[ cRegions["X"]["VARIABLE"] ] >= x_region[2] ) & ( inputs_src[ cRegions["Y"]["VARIABLE"] ] == y_region[1] ) ]
    inputs_tgt_region["D"] = inputs_tgt.loc[ ( inputs_tgt[ cRegions["X"]["VARIABLE"] ] >= x_region[2] ) & ( inputs_tgt[ cRegions["Y"]["VARIABLE"] ] == y_region[1] ) ]
  elif not config.regions["X"]["INCLUSIVE"] and config.regions["Y"]["INCLUSIVE"]:
    inputs_src_region["D"] = inputs_src.loc[ ( inputs_src[ cRegions["X"]["VARIABLE"] ] == x_region[2] ) & ( inputs_src[ cRegions["Y"]["VARIABLE"] ] >= y_region[1] ) ]
    inputs_tgt_region["D"] = inputs_tgt.loc[ ( inputs_tgt[ cRegions["X"]["VARIABLE"] ] == x_region[2] ) & ( inputs_tgt[ cRegions["Y"]["VARIABLE"] ] >= y_region[1] ) ]
  else:
    inputs_src_region["D"] = inputs_src.loc[ ( inputs_src[ cRegions["X"]["VARIABLE"] ] == x_region[2] ) & ( inputs_src[ cRegions["Y"]["VARIABLE"] ] == y_region[1] ) ]
    inputs_tgt_region["D"] = inputs_tgt.loc[ ( inputs_tgt[ cRegions["X"]["VARIABLE"] ] == x_region[2] ) & ( inputs_tgt[ cRegions["Y"]["VARIABLE"] ] == y_region[1] ) ]

  print( ">> Yields in each region:" )
  for region in inputs_src_region:
    print( "  + Region {}: Source = {}, Target = {}".format( region, inputs_src_region[region].shape[0], inputs_tgt_region[region].shape[0] ) )

  print( ">> Encoding and normalizing source inputs" )
  source_enc_region = {}
  target_enc_region = {}
  encoder = {}
  source_nrm_region = {}
  target_nrm_region = {}
  inputmeans = np.hstack( [ float( mean ) for mean in params[ "INPUTMEANS" ] ] )
  inputsigmas = np.hstack( [ float( sigma ) for sigma in params[ "INPUTSIGMAS" ] ] )
  for region in inputs_src_region:
    encoder[region] = abcdnn.OneHotEncoder_int( categorical, lowerlimit = lowerlimit, upperlimit = upperlimit )
    source_enc_region[ region ] = encoder[region].encode( inputs_src_region[ region ].to_numpy( dtype = np.float32 ) )
    target_enc_region[ region ] = encoder[region].encode( inputs_tgt_region[ region ].to_numpy( dtype = np.float32 ) )
    source_nrm_region[ region ] = ( source_enc_region[ region ] - inputmeans ) / inputsigmas
    target_nrm_region[ region ] = ( target_enc_region[ region ] - inputmeans ) / inputsigmas
    
  return source_nrm_region, target_nrm_region
  
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
  loss_closure = { "A": [], "B": [] }
  for i in range( nBatches ):
    sBatch, tBatch = get_batch( source, target, bSize, region )
    sPred = model( sBatch.astype( "float32" ), training = bayesian ) # applying prediction when getting the loss
    if closure:
      loss_A = abcdnn.mix_rbf_mmd2( tf.constant( sPred[:,0], shape = [ bSize, 1 ] ), tf.constant( tBatch[:,0].astype( "float32" ), shape = [ bSize, 1 ] ), sigmas = config.params[ "MODEL" ][ "MMD SIGMAS" ], wts = config.params[ "MODEL" ][ "MMD WEIGHTS" ] )
      loss_B = abcdnn.mix_rbf_mmd2( tf.constant( sPred[:,1], shape = [ bSize, 1 ] ), tf.constant( tBatch[:,1].astype( "float32" ), shape = [ bSize, 1 ] ), sigmas = config.params[ "MODEL" ][ "MMD SIGMAS" ], wts = config.params[ "MODEL" ][ "MMD WEIGHTS" ] )
      if args.verbose: print( "  Batch {:<4}: Loss A = {:.4f}, Loss B = {:.4}".format( str( i + 1 ) + ".", loss_A, loss_B ) )
      loss_closure[ "A" ].append( loss_A )
      loss_closure[ "B" ].append( loss_B )
    iLoss = abcdnn.mix_rbf_mmd2( sPred, tBatch[:,:2].astype( "float32" ), sigmas = config.params[ "MODEL" ][ "MMD SIGMAS" ], wts = config.params[ "MODEL" ][ "MMD WEIGHTS" ] )
    if args.verbose: print( "  Batch {:<4}: {:.4f}".format( str( i + 1 ) + ".", iLoss ) )
    loss.append( iLoss )
  lMean = np.mean( loss )
  lStd = np.std( loss )
  if closure: 
    lMean_A = np.mean( loss_closure[ "A" ] )
    lMean_B = np.mean( loss_closure[ "B" ] )
    lStd_A = np.std( loss_closure[ "A" ] )
    lStd_B = np.std( loss_closure[ "B" ] )
    print( "[ABCDNN] Closure: {:.4f}%".format( abs( 100. * ( lMean_A - lMean_B ) / lMean ) ) )
    print( "  + Combined Loss: {:.4f} pm {:.4f}".format( lMean, lStd ) )
    print( "  + Loss A: {:.4f} pm {:.4f}".format( lMean_A, lStd_A ) )
    print( "  + Loss B: {:.4f} pm {:.4f}".format( lMean_B, lStd_B ) )
    return lMean_A, lMean_B, lStd_A, lStd_B
  else:
    return lMean, lStd

def extended_ABCD( target ):
  count = {}
  for region in target:
    count[ region ] = len( target[region] )
  pYield = float( count["B"] * count["X"] * count["C"]**2 / ( count["A"]**2 * count["Y"] ) )
  pStat = np.sqrt( pYield )
  pSyst = np.sqrt( 1./count["B"] + 1./count["X"] + 1./count["Y"] + 4./count["C"] + 4./count["A"] ) * pYield
  print( "[Extended ABCD] Predicted Yield in Signal Region D: {:.2f} pm {:.2f} (stat) pm {:.2f} (syst)".format( pYield, pStat, pSyst ) )
  return pYield, pStat, pSyst

def non_closure_eABCD( source_data, target_data ):
  pYield, _, _ = extended_ABCD( target_data )
  oYield = len( target_data["D"] )
  syst = 100. * abs( ( pYield - oYield ) / oYield )
  print( "[Non-Closure ABCD] Observed: {}, Expected: {:.2f}, % Difference: {:.2f}".format( oYield, pYield, syst ) )
  return syst

def main():
  print( "[START] Evaluating model {} on {} batches of size {}".format( args.tag, args.batch, args.size ) )
  if args.bayesian: print( "[OPTION] Running with dropout on inference for Bayesian Approximation of model uncertainty" )
  with open( os.path.join( "Results/", args.tag + ".json" ), "r" ) as f:
    params = load_json( f.read() )
  source_data, target_data = prepare_data( args.source, args.target, config.variables, config.regions, params )
  NAF = prepare_model( args.tag, params )
  mean, std = get_loss( NAF, source_data, target_data, args.region, int( args.size ), int( args.batch ), args.bayesian, False )
  if args.closure:
    _, _, _, _ = get_loss( NAF, source_data, target_data, args.region, int( args.size ), int( args.batch ), args.bayesian, True )
    _ = non_closure_eABCD( source_data, target_data )
  if args.stat: 
    _, _, _ = extended_ABCD( target_data )
  print( "[DONE] MMD Loss in Region {}: {:.5f} pm {:.5f}".format( args.region, mean, std ) )

main()
