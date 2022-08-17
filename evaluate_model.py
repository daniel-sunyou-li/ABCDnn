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
parser.add_argument( "-s", "--size", default = "1028", help = "Size of each batch for computing MMD loss" )
parser.add_argument( "-r", "--region", default = "D", help = "Region to evaluate (X,Y,A,B,C,D)" )
parser.add_argument( "--bayesian", action = "store_true", help = "Run Bayesian approximation to estimate model uncertainty" )
parser.add_argument( "--closure", action = "store_true", help = "Get the closure uncertainty (i.e. % difference between predicted and true yield)" )
parser.add_argument( "--stat", action = "store_true", help = "Get the statistical uncertainty of predicted yield" )

args = parser.parse_args()

def prepare_data():
  sFile = uproot.open( args.source )
  tFile = uproot.open( args.target )
  sTree = sFile[ "Events" ]
  tTree = tFile[ "Events" ]
  
  variables = [ str( key ) for key in config.variables.keys() if config.variables[key]["TRANSFORM"] ]
  variables_transform = [ str( key ) for key in config.variables.keys() if config.variables[key]["TRANSFORM"] ]
  if config.regions["Y"]["VARIABLE"] in config.variables and config.regions["X"]["VARIABLE"] in config.variables:
    variables.append( config.regions["Y"]["VARIABLE"] )
    variables.append( config.regions["X"]["VARIABLE"] )
  else:
    sys.exit( "[ERROR] Control variables not listed in config.variables, please add. Exiting..." )
  categorical = [ config.variables[ vName ][ "CATEGORICAL" ] for vName in variables ]
  lowerlimit = [ config.variables[ vName ][ "LIMIT" ][0] for vName in variables ]
  upperlimit = [ config.variables[ vName ][ "LIMIT" ][1] for vName in variables ]
  
  inputs_src = sTree.pandas.df( variables )
  x_region = np.linspace( config.regions[ "X" ][ "MIN" ], config.regions[ "X" ][ "MAX" ], config.regions[ "X" ][ "MAX" ] - config.regions[ "X" ][ "MIN" ] + 1 )
  y_region = np.linspace( config.regions[ "Y" ][ "MIN" ], config.regions[ "Y" ][ "MAX" ], config.regions[ "Y" ][ "MAX" ] - config.regions[ "Y" ][ "MIN" ] + 1 )
  inputs_src_region = { region: None for region in [ "X", "Y", "A", "B", "C", "D" ] }
  print( ">> Found {} total source entries".format( inputs_src.shape[0] ) )
  inputs_tgt = tTree.pandas.df( variables ) 
  inputs_tgt_region = { region: None for region in [ "X", "Y", "A", "B", "C", "D" ] }
  print( ">> Found {} total target entries".format( inputs_tgt.shape[0] ) )

  inputs_src_region["X"] = inputs_src.loc[ ( inputs_src[ config.regions["X"]["VARIABLE"] ] == x_region[0] ) & ( inputs_src[ config.regions["Y"]["VARIABLE"] ] == y_region[0] ) ]
  inputs_tgt_region["X"] = inputs_tgt.loc[ ( inputs_tgt[ config.regions["X"]["VARIABLE"] ] == x_region[0] ) & ( inputs_tgt[ config.regions["Y"]["VARIABLE"] ] == y_region[0] ) ]
  inputs_src_region["A"] = inputs_src.loc[ ( inputs_src[ config.regions["X"]["VARIABLE"] ] == x_region[1] ) & ( inputs_src[ config.regions["Y"]["VARIABLE"] ] == y_region[0] ) ]
  inputs_tgt_region["A"] = inputs_tgt.loc[ ( inputs_tgt[ config.regions["X"]["VARIABLE"] ] == x_region[1] ) & ( inputs_tgt[ config.regions["Y"]["VARIABLE"] ] == y_region[0] ) ]

  if config.regions["Y"]["INCLUSIVE"]:
    inputs_src_region["Y"] = inputs_src.loc[ ( inputs_src[ config.regions["X"]["VARIABLE"] ] == x_region[0] ) & ( inputs_src[ config.regions["Y"]["VARIABLE"] ] >= y_region[1] ) ]
    inputs_tgt_region["Y"] = inputs_tgt.loc[ ( inputs_tgt[ config.regions["X"]["VARIABLE"] ] == x_region[0] ) & ( inputs_tgt[ config.regions["Y"]["VARIABLE"] ] >= y_region[1] ) ]
    inputs_src_region["C"] = inputs_src.loc[ ( inputs_src[ config.regions["X"]["VARIABLE"] ] == x_region[1] ) & ( inputs_src[ config.regions["Y"]["VARIABLE"] ] >= y_region[1] ) ]
    inputs_tgt_region["C"] = inputs_tgt.loc[ ( inputs_tgt[ config.regions["X"]["VARIABLE"] ] == x_region[1] ) & ( inputs_tgt[ config.regions["Y"]["VARIABLE"] ] >= y_region[1] ) ]
  else:
    inputs_src_region["Y"] = inputs_src.loc[ ( inputs_src[ config.regions["X"]["VARIABLE"] ] == x_region[0] ) & ( inputs_src[ config.regions["Y"]["VARIABLE"] ] == y_region[1] ) ]
    inputs_tgt_region["Y"] = inputs_tgt.loc[ ( inputs_tgt[ config.regions["X"]["VARIABLE"] ] == x_region[0] ) & ( inputs_tgt[ config.regions["Y"]["VARIABLE"] ] == y_region[1] ) ]
    inputs_src_region["C"] = inputs_src.loc[ ( inputs_src[ config.regions["X"]["VARIABLE"] ] == x_region[1] ) & ( inputs_src[ config.regions["Y"]["VARIABLE"] ] == y_region[1] ) ]
    inputs_tgt_region["C"] = inputs_tgt.loc[ ( inputs_tgt[ config.regions["X"]["VARIABLE"] ] == x_region[1] ) & ( inputs_tgt[ config.regions["Y"]["VARIABLE"] ] == y_region[1] ) ]

  if config.regions["X"]["INCLUSIVE"]:
    inputs_src_region["B"] = inputs_src.loc[ ( inputs_src[ config.regions["X"]["VARIABLE"] ] >= x_region[2] ) & ( inputs_src[ config.regions["Y"]["VARIABLE"] ] == y_region[0] ) ]
    inputs_tgt_region["B"] = inputs_tgt.loc[ ( inputs_tgt[ config.regions["X"]["VARIABLE"] ] >= x_region[2] ) & ( inputs_tgt[ config.regions["Y"]["VARIABLE"] ] == y_region[0] ) ]
  else:
    inputs_src_region["B"] = inputs_src.loc[ ( inputs_src[ config.regions["X"]["VARIABLE"] ] == x_region[2] ) & ( inputs_src[ config.regions["Y"]["VARIABLE"] ] == y_region[0] ) ]
    inputs_tgt_region["B"] = inputs_tgt.loc[ ( inputs_tgt[ config.regions["X"]["VARIABLE"] ] == x_region[2] ) & ( inputs_tgt[ config.regions["Y"]["VARIABLE"] ] == y_region[0] ) ]

  if config.regions["X"]["INCLUSIVE"] and config.regions["Y"]["INCLUSIVE"]:
    inputs_src_region["D"] = inputs_src.loc[ ( inputs_src[ config.regions["X"]["VARIABLE"] ] >= x_region[2] ) & ( inputs_src[ config.regions["Y"]["VARIABLE"] ] >= y_region[1] ) ]
    inputs_tgt_region["D"] = inputs_tgt.loc[ ( inputs_tgt[ config.regions["X"]["VARIABLE"] ] >= x_region[2] ) & ( inputs_tgt[ config.regions["Y"]["VARIABLE"] ] >= y_region[1] ) ]
  elif config.regions["X"]["INCLUSIVE"] and not config.regions["Y"]["INCLUSIVE"]:
    inputs_src_region["D"] = inputs_src.loc[ ( inputs_src[ config.regions["X"]["VARIABLE"] ] >= x_region[2] ) & ( inputs_src[ config.regions["Y"]["VARIABLE"] ] == y_region[1] ) ]
    inputs_tgt_region["D"] = inputs_tgt.loc[ ( inputs_tgt[ config.regions["X"]["VARIABLE"] ] >= x_region[2] ) & ( inputs_tgt[ config.regions["Y"]["VARIABLE"] ] == y_region[1] ) ]
  elif not config.regions["X"]["INCLUSIVE"] and config.regions["Y"]["INCLUSIVE"]:
    inputs_src_region["D"] = inputs_src.loc[ ( inputs_src[ config.regions["X"]["VARIABLE"] ] == x_region[2] ) & ( inputs_src[ config.regions["Y"]["VARIABLE"] ] >= y_region[1] ) ]
    inputs_tgt_region["D"] = inputs_tgt.loc[ ( inputs_tgt[ config.regions["X"]["VARIABLE"] ] == x_region[2] ) & ( inputs_tgt[ config.regions["Y"]["VARIABLE"] ] >= y_region[1] ) ]
  else:
    inputs_src_region["D"] = inputs_src.loc[ ( inputs_src[ config.regions["X"]["VARIABLE"] ] == x_region[2] ) & ( inputs_src[ config.regions["Y"]["VARIABLE"] ] == y_region[1] ) ]
    inputs_tgt_region["D"] = inputs_tgt.loc[ ( inputs_tgt[ config.regions["X"]["VARIABLE"] ] == x_region[2] ) & ( inputs_tgt[ config.regions["Y"]["VARIABLE"] ] == y_region[1] ) ]

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
    source_enc_region[ region ] = encoder[region].encode( source_enc_region[ region ].to_numpy( dtype = np.float32 ) )
    target_enc_region[ region ] = encoder[region].encode( target_enc_region[ region ].to_numpy( dtype = np.float32 ) )
    source_nrm_region[ region ] = ( source_enc_region[ region ] - inputmeans ) / inputsigmas
    target_enc_region[ region ] = ( target_enc_region[ region ] - inputmeans ) / inputsigmas
    
  return source_nrm_region, target_nrm_region
  
def prepare_model():
  # to-do
  return model

def get_batch():
  # to-do
  return batch

def mmd_loss( sPred, tBatch ):
  # to-do
  return mmd_loss

def get_loss( model, region, nBatches, bayesian = False ):
  # to-do
  loss = []
  source, target = prepare_data( region )
  for i in range( nBatches ):
    sBatch, tBatch = get_batch( bSize, source, target )
    sPred = model( np.asarray( sBatch ), training = bayesian )
    loss.append( mmd_loss( sPred, tBatch ) )
  lMean = np.mean( loss )
  lStd = np.std( loss )
  return lMean, lStd

def extended_ABCD():
  # to-do
  return pYield

def non_closure():
  # to-do
  pYield = extended_ABCD()
  
  return 100. * abs( ( pYield - oYield ) / oYield )

def main():
  # to-do
  source_data, target_data = prepare_data()
  NAF = prepare_model()
  mean, std = get_loss( NAF, args.region, args.batch, args.bayesian )
  print( "[DONE] For model {} on {} batches of size {}:".format( args.tag, args.batch, args.size ) )
  print( "  + Mean: {:.4f}".format(  ) )
