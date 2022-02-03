# this script is run on the condor node for applying the trained ABCDnn model to ttbar samples
# last updated 11/15/2021 by Daniel Li

import numpy as np
import os
import uproot
import abcdnn
from argparse import ArgumentParser
from json import loads as load_json
from array import array

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import config

parser = ArgumentParser()
parser.add_argument( "-s", "--source", required = True )
parser.add_argument( "-c", "--checkpoints", nargs = "+", required = True )

args = parser.parse_args()

import ROOT

# load in json file

models = {}
disc_tags = {}
track_tags = []
print( ">> Loading checkpoints to NAF models:" )
for checkpoint in args.checkpoints:
  print( "  + {}".format( checkpoint ) )
  with open( checkpoint + ".json", "r" ) as f:
    params  = load_json( f.read() )
    if params[ "DISC TAG" ] in track_tags:
      print( "[WARN] Found redundant discriminator tag. Replacing {} --> {}".format( params[ "DISC TAG" ], checkpoint ) )
      disc_tags[ checkpoint ] = checkpoint
    else:
      disc_tags[ checkpoint ] = params["DISC TAG"]
    models[ checkpoint ] = abcdnn.NAF(
      inputdim    = params["INPUTDIM"],
      conddim     = params["CONDDIM"],
      activation  = params["ACTIVATION"],
      regularizer = params["REGULARIZER"],
      nodes_cond  = params["NODES_COND"],
      hidden_cond = params["HIDDEN_COND"],
      depth       = params["DEPTH"],
      permute = True
    ) 
  models[ checkpoint ].load_weights( checkpoint )

print( ">> Formatting MC sample..." )

upFile = uproot.open( args.source )
upTree = upFile[ "ljmet" ]

variables = []
v_in = []
categorical = []
lowerlimit = []
upperlimit = []
predictions = {}

for variable in sorted( config.variables.keys() ):
  if config.variables[ variable ][ "TRANSFORM" ]: v_in.append( variable )
  variables.append( variable )
  categorical.append( config.variables[ variable ][ "CATEGORICAL" ] )
  upperlimit.append( config.variables[ variable ][ "LIMIT" ][1] )
  lowerlimit.append( config.variables[ variable ][ "LIMIT" ][0] )

_onehotencoder = abcdnn.OneHotEncoder_int( categorical, lowerlimit = lowerlimit, upperlimit = upperlimit )

inputs_mc = upTree.pandas.df( variables )
x_upper = inputs_mc[ config.regions[ "X" ][ "VARIABLE" ] ].max() if config.regions[ "X" ][ "INCLUSIVE" ] else config.regions[ "X" ][ "MAX" ]
y_upper = inputs_mc[ config.regions[ "Y" ][ "VARIABLE" ] ].max() if config.regions[ "Y" ][ "INCLUSIVE" ] else config.regions[ "Y" ][ "MAX" ]
inputs_mc[ config.regions[ "X" ][ "VARIABLE" ] ].clip( config.regions[ "X" ][ "MIN" ], x_upper )
inputs_mc[ config.regions[ "Y" ][ "VARIABLE" ] ].clip( config.regions[ "Y" ][ "MIN" ], y_upper )

inputs_mc_enc = _onehotencoder.encode( inputs_mc.to_numpy( dtype = np.float32 ) )

print( ">> Applying normalization to MC inputs..." )
inputmeans = np.hstack( [ float( mean ) for mean in params[ "INPUTMEANS" ] ] )
inputsigma = np.hstack( [ float( sigma ) for sigma in params[ "INPUTSIGMAS" ] ] )
normedinputs_mc = ( inputs_mc_enc - inputmeans ) / inputsigma

print( "[START] Predicting using model checkpoints" )
for checkpoint in args.checkpoints:
  predictions[ checkpoint ] = []
  for prediction in NAF.predict( normedinputs_mc )
    predictions[ checkpoint ].append( [
      prediction[0] * inputsigma[0] + inputmeans[0],
      prediction[1] * inputsigma[1] + inputmeans[1]
    ] )


# populate the step 3
rFile_in = ROOT.TFile.Open( args.source )
rTree_in = rFile_in.Get( "ljmet" )

rFile_out = ROOT.TFile( args.source.replace( "hadd.root", "ABCDnn_hadd.root" ).split("/")[-1],  "RECREATE" )
rFile_out.cd()
rTree_out = rTree_in.CloneTree(0)

print( "[START] Adding new branches to {} ({} entries):".format( args.source.split("/")[-1], rTree_in.GetEntries() ) )
arrays = {}
branches = {}
for checkpoint in args.checkpoints:
  arrays[ checkpoint ] = {}
  branches[ checkpoint ] = {}
  for variable in v_in:
    arrays[ checkpoint ][ variable ] = array( "f", [0.] )
    branches[ checkpoint ][ variable ] = rTree_out.Branch( "{}_{}".format( str( variable ), disc_tags[ checkpoint ] ) , arrays[ checkpoint ][ variable ], "{}_{}/F".format( str( variable ), disc_tags[ checkpoint ] ) );
    print( "  + {}_{}".format( str( variable ), disc_tags[ checkpoint ] ) )    

for i in range( rTree_in.GetEntries() ): 
  rTree_in.GetEntry(i)
  for checkpoint in args.checkpoints:
    for j, variable in enumerate( v_in ):
      arrays[ checkpoint ][ variable ][0] = predictions[ checkpoint ][i][j]
      rTree_out.Fill()
  
rTree_out.Write()
rFile_out.Write()
rFile_out.Close()

