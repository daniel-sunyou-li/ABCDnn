# this script is run on the condor node for applying the trained ABCDnn model to ttbar samples
# last updated 11/15/2021 by Daniel Li

import numpy as np
import os
import uproot
import abcdnn
from argparse import ArgumentParser
from json import loads as load_json
from array import array
import ROOT

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import config

parser = ArgumentParser()
parser.add_argument( "-j", "--json", required = True )
parser.add_argument( "-s", "--source", required = True )
parser.add_argument( "-c", "--checkpoint", required = True )

args = parser.parse_args()

# load in json file
print( ">> Reading in {} for hyper parameters...".format( args.json ) )
with open( args.json, "r" ) as f:
  params = load_json( f.read() )

print( ">> Setting up NAF model..." )
NAF = abcdnn.NAF( 
  inputdim = params["INPUTDIM"],
  conddim = params["CONDDIM"],
  activation = params["ACTIVATION"], 
  regularizer = params["REGULARIZER"],
  nodes_cond = params["NODES_COND"],
  hidden_cond = params["HIDDEN_COND"],
  nodes_trans = params["NODES_TRANS"],
  depth = params["DEPTH"],
  permute = True
)

print( ">> Loading checkpoint weights from {}...".format( args.checkpoint ) )
NAF.load_weights( args.checkpoint )

print( ">> Formatting MC sample..." )

upFile = uproot.open( args.source )
upTree = upFile[ "ljmet" ]

variables = []
v_in = []
categorical = []
lowerlimit = []
upperlimit = []
for variable in sorted( list( config.variables.keys() ) ):
  if config.variables[ variable ][ "TRANSFORM" ] == True: v_in.append( variable )
  variables.append( variable )
  categorical.append( config.variables[ variable ][ "CATEGORICAL" ] )
  upperlimit.append( config.variables[ variable ][ "LIMIT" ][1] )
  lowerlimit.append( config.variables[ variable ][ "LIMIT" ][0] )

_onehotencoder = abcdnn.OneHotEncoder_int( categorical, lowerlimit = lowerlimit, upperlimit = upperlimit )

inputs_mc = upTree.pandas.df( variables )
inputs_mc_enc = _onehotencoder.encode( inputs_mc.to_numpy( dtype = np.float32 ) )

print( ">> Applying normalization to MC inputs..." )
inputmeans = np.hstack( [ float( mean ) for mean in params[ "INPUTMEANS" ] ] )
inputsigma = np.hstack( [ float( sigma ) for sigma in params[ "INPUTSIGMAS" ] ] )
normedinputs_mc = ( inputs_mc_enc - inputmeans ) / inputsigma

print( ">> Transforming MC samples..." )
predict_x = NAF.predict( normedinputs_mc )
predict_norm = []

for predict in predict_x:
  predict_norm.append( [
    predict[0] * inputsigma[0] + inputmeans[0],    
    predict[1] * inputsigma[1] + inputmeans[1]
  ] )


# populate the step 3
rFile_in = ROOT.TFile.Open( "{}".format( args.source ) )
rTree_in = rFile_in.Get( "ljmet" )
branches_in = [ branch.GetName() for branch in rTree_in.GetListOfBranches() ]

rFile_out = ROOT.TFile( args.source.replace( "hadd.root", "abcdnn.root" ).split("/")[-1],  "RECREATE" )
rFile_out.cd()
rTree_out = rTree_in.CloneTree(0)

print( ">> Creating new branches..." )
arrays = {}
branches = {}
for variable in v_in:
  arrays[ variable ] = array( "f", [0.] )
  branches[ variable ] = rTree_out.Branch( str( variable ) + "_t", arrays[ variable ], str( variable ) + "_t/F" );

print( ">> Looping through {} ({} entries)".format( args.source.split("/")[-1], rTree_in.GetEntries() ) )

for i in range( rTree_in.GetEntries() ): 
  rTree_in.GetEntry(i)
  for j, variable in enumerate( v_in ):
    arrays[ variable ][0] = predict_norm[i][j]
  rTree_out.Fill()
  
rTree_out.Write()
rFile_out.Write()
rFile_out.Close()

