# this script is run on the condor node for applying the trained ABCDnn model to ttbar samples
# last updated 11/15/2021 by Daniel Li

import numpy as np
import subprocess
import os
from tqdm.auto import tqdm
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
parser.add_argument( "-c", "--checkpoints", nargs = "+", required = True )
parser.add_argument( "-y", "--year", required = True )
parser.add_argument( "-l", "--location", default = "LPC" )
parser.add_argument( "-s", "--storage", default = "LOCAL", help = "LOCAL,EOS" )
parser.add_argument( "--test", action = "store_true" )
args = parser.parse_args()

import ROOT

# load in json file

models = {}
transfers = {}
transfer_errs = {}
disc_tags = {}
track_tags = []
print( ">> Loading checkpoints to NAF models:" )
for checkpoint in args.checkpoints:
  print( "  + {}".format( checkpoint ) )
  with open( "Results/" + checkpoint + ".json", "r" ) as f:
    params  = load_json( f.read() )
    if params[ "DISC TAG" ][0] in track_tags:
      print( "[WARN] Found redundant discriminator tag. Replacing {} --> {}".format( params[ "DISC TAG" ], checkpoint ) )
      disc_tags[ checkpoint ] = checkpoint
    else:
      disc_tags[ checkpoint ] = params["DISC TAG"][0]
    transfers[ checkpoint ] = float( params[ "TRANSFER" ] )
    transfer_errs[ checkpoint ] = float( params[ "TRANSFER ERR" ] )
    models[ checkpoint ] = abcdnn.NAF(
      inputdim    = params["INPUTDIM"],
      conddim     = params["CONDDIM"],
      activation  = params["ACTIVATION"],
      regularizer = params["REGULARIZER"],
      nodes_cond  = params["NODES_COND"],
      hidden_cond = params["HIDDEN_COND"],
      nodes_trans = params["NODES_TRANS"],
      depth       = params["DEPTH"],
      permute = True
    ) 
  models[ checkpoint ].load_weights( "Results/" + checkpoint )



# populate the step 3
def fill_tree( sample ):
  if args.storage == "EOS":
    samples_done = subprocess.check_output( "eos root://cmseos.fnal.gov ls /store/user/{}/{}".format( config.eosUserName, config.sampleDir[ args.year ] ), shell = True ).split( "\n" )[:-1]
    if sample in  samples_done:
      print( ">> [WARN] {} already processed".format( sample ) )
      return
  print( ">> Formatting sample: {}".format( sample ) )
  sample = os.path.join( config.sourceDir[ args.location ], config.sampleDir[ args.year ], sample )
  upFile = uproot.open( sample )
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
    predictions[ checkpoint ] = models[ checkpoint ].predict( normedinputs_mc )[:,0:2] * inputsigma[0:2] + inputmeans[0:2]

  rFile_in = ROOT.TFile.Open( sample )
  rTree_in = rFile_in.Get( "ljmet" )

  rFile_out = ROOT.TFile( sample.replace( "hadd.root", "ABCDnn_hadd.root" ).split("/")[-1],  "RECREATE" )
  rFile_out.cd()
  rTree_out = rTree_in.CloneTree(0)

  print( "[START] Adding new branches to {} ({} entries):".format( sample.split("/")[-1], rTree_in.GetEntries() ) )
  arrays = {}
  branches = {}
  for checkpoint in args.checkpoints:
    arrays[ checkpoint ] = { 
      "transfer_{}".format( disc_tags[ checkpoint ] ): array( "f", [0.] ),
      "transfer_err_{}".format( disc_tags[ checkpoint ] ): array( "f", [0.] )
    }
    branches[ checkpoint ] = {
      "transfer_{}".format( disc_tags[ checkpoint ] ): rTree_out.Branch( "transfer_{}".format( disc_tags[ checkpoint ] ), arrays[ checkpoint ][ "transfer_{}".format( disc_tags[ checkpoint ] ) ], "transfer_{}/F".format( disc_tags[ checkpoint ] ) ),
      "transfer_err_{}".format( disc_tags[ checkpoint ] ): rTree_out.Branch( "transfer_err_{}".format( disc_tags[ checkpoint ] ), arrays[ checkpoint ][ "transfer_err_{}".format( disc_tags[ checkpoint ] ) ], "transfer_err_{}/F".format( disc_tags[ checkpoint ] ) ) 
    }
    print( "  + transfer_{}".format( disc_tags[ checkpoint ] ) )
    print( "  + transfer_err_{}".format( disc_tags[ checkpoint ] ) )
    for variable in v_in:
      arrays[ checkpoint ][ variable ] = array( "f", [0.] )
      branches[ checkpoint ][ variable ] = rTree_out.Branch( "{}_{}".format( str( variable ), disc_tags[ checkpoint ] ) , arrays[ checkpoint ][ variable ], "{}_{}/F".format( str( variable ), disc_tags[ checkpoint ] ) );
      print( "  + {}_{}".format( str( variable ), disc_tags[ checkpoint ] ) )    
  print( ">> Looping through events" )
  for i in tqdm( range( rTree_in.GetEntries() ) ): 
    rTree_in.GetEntry(i)
    for checkpoint in args.checkpoints:
      arrays[ checkpoint ][ "transfer_{}".format( disc_tags[ checkpoint ] ) ][0] = transfers[ checkpoint ]
      arrays[ checkpoint ][ "transfer_err_{}".format( disc_tags[ checkpoint ] ) ][0] = transfer_errs[ checkpoint ]
      for j, variable in enumerate( v_in ):
        arrays[ checkpoint ][ variable ][0] = predictions[ checkpoint ][i][j]
    rTree_out.Fill()
  
  rTree_out.Write()
  rFile_out.Write()
  rFile_out.Close()
  if args.storage == "EOS":
    os.system( "xrdcp -vp {} {}".format( sample, os.path.join( config.sourceDir[ "CONDOR" ], config.sampleDir[ args.year ] ) ) )
    os.system( "rm {}".format( sample.split( "/" )[-1].replace( "hadd", "ABCDnn_hadd" ) ) )
  del rTree_in, rFile_in, rTree_out, rFile_out

if args.test:
  fill_tree( config.samples_apply[ args.year ][0] )
else:
  for sample in config.samples_apply[ args.year ]:
    fill_tree( sample )
  

