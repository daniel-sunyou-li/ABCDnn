# this script is run on the condor node for applying the trained ABCDnn model to ttbar samples
# last updated 11/15/2021 by Daniel Li

import numpy as np
import subprocess
import os
import uproot
import abcdnn
import sys
from argparse import ArgumentParser
from json import loads as load_json
from array import array
import multiprocessing
from tqdm.auto import tqdm

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import config

parser = ArgumentParser()
parser.add_argument( "-c", "--checkpoints", nargs = "+", required = True )
parser.add_argument( "-f", "--condor", default = "Single file submission for Condor jobs" )
parser.add_argument( "-y", "--year", required = True )
parser.add_argument( "-l", "--location", default = "LPC" )
parser.add_argument( "-s", "--storage", default = "LOCAL", help = "LOCAL,EOS,BRUX" )
parser.add_argument( "--test", action = "store_true" )
parser.add_argument( "--closure", nargs = "*", help = "Add two additional branches for closure shift up and down for specified model." )
args = parser.parse_args()

import ROOT

dLocal = "samples_ABCDNN_UL{}/nominal/".format( args.year )

closure_checkpoints = []
try:
  if "," in args.closure:
    for arg in args.closure.split(","):
      closure_checkpoints.append( arg )
except:
  pass
if args.closure is not None: closure_checkpoints = args.closure

checkpoints = []
if "," in args.checkpoints:
  for arg in args.checkpoints.split( "," ):
    checkpoints.append( arg ) 
else:
  for arg in args.checkpoints:
    checkpoints.append( arg )

if not os.path.exists( dLocal ):
  os.system( "mkdir -vp {}".format( dLocal ) )

# load in json file

models = {}
disc_tags = {}
closure = {}
variables = {}
variables_transform = {}
upperlimits = {}
lowerlimits = {}
categoricals = {}
regions = {}
transfers = {}
means = {}
sigmas = {}
means_pred = {}
sigmas_pred = {}
track_tags = []

print( "[START] Loading checkpoints to NAF models:" )
for checkpoint in checkpoints:
  with open( "Results/" + checkpoint + ".json", "r" ) as f:
    params  = load_json( f.read() )
    if params[ "DISC TAG" ][0] in track_tags:
      print( "[WARN] Found redundant discriminator tag. Replacing {} --> {}".format( params[ "DISC TAG" ], checkpoint ) )
      disc_tags[ checkpoint ] = checkpoint
    else:
      disc_tags[ checkpoint ] = params["DISC TAG"][0]
      track_tags.append( params[ "DISC TAG" ][0] )
    if checkpoint in closure_checkpoints:
      try:
        closure[ checkpoint ] = float( params["CLOSURE"] )
      except:
        sys.exit( "[WARNING] {0}.json missing CLOSURE entry necessary for --closure arugment. Calculate using evaluate_model.py --closure and then add to {0}.json.".format( checkpoint ) )
    regions[ checkpoint ] = params[ "REGIONS" ]
    transfers[ checkpoint ] = float( params[ "TRANSFER" ] )
    variables[ checkpoint ] = [ str( key ) for key in sorted( params[ "VARIABLES" ] ) if params[ "VARIABLES" ][ key ][ "TRANSFORM" ] ]
    variables_transform[ checkpoint ] = [ str( key ) for key in sorted( params[ "VARIABLES" ] ) if params[ "VARIABLES" ][ key ][ "TRANSFORM" ] ]
    variables[ checkpoint ].append( params[ "REGIONS" ][ "Y" ][ "VARIABLE" ] )
    variables[ checkpoint ].append( params[ "REGIONS" ][ "X" ][ "VARIABLE" ] )
    categoricals[ checkpoint ] = [ params[ "VARIABLES" ][ key ][ "CATEGORICAL" ] for key in variables[ checkpoint ] ]
    lowerlimits[ checkpoint ] = [ params[ "VARIABLES" ][ key ][ "LIMIT" ][0] for key in variables[ checkpoint ] ]
    upperlimits[ checkpoint ] = [ params[ "VARIABLES" ][ key ][ "LIMIT" ][1] for key in variables[ checkpoint ] ]
    means[ checkpoint ] = params[ "INPUTMEANS" ]
    sigmas[ checkpoint ] = params[ "INPUTSIGMAS" ]    
    means_pred[ checkpoint ] = params[ "SIGNAL_MEAN" ] 
    sigmas_pred[ checkpoint ] = params[ "SIGNAL_SIGMA" ] 

    models[ checkpoint ] = abcdnn.NAF(
      inputdim    = params["INPUTDIM"],
      conddim     = params["CONDDIM"],
      activation  = params["ACTIVATION"],
      regularizer = params["REGULARIZER"],
      initializer = params["INITIALIZER"],
      nodes_cond  = params["NODES_COND"],
      hidden_cond = params["HIDDEN_COND"],
      nodes_trans = params["NODES_TRANS"],
      depth       = params["DEPTH"],
      permute     = params["PERMUTE"]
    ) 
  models[ checkpoint ].load_weights( "Results/" + checkpoint )
  print( "  + {} --> {}".format( checkpoint, disc_tags[ checkpoint ] ) )

# populate the step 3
def fill_tree( sample, dLocal ):
  sampleDir = config.sampleDir[ args.year ]
  if args.storage == "EOS":
    try: 
      samples_done = subprocess.check_output( "eos root://cmseos.fnal.gov ls /store/user/{}/{}".format( "dali", sampleDir.replace( "step3", "step3_ABCDnn" ) ), shell = True ).split( "\n" )[:-1]
    except: 
      samples_done = []
    if sample.replace( "hadd", "ABCDnn_hadd" ) in samples_done:
      print( ">> [WARN] {} already processed".format( sample ) )
      return
  print( "[START] Formatting sample: {}".format( sample ) )
  sample_ACBDnn = sample.replace( "hadd", "ABCDnn_hadd" )
  sample = os.path.join( config.sourceDir[ args.location ], sampleDir, sample ).replace( "/isilon/hadoop", "" )
  upFile = uproot.open( sample )
  upTree = upFile[ "ljmet" ]

  predictions = {}
  print( ">> Formatting, encoding and predicting on MC events" )

  # shift the input value by a gaussian-term
  def shift_encode( input_, mean_, sigma_, const_, shift_ ): 
    input_scale = input_.copy()
    if shift_ == "UP":
      input_scale[0] = input_[0] + const_ * input_[0] * ( 1 - input_[0] ) * np.exp( - ( ( input_[0] - mean_[0] ) / sigma_[0] )**2 )
      input_scale[1] = input_[1] + const_ * input_[1] * ( 1 - input_[1] ) *  np.exp( - ( ( input_[1] - mean_[1] ) / sigma_[1] )**2 )
    if shift_ == "DN":
      input_scale[0] = input_[0] - const_ * input_[0] * ( 1 - input_[0] ) * np.exp( - ( ( input_[0] - mean_[0] ) / sigma_[0] )**2 )
      input_scale[1] = input_[1] - const_ * input_[1] * ( 1 - input_[1] ) * np.exp( - ( ( input_[1] - mean_[1] ) / sigma_[1] )**2 )
    return input_scale

  # this should run on the predicted ABCDnn output to vary the peak location while fixing the tails to prescribe an uncertainty to the peak shape
  def shift_peak( input_, mean_, sigma_, const_, shift_ ): 
    # const_ represents the maximum value the peak can shift
    if shift_ == "UP":
      return input_ + const_ * input_ * ( 1 - input_ ) * np.exp( -( ( input_ - mean_ ) / sigma_ )**2 )
    elif shift_ == "DN":
      return input_ - const_ * input_ * ( 1 - input_ ) * np.exp( -( ( input_ - mean_ ) / sigma_ )**2 )
    else:
      quit( "[WARN] Invalid shift used, quitting..." )

  def shift_tail( input_, mean_, sigma_, const_, shift_ ):
    # const_ represents the maximum value the tails can shift
    if shift_ == "UP":
      return input_ + const_ * input_ * ( 1 - input_ ) * ( 1 - np.exp( - 0.5 * ( ( input_ - mean_ ) / sigma_ )**2 ) ) 
    elif shift_ == "DN":
      return input_ - const_ * input_ * ( 1 - input_ ) * ( 1 - np.exp( - 0.5 * ( ( input_ - mean_ ) / sigma_ )**2 ) )
    else:
      quit( "[WARN] Invalid shift used, quitting..." )

  def predict( checkpoint ):
    
    encoders = abcdnn.OneHotEncoder_int( categoricals[ checkpoint ], lowerlimit = lowerlimits[ checkpoint ], upperlimit = upperlimits[ checkpoint ] )
    inputs_mc = upTree.pandas.df( variables[ checkpoint ] )
    inputmean   = np.hstack( [ float( mean ) for mean in means[ checkpoint ] ] )
    inputsigma  = np.hstack( [ float( sigma ) for sigma in sigmas[ checkpoint ] ] )
    
    x_upper = inputs_mc[ regions[ checkpoint ][ "X" ][ "VARIABLE" ] ].max() if regions[ checkpoint ][ "X" ][ "INCLUSIVE" ] else regions[ checkpoint ][ "X" ][ "MAX" ]
    y_upper = inputs_mc[ regions[ checkpoint ][ "Y" ][ "VARIABLE" ] ].max() if regions[ checkpoint ][ "Y" ][ "INCLUSIVE" ] else regions[ checkpoint ][ "Y" ][ "MAX" ]
    inputs_mc[ regions[ checkpoint ][ "X" ][ "VARIABLE" ] ].clip( regions[ checkpoint ][ "X" ][ "MIN" ], x_upper )
    inputs_mc[ regions[ checkpoint ][ "Y" ][ "VARIABLE" ] ].clip( regions[ checkpoint ][ "Y" ][ "MIN" ], y_upper )
    inputs_enc = { 
      "NOM": encoders.encode( inputs_mc.to_numpy( dtype = np.float32 ) ), 
    }
    if checkpoint in closure:
      inputs_enc[ "CLOSUREUP" ] = []
      inputs_enc[ "CLOSUREDN" ] = []

    for input_ in inputs_enc[ "NOM" ]:
      if checkpoint in closure:
        inputs_enc[ "CLOSUREUP" ].append( shift_encode( input_, inputmean, inputsigma, closure[ checkpoint ], "UP" ) )
        inputs_enc[ "CLOSUREDN" ].append( shift_encode( input_, inputmean, inputsigma, closure[ checkpoint ], "DN" ) )
    
    inputs_norm = {}
    predictions[ checkpoint ] = {}
    for key in inputs_enc:
      inputs_norm[ key ] = ( inputs_enc[ key ] - inputmean ) / inputsigma 

    for key_ in inputs_norm:
      predictions[ checkpoint ][ key_ ] = models[ checkpoint ].predict( inputs_norm[ key_ ] ) * inputsigma[0:2] + inputmean[0:2]
    
  for checkpoint in tqdm( checkpoints ):
    predict( checkpoint )

  rFile_in = ROOT.TFile.Open( sample.replace( "/isilon/hadoop", "" ) )
  rTree_in = rFile_in.Get( "ljmet" )
  rTree_in.SetBranchStatus( "*", 0 )
  for bName in config.branches:
    rTree_in.SetBranchStatus( bName, 1 )

  rFile_out = ROOT.TFile( os.path.join( dLocal, sample.replace( "hadd", "ABCDnn_hadd" ).split("/")[-1] ),  "RECREATE" )
  rFile_out.cd()
  rTree_out = rTree_in.CloneTree(0)

  print( "[START] Adding new branches to {} ({} entries):".format( sample.split("/")[-1], rTree_in.GetEntries() ) )
  arrays = {}
  branches = {}
  for checkpoint in checkpoints:
    arrays[ checkpoint ] = { 
      "transfer_{}".format( disc_tags[ checkpoint ] ): array( "f", [0.] )
    } 
    branches[ checkpoint ] = { 
      "transfer_{}".format( disc_tags[ checkpoint ] ): rTree_out.Branch( "transfer_{}".format( disc_tags[ checkpoint ] ), arrays[ checkpoint ][ "transfer_{}".format( disc_tags[ checkpoint ] ) ], "transfer_{}/F".format( disc_tags[ checkpoint ] ) ), 
    }
    print( " + transfer_{}".format( disc_tags[ checkpoint ] ) )
    for variable in variables_transform[ checkpoint ]:
      arrays[ checkpoint ][ variable ] = array( "f", [0.] )
      branches[ checkpoint ][ variable ] = rTree_out.Branch( "{}_{}".format( str( variable ), disc_tags[ checkpoint ] ) , arrays[ checkpoint ][ variable ], "{}_{}/F".format( str( variable ), disc_tags[ checkpoint ] ) );
      print( " + {}_{}".format( str( variable ), disc_tags[ checkpoint ] ) )    
      for shift_ in [ "UP", "DN" ]:
        # add the output peak and tail shift branches
        arrays[ checkpoint ][ variable + "_PEAK" + shift_ ] = array( "f", [0.] )
        branches[ checkpoint ][ variable + "_PEAK" + shift_ ] = rTree_out.Branch( 
          "{}_{}_PEAK{}".format( str(variable), disc_tags[checkpoint], shift_ ),
          arrays[ checkpoint ][ variable + "_PEAK" + shift_ ], 
          "{}_{}_PEAK{}/F".format( str(variable), disc_tags[checkpoint], shift_ )
        );
        arrays[ checkpoint ][ variable + "_TAIL" + shift_ ] = array( "f", [0.] )
        branches[ checkpoint ][ variable + "_TAIL" + shift_ ] = rTree_out.Branch(
          "{}_{}_TAIL{}".format( str(variable), disc_tags[checkpoint], shift_ ),
          arrays[ checkpoint ][ variable + "_TAIL" + shift_ ],
          "{}_{}_TAIL{}/F".format( str(variable), disc_tags[checkpoint], shift_ )
        );
        print( " + {}_{}_PEAK{}".format( str(variable), disc_tags[ checkpoint ], shift_ ) )
        print( " + {}_{}_TAIL{}".format( str(variable), disc_tags[ checkpoint ], shift_ ) )
        # add the closure branch for varying the input to ABCDnn 
        if checkpoint in closure and "transfer" not in variable:
          arrays[checkpoint][variable + "_CLOSURE" + shift_] = array( "f", [0.] )
          branches[checkpoint][variable + "_CLOSURE" + shift_] = rTree_out.Branch( "{}_{}_CLOSURE{}".format( str( variable ), disc_tags[ checkpoint ], shift_ ), arrays[checkpoint][variable + "_CLOSURE" + shift_], "{}_{}_CLOSURE{}/F".format( str(variable), disc_tags[ checkpoint ], shift_ ) );
          print( " + {}_{}_CLOSURE{}".format( str( variable ), disc_tags[ checkpoint ], shift_ ) )
                
  print( ">> Looping through events" )
  def loop_tree( rTree_in, rTree_out, i ):
    rTree_in.GetEntry(i)
    for checkpoint in checkpoints:
      arrays[ checkpoint ][ "transfer_{}".format( disc_tags[ checkpoint ] ) ][0] = transfers[ checkpoint ]
      for j, variable in enumerate( variables_transform[ checkpoint ] ):
        arrays[checkpoint][variable][0] = predictions[checkpoint]["NOM"][i][j]
        for shift_ in [ "UP", "DN" ]:
          arrays[checkpoint][variable + "_PEAK" + shift_][0] = shift_peak( predictions[checkpoint]["NOM"][i][j], float( means_pred[checkpoint][j] ), float( sigmas_pred[checkpoint][j] ), closure[checkpoint], shift_ )
          arrays[checkpoint][variable + "_TAIL" + shift_][0] = shift_tail( predictions[checkpoint]["NOM"][i][j], float( means_pred[checkpoint][j] ), float( sigmas_pred[checkpoint][j] ), closure[checkpoint], shift_ )
          if checkpoint in closure:
            arrays[checkpoint][variable + "_CLOSURE" + shift_][0] = predictions[checkpoint]["CLOSURE" + shift_][i][j]
    rTree_out.Fill()
  
  for i in tqdm( range( rTree_in.GetEntries() ) ): 
    loop_tree( rTree_in, rTree_out, i )    

  rTree_out.Write()
  rFile_out.Write()
  rFile_out.Close()
  if args.storage == "EOS":
    os.system( "xrdcp -vfp {} {}".format( os.path.join( dLocal, sample.split( "/" )[-1].replace( "hadd", "ABCDnn_hadd" ) ), os.path.join( config.targetDir[ "LPC" ], sampleDir.replace( "step3", "step3_ABCDnn" ) ) ) )
    os.system( "rm {}".format( os.path.join( dLocal, sample.split( "/" )[-1].replace( "hadd", "ABCDnn_hadd" ) ) ) )
  del rTree_in, rFile_in, rTree_out, rFile_out

if args.test:
  fill_tree( config.samples_apply[ args.year ][0], dLocal )
else:
  for sample in config.samples_apply[ args.year ]:
    fill_tree( sample, dLocal )
