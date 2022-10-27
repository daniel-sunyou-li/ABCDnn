# this script is run on the condor node for applying the trained ABCDnn model to ttbar samples
# last updated 11/15/2021 by Daniel Li

import numpy as np
import subprocess
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
parser.add_argument( "-c", "--checkpoints", nargs = "+", required = True )
parser.add_argument( "-f", "--fCondor", default = "Single file submission for Condor jobs" )
parser.add_argument( "-y", "--year", required = True )
parser.add_argument( "-l", "--location", default = "LPC" )
parser.add_argument( "-s", "--storage", default = "LOCAL", help = "LOCAL,EOS,BRUX" )
parser.add_argument( "--condor", action = "store_true" )
parser.add_argument( "--test", action = "store_true" )
parser.add_argument( "--closure", nargs = "*", help = "Add two additional branches for closure shift up and down for specified model. Need to manually add shift percent value in .json." )
parser.add_argument( "--JECup", action = "store_true", help = "Application to JECup samples" )
parser.add_argument( "--JECdown", action = "store_true", help = "Application to JECdown samples" )
args = parser.parse_args()

if not args.condor:
  from tqdm.auto import tqdm

import ROOT

if args.JECup and args.JECdown:
  sys.exit( "[ERROR] Cannot run both --JECup and --JECdown, choose one." )

dLocal = "samples_ABCDNN_UL{}".format( args.year )

if args.JECup: dLocal += "/JECup" 
elif args.JECdown: dLocal += "/JECdown" 
else: dLocal +=  "/nominal"

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
transfers = {}
transfer_errs = {}
disc_tags = {}
closure = {}
variables = {}
variables_transform = {}
upperlimits = {}
lowerlimits = {}
categoricals = {}
regions = {}
means = {}
sigmas = {}
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
    transfer_errs[ checkpoint ] = float( params[ "TRANSFER ERR" ] )
    variables[ checkpoint ] = [ str( key ) for key in sorted( params[ "VARIABLES" ] ) if params[ "VARIABLES" ][ key ][ "TRANSFORM" ] ]
    variables_transform[ checkpoint ] = [ str( key ) for key in sorted( params[ "VARIABLES" ] ) if params[ "VARIABLES" ][ key ][ "TRANSFORM" ] ]
    variables[ checkpoint ].append( params[ "REGIONS" ][ "Y" ][ "VARIABLE" ] )
    variables[ checkpoint ].append( params[ "REGIONS" ][ "X" ][ "VARIABLE" ] )
    categoricals[ checkpoint ] = [ params[ "VARIABLES" ][ key ][ "CATEGORICAL" ] for key in variables[ checkpoint ] ]
    lowerlimits[ checkpoint ] = [ params[ "VARIABLES" ][ key ][ "LIMIT" ][0] for key in variables[ checkpoint ] ]
    upperlimits[ checkpoint ] = [ params[ "VARIABLES" ][ key ][ "LIMIT" ][1] for key in variables[ checkpoint ] ]
    means[ checkpoint ] = params[ "INPUTMEANS" ]
    sigmas[ checkpoint ] = params[ "INPUTSIGMAS" ]    

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
  if args.JECdown: sampleDir = sampleDir.replace( "nominal", "JECdown" )
  elif args.JECup: sampleDir = sampleDir.replace( "nominal", "JECup" )
  if args.storage == "EOS":
    try: 
      samples_done = subprocess.check_output( "eos root://cmseos.fnal.gov ls /store/group/{}/{}".format( "lpcljm", sampleDir.replace( "step3", "step3_ABCDnn" ) ), shell = True ).split( "\n" )[:-1]
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
  def predict( checkpoint ):
    encoders = abcdnn.OneHotEncoder_int( categoricals[ checkpoint ], lowerlimit = lowerlimits[ checkpoint ], upperlimit = upperlimits[ checkpoint ] )
    inputs_mc = upTree.pandas.df( variables[ checkpoint ] )
    x_upper = inputs_mc[ regions[ checkpoint ][ "X" ][ "VARIABLE" ] ].max() if regions[ checkpoint ][ "X" ][ "INCLUSIVE" ] else regions[ checkpoint ][ "X" ][ "MAX" ]
    y_upper = inputs_mc[ regions[ checkpoint ][ "Y" ][ "VARIABLE" ] ].max() if regions[ checkpoint ][ "Y" ][ "INCLUSIVE" ] else regions[ checkpoint ][ "Y" ][ "MAX" ]
    inputs_mc[ regions[ checkpoint ][ "X" ][ "VARIABLE" ] ].clip( regions[ checkpoint ][ "X" ][ "MIN" ], x_upper )
    inputs_mc[ regions[ checkpoint ][ "Y" ][ "VARIABLE" ] ].clip( regions[ checkpoint ][ "Y" ][ "MIN" ], y_upper )
    inputs_enc = encoders.encode( inputs_mc.to_numpy( dtype = np.float32 ) )

    inputmean   = np.hstack( [ float( mean ) for mean in means[ checkpoint ] ] )
    inputsigma  = np.hstack( [ float( sigma ) for sigma in sigmas[ checkpoint ] ] )
    inputs_norm = ( inputs_enc - inputmean ) / inputsigma

    predictions[ checkpoint ] = models[ checkpoint ].predict( np.asarray( inputs_norm ) ) * inputsigma[0:2] + inputmean[0:2]
  
  if args.condor:
    for checkpoint in checkpoints:
      predict( checkpoint )
  else:
    for checkpoint in tqdm( checkpoints ):
      predict( checkpoint )

  rFile_in = ROOT.TFile.Open( sample.replace( "/isilon/hadoop", "" ) )
  rTree_in = rFile_in.Get( "ljmet" )

  rFile_out = ROOT.TFile( os.path.join( dLocal, sample.replace( "hadd", "ABCDnn_hadd" ).split("/")[-1] ),  "RECREATE" )
  rFile_out.cd()
  rTree_out = rTree_in.CloneTree(0)

  print( "[START] Adding new branches to {} ({} entries):".format( sample.split("/")[-1], rTree_in.GetEntries() ) )
  arrays = {}
  branches = {}
  for checkpoint in checkpoints:
    arrays[ checkpoint ] = { 
      "transfer_{}".format( disc_tags[ checkpoint ] ): array( "f", [0.] ),
    } 
    branches[ checkpoint ] = {
      "transfer_{}".format( disc_tags[ checkpoint ] ): rTree_out.Branch( "transfer_{}".format( disc_tags[ checkpoint ] ), arrays[ checkpoint ][ "transfer_{}".format( disc_tags[ checkpoint ] ) ], "transfer_{}/F".format( disc_tags[ checkpoint ] ) ),
    }
    print( "  + transfer_{}".format( disc_tags[ checkpoint ] ) )
    for variable in variables_transform[ checkpoint ]:
      arrays[ checkpoint ][ variable ] = array( "f", [0.] )
      branches[ checkpoint ][ variable ] = rTree_out.Branch( "{}_{}".format( str( variable ), disc_tags[ checkpoint ] ) , arrays[ checkpoint ][ variable ], "{}_{}/F".format( str( variable ), disc_tags[ checkpoint ] ) );
      if checkpoint in closure and "MODELUP" not in variable and "MODELDN" not in variable and "transfer" not in variable:
        arrays[ checkpoint ][ variable + "_CLOSUREUP" ] = array( "f", [0.] )
        branches[ checkpoint ][ variable + "_CLOSUREUP" ] = rTree_out.Branch( "{}_{}_CLOSUREUP".format( str( variable ), disc_tags[ checkpoint ] ), arrays[ checkpoint ][ variable + "_CLOSUREUP" ], "{}_{}_CLOSUREUP/F".format( str( variable ), disc_tags[ checkpoint ] ) );
        arrays[ checkpoint ][ variable + "_CLOSUREDN" ] = array( "f", [0.] )
        branches[ checkpoint ][ variable + "_CLOSUREDN" ] = rTree_out.Branch( "{}_{}_CLOSUREDN".format( str( variable ), disc_tags[ checkpoint ] ), arrays[ checkpoint ][ variable + "_CLOSUREDN" ], "{}_{}_CLOSUREDN/F".format( str( variable ), disc_tags[ checkpoint ] ) );
        print( "  + {}_{}_CLOSUREUP".format( str( variable ), disc_tags[ checkpoint ] ) )
        print( "  + {}_{}_CLOSUREDN".format( str( variable ), disc_tags[ checkpoint ] ) )
                
      print( "  + {}_{}".format( str( variable ), disc_tags[ checkpoint ] ) )    
  print( ">> Looping through events" )
  def loop_tree( rTree_in, rTree_out, i ):
    rTree_in.GetEntry(i)
    for checkpoint in checkpoints:
      arrays[ checkpoint ][ "transfer_{}".format( disc_tags[ checkpoint ] ) ][0] = transfers[ checkpoint ]
      for j, variable in enumerate( variables_transform[ checkpoint ] ):
        arrays[ checkpoint ][ variable ][0] = predictions[ checkpoint ][i][j]
        if checkpoint in closure and "MODELUP" not in variable and "MODELDN" not in variable and "transfer" not in variable:
          arrays[ checkpoint ][ variable + "_CLOSUREUP" ][0] = ( 1. + closure[ checkpoint ] ) * predictions[ checkpoint ][i][j]
          arrays[ checkpoint ][ variable + "_CLOSUREDN" ][0] = ( 1. / ( 1. + closure[ checkpoint ] ) ) * predictions[ checkpoint ][i][j]
    rTree_out.Fill()
  
  if args.condor:
    progress = np.around( np.linspace( 0, rTree_in.GetEntries(), 11 ), 0 )
    for i in range( rTree_in.GetEntries() ):
      loop_tree( rTree_in, rTree_out, i )
      if i in progress:
        print( ">> {}/{} events processed".format( i, rTree_in.GetEntries() ) )
  else:
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
elif args.condor:
  fill_tree( args.fCondor, dLocal )
else:
  for sample in config.samples_apply[ args.year ]:
    fill_tree( sample, dLocal )
