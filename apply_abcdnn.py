# apply a trained ABCDnn model to ttbar samples and create new step3's
# last updated 10/25/2021 by Daniel Li

import glob, os, sys, subprocess
import config
import numpy as np
from datetime import datetime
from argparse import ArgumentParser

# read in arguments
parser = ArgumentParser()
parser.add_argument( "-y", "--year", required = True, default = "2017" )
parser.add_argument( "-t", "--test", action = "store_true" )
parser.add_argument( "-l", "--log", default= "application_log_" + datetime.now().strftime("%d.%b.%Y") )
parser.add_argument( "-r", "--resubmit", action = "store_true" )
parser.add_argument( "-c", "--checkpoints", nargs = "+", required = True, help = "Trained weight tag(s)" )
parser.add_argument( "-loc", "--location", default = "LPC", help = "LPC,BRUX" )
args = parser.parse_args()

# check if folder has necessary components

# paths 
condorDir    = config.sourceDir[ "CONDOR" ]
step3Samples = { "nominal": [] }

# determine which samples to run on
if args.test: print( ">> Running in test mode..." )

if args.location == "LPC":
  sourceDir = config.sourceDir[ "LPC" ]
elif args.location == "BRUX":
  sourceDir = config.sourceDir[ "BRUX" ]


print( ">> Running checkpoints: " )
all_checkpoints = [ checkpoint.split( "." )[0] for checkpoint in os.listdir( "Results/" ) if ".data" in checkpoint ]
for checkpoint in args.checkpoints:
  if "." in checkpoint: 
    print( "[WARN] Do not include file extension in -c (--checkpoints). Please resubmit {} as {}".format( checkpoint, checkpoint.split( "." )[0] ) )
  elif checkpoint not in all_checkpoints:
    print( "[ERR] {} is not a valid checkpoint. Please resubmit.".format( checkpoint ) )
  else:
    print( "  + {}".format( checkpoint ) )
    
if args.resubmit: 
  print( "[OPT] Running in resubmit mode." )
  print( ">> Resubmitting ABCDnn to samples:" )
  doneSamples = {
    tag: [ sample for sample in subprocess.check_output( "eos root://cmseos.fnal.gov ls /store/user/{}/{}/{}/".format( config.eosUserName, config.sampleDir[ args.year ], tag ), shell = True ).split( "\n" )[:-1] if "ABCDNN" in sample.upper() ] for tag in [ "nominal" ]
  }
  for sample in config.samples_apply[ args.year ]:
    if sample.replace( "hadd", "ABCDnn_hadd" ) not in doneSamples[ "nominal" ]: 
      step3Samples[ "nominal" ].append( sample )
else: 
  print( "[OPT] Running in submit mode." )
  print( ">> Applying ABCDnn to samples:" )
  step3Samples[ "nominal" ] = [ config.samples_apply[ args.year ][0] ] if args.test else config.samples_apply[ args.year ]

for sample in step3Samples[ "nominal" ]:
  print( "  + {}".format( sample ) )

# general methods
def check_voms():
  print( ">> Checking VOMS" )
  try:
    output = subprocess.check_output( "voms-proxy-info", shell = True )
    if output.rfind( "timeleft" ) > - 1:
      if int( output[ output.rfind(": ")+2: ].replace( ":", "" ) ) > 0:
        print( "[OK ] VOMS found" )
        return True
      return False
  except: return False

def voms_init():
  if not check_voms():
    print( ">> Initializing VOMS" )
    output = subprocess.check_output( "voms-proxy-init --voms cms", shell = True )
    if "failure" in output:
      print( "[WARN] Incorrect password entered. Try again." )
      voms_init()
    print( "[OK] VOMS initialized" )

def create_tar():
  tarDir = os.getcwd()
  if "ABCDnn.tgz" in os.listdir( os.getcwd() ): os.system( "rm ABCDnn.tgz" ) 
  os.system( "tar -C {} -zcvf ABCDnn.tgz --exclude=\"{}\" --exclude=\"{}\" --exclude=\"{}\" --exclude=\"{}\" {}".format(
      os.path.join( os.getcwd(), "../../../" ),
      "ABCDnn/Data/*",
      "ABCDnn/Results/*.gif",
      "ABCDnn/Results/*.png",
      "*.tgz",
      "CMSSW_10_6_19/"
    )
  )
  print( ">> Transferring ABCDnn.tgz to EOS" )
  os.system( "xrdcp -f ABCDnn.tgz root://cmseos.fnal.gov//store/user/{}".format( config.eosUserName ) )

def condor_job( fileName, condorDir, sampleDir, logDir, checkpoints, tag ):
  request_memory = "5120" 
  if "tttosemilepton" in fileName.lower() and "ttjj" in fileName.lower(): request_memory = "10240" 
  if args.resubmit != None: request_memory = "16384"
  dict = {
    "SAMPLENAMEIN"  : fileName,
    "SAMPLENAMEOUT" : fileName.replace( "hadd", "ABCDnn_hadd" ),
    "CONDORDIR"     : condorDir,           
    "SAMPLEDIR"     : sampleDir,   
    "TAG"           : tag,
    "CHECKPOINTS"   : checkpoints,
    "CKP_JSON"      : [ checkpoint + ".json" for checkpoint in checkpoints ], 
    "CKP_INDEX"     : [ checkpoint + ".index" for checkpoint in checkpoints ], 
    "CKP_DATA"      : [ checkpoint + ".data-00000-of-00001" for checkpoint in checkpoints ],
    "LOGDIR"        : logDir,              
    "MEMORY"        : request_memory
  }
  jdfName = "{}/{}_{}.job".format( logDir, fileName.replace( "hadd", "ABCDnn_hadd" ), tag )
  jdf = open( jdfName, "w" )
  jdf.write(
"""universe = vanilla
Executable = apply_abcdnn.sh
Should_Transfer_Files = Yes
Transfer_Input_Files = %(CKP_JSON)s %(CKP_INDEX)s %(CKP_DATA)s
WhenToTransferOutput = ON_EXIT
request_memory = %(MEMORY)s
Output = %(LOGDIR)s/%(SAMPLENAMEOUT)s_%(TAG)s.out
Error = %(LOGDIR)s/%(SAMPLENAMEOUT)s_%(TAG)s.err
Log = %(LOGDIR)s/%(SAMPLENAMEOUT)s_%(TAG)s.log
Notification = Never
Arguments = %(CONDORDIR)s %(SAMPLENAMEIN)s %(SAMPLENAMEOUT)s %(SAMPLEDIR)s %(CHECKPOINTS)s
Queue 1"""%dict
  )
  jdf.close()
  os.system( "condor_submit {}".format( jdfName ) )

def submit_jobs( files, key, checkpoints, condorDir, logDir, sampleDir ):
  os.system( "mkdir -vp " + logDir )
  jobCount = 0
  print( "[START] Submitting {} Condor jobs: ".format( key ) )
  for file in files[key]:
    print( "  + {}".format( file ) )
    condor_job( file.split(".")[0], condorDir, sampleDir, logDir, checkpoints, key )
    jobCount += 1
  print( "[DONE] {} jobs submitted.".format( jobCount ) )
  return jobCount

def main( files ):
  count = submit_jobs( files, "nominal", args.checkpoints, condorDir, args.log, os.path.join( config.sourceDir[ args.location ], config.sampleDir[ args.year ] ) )

voms_init()
main( step3Samples )




