# apply a trained ABCDnn model to ttbar samples and create new step3's
# last updated 10/20/2022 by Daniel Li

import glob, os, sys, subprocess
import config
import numpy as np
from datetime import datetime
from argparse import ArgumentParser

# read in arguments
parser = ArgumentParser()
parser.add_argument( "-y", "--year", required = True, default = "2017" )
parser.add_argument( "-c", "--checkpoints", nargs = "+", required = True, help = "Trained weight tag(s)" )
parser.add_argument( "--closure", nargs = "+" )
parser.add_argument( "--test", action = "store_true" )
parser.add_argument( "--JECup", action = "store_true" )
parser.add_argument( "--JECdown", action = "store_true" )
parser.add_argument( "-s", "--dSource", default = "LPC", help = "Step3 source location: LPC,BRUX" )
parser.add_argument( "-l", "--dTarget", default = "LPC", help = "ABCDnn target location: LPC,BRUX" )
parser.add_argument( "--log", default= "application_log_" + datetime.now().strftime("%d.%b.%Y") )
parser.add_argument( "--resubmit", action = "store_true" )
args = parser.parse_args()

# check if folder has necessary components

tags = [ "nominal" ]
if args.JECup: tags + [ "JECup" ]
if args.JECdown: tags + [ "JECdown" ]

step3Samples = { tag: [] for tag in tags }

# determine which samples to run on
if args.test: print( ">> Running in test mode..." )

if args.dSource == "LPC":
  sourceDir = config.sourceDir[ "LPC" ]
elif args.dSource == "BRUX":
  sourceDir = config.sourceDir[ "BRUX" ]
if args.dTarget == "LPC":
  targetDir = config.targetDir[ "LPC" ]
elif args.dTarget == "BRUX":
  targetDir = config.targetDir[ "BRUX" ]

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
  print( ">> Resubmitting ABCDnn jobs" )
  doneSamples = {
    tag: [ sample for sample in subprocess.check_output( "eos root://cmseos.fnal.gov ls {}".format( os.path.join( targetDir, config.sampleDir[ args.year ], tag ) ), shell = True ).split( "\n" )[:-1] ] for tag in tags
  }
  for tag in tags:
    for sample in config.samples_apply[ args.year ]:
      if sample.replace( "hadd", "ABCDnn_hadd" ) not in doneSamples[ tag ]: 
        step3Samples[ tag ].append( sample )
else: 
  print( "[OPT] Running in submit mode." )
  print( ">> Submitting ABCDnn jobs" )
  for tag in tags:
    step3Samples[ tag ] = [ config.samples_apply[ args.year ][0] ] if args.test else config.samples_apply[ args.year ] 
    print( "  + {}: {} jobs".format( tag, len( step3Samples[ tag ] ) ) )

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
      "ABCDnn/Results/*",
      "ABCDnn/*log*/*",
      "*.tgz",
      "CMSSW_10_6_29/"
    )
  )
  print( ">> Transferring ABCDnn.tgz to EOS" )
  os.system( "xrdcp -f ABCDnn.tgz root://cmseos.fnal.gov//store/user/{}".format( config.eosUserName ) )

def condor_job( fileName, condorDir, sampleDir, logDir, checkpoints, closure, tag ):
  request_memory = "5120" 
  if "tttosemilepton" in fileName.lower() and "ttjj" in fileName.lower(): request_memory = "10240" 
  if args.resubmit != None: request_memory = "16384"
  dict = {
    "YEAR"          : args.year,
    "SAMPLENAMEIN"  : fileName,
    "SAMPLENAMEOUT" : fileName.replace( "hadd", "ABCDnn_hadd" ),
    "SOURCEDIR"     : args.dSource,
    "TARGETDIR"     : args.dTarget,
    "CONDORDIR"     : condorDir,           
    "SAMPLEDIR"     : sampleDir,   
    "TAG"           : tag,
    "CHECKPOINTS"   : ",".join( checkpoints ),
    "CKP_JSON"      : ", ".join( [ "Results/" + checkpoint + ".json" for checkpoint in checkpoints ] ), 
    "CKP_INDEX"     : ",  ".join( [ "Results/" + checkpoint + ".index" for checkpoint in checkpoints ] ), 
    "CKP_DATA"      : ",  ".join( [ "Results/" + checkpoint + ".data-00000-of-00001" for checkpoint in checkpoints ] ),
    "CLOSURE"       : ",".join( closure ),
    "LOGDIR"        : logDir,              
    "MEMORY"        : request_memory
  }
  jdfName = "{}/{}_{}.job".format( logDir, fileName.replace( "hadd", "ABCDnn_hadd" ), tag )
  jdf = open( jdfName, "w" )
  jdf.write(
"""universe = vanilla
Executable = condor_abcdnn.sh
Should_Transfer_Files = Yes
Transfer_Input_Files = %(CKP_JSON)s, %(CKP_INDEX)s, %(CKP_DATA)s, abcdnn.py, apply_abcdnn.py, config.py
WhenToTransferOutput = ON_EXIT
request_memory = %(MEMORY)s
Output = %(LOGDIR)s/%(SAMPLENAMEOUT)s_%(TAG)s.out
Error = %(LOGDIR)s/%(SAMPLENAMEOUT)s_%(TAG)s.err
Log = %(LOGDIR)s/%(SAMPLENAMEOUT)s_%(TAG)s.log
Notification = Never
Arguments = %(CONDORDIR)s %(SOURCEDIR)s %(TARGETDIR)s %(SAMPLEDIR)s %(SAMPLENAMEIN)s %(SAMPLENAMEOUT)s %(CHECKPOINTS)s %(CLOSURE)s %(TAG)s %(YEAR)s
Queue 1"""%dict
  )
  jdf.close()
  os.system( "condor_submit {}".format( jdfName ) )

def submit_jobs( files, tag, checkpoints, closure, condorDir, logDir, sampleDir ):
  os.system( "mkdir -vp " + logDir )
  jobCount = 0
  print( "[START] Submitting {} Condor jobs: ".format( tag ) )
  for file in files[tag]:
    print( "  + {}".format( file ) )
    condor_job( file.split(".")[0], condorDir, sampleDir, logDir, checkpoints, closure, tag )
    jobCount += 1
  print( "[DONE] {} jobs submitted.".format( jobCount ) )
  return jobCount

def main( files ):
  #create_tar()
  for tag in files:
    count = submit_jobs( files, tag, args.checkpoints, args.closure, config.condorDir, args.log, os.path.join( config.targetDir[ args.dSource ], config.sampleDir[ args.year ] ) )

voms_init()
main( step3Samples )
