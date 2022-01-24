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
parser.add_argument( "-w", "--weights", required = True, help = "Trained weight tag" )
args = parser.parse_args()

# check if folder has necessary components

# paths 
condorDir    = config.condorDir
step3Samples = { 
  "nominal": [] 
}

# determine which samples to run on
if args.test: print( ">> Running in test mode..." )
if args.resubmit: 
  print( ">> Running in resubmit mode, resubmitting the following samples" )
  doneSamples = {
    tag: [ sample for sample in subprocess.check_output( "eos root://cmseos.fnal.gov ls /store/user/{}/{}/{}/".format( config.eosUserName, config.sampleDir[ args.year ], tag ), shell = True ).split( "\n" )[:-1] if "abcdnn" in sample ] for tag in [ "nominal", "JERup", "JERdown", "JECup", "JECdown" ]
  }
  for sample in config.samples_apply[ args.year ]:
    if sample.replace( "hadd", "abcdnn" ) not in doneSamples: step3Samples[ "nominal" ].append( sample )
else: 
  print( ">> Applying trained model {} to the following samples:" )
  step3Samples[ "nominal" ] = [ config.samples_apply[ args.year ][0] ] if args.test else config.samples_apply[ args.year ]
for sample in step3Samples[ "nominal" ]:
  print( "    - {}".format( sample ) )

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
    print( "[OK ] VOMS initialized" )

def create_tar():
  tarDir = os.getcwd()
  if "CMSSW946_ttt.tgz" in os.listdir( os.getcwd() ): os.system( "rm ABCDnn.tgz" ) 
  os.system( "tar -C ~/nobackup/TTT-singleLep/ -zcvf ABCDnn.tgz --exclude\"{}\" --exclude=\"{}\" --exclude=\"{}\" --exclude=\"{}\" --exclude=\"{}\" --exclude=\"{}\" {}".format(
      "ABCDnn/Data/*",
      "DNN/*",
      "FWLJMET/*",
      "LJMet-Slimmer-3tops/*",
      "singleLepAnalyzer/*",
      "*.tgz",
      "CMSSW_9_4_6_patch1/"
    )
  )
  print( ">> Transferring ABCDnn.tgz to EOS" )
  os.system( "xrdcp -f CMSSW946_ttt.tgz root://cmseos.fnal.gov//store/user/{}".format( config.eosUserName ) )

def condor_job( fileName, condorDir, sampleDir, logDir, weight, tag ):
  request_memory = "5120" 
  if "tttosemilepton" in fileName.lower() and "ttjj_hadd" in fileName.lower(): request_memory = "10240" 
  if args.resubmit != None: request_memory = "16384"
  dict = {
    "SAMPLENAMEIN"  : fileName,
    "SAMPLENAMEOUT" : fileName.replace( "hadd", "abcdnn" ),
    "CONDORDIR" : condorDir,           
    "SAMPLEDIR" : sampleDir,   
    "TAG"       : tag,      
    "WEIGHT"    : weight,  
    "LOGDIR"    : logDir,              
    "MEMORY"    : request_memory
  }
  jdfName = "{}/{}_{}.job".format( logDir, fileName.replace( "hadd", "abcdnn" ), tag )
  jdf = open( jdfName, "w" )
  jdf.write(
"""universe = vanilla
Executable = apply_abcdnn.sh
Should_Transfer_Files = Yes
WhenToTransferOutput = ON_EXIT
request_memory = %(MEMORY)s
Output = %(LOGDIR)s/%(SAMPLENAMEOUT)s_%(TAG)s.out
Error = %(LOGDIR)s/%(SAMPLENAMEOUT)s_%(TAG)s.err
Log = %(LOGDIR)s/%(SAMPLENAMEOUT)s_%(TAG)s.log
Notification = Never
Arguments = %(CONDORDIR)s %(SAMPLENAMEIN)s %(SAMPLENAMEOUT)s %(SAMPLEDIR)s %(WEIGHT)s %(TAG)s
Queue 1"""%dict
  )
  jdf.close()
  os.system( "condor_submit {}".format( jdfName ) )

def submit_jobs( files, key, weight, condorDir, logDir, sampleDir ):
  os.system( "mkdir -p " + logDir )
  jobCount = 0
  for file in files[key]:
    print( ">> Submitting Condor job for {}/{}".format( key, file ) )
    condor_job( file.split(".")[0], condorDir, sampleDir, logDir, weight, key )
    jobCount += 1
  print( ">> {} jobs submitted for {}/...".format( jobCount, key ) )
  return jobCount

def main( files ):
  count = submit_jobs( files, "nominal", args.weights, condorDir, args.log, config.sampleDir[ args.year ] )

voms_init()
main( step3Samples )




