import os, sys, getpass, pexpect
from argparse import ArgumentParser
import config

parser = ArgumentParser()
parser.add_argument( "-y", "--year", required = True, help = "2016APV,2016,2017,2018" )
args = parser.parse_args()

samples = config.samples_apply[ args.year ] 
in_path  = os.path.join( config.sourceDir[ "CONDOR" ], config.sampleDir[ args.year ] )
out_path = os.path.join( config.sourceDir[ "BRUX" ].split( "//" )[-1], config.sampleDir[ args.year ] )

if not os.path.exists( out_path ): os.system( "mkdir -vp {}".format( out_path ) )

print( ">> Transferring samples from {} --> {}".format( in_path, out_path ) )
for sample in samples:
  os.system( "xrdcp -r {} {}".format( in_path, out_path ) ) 
