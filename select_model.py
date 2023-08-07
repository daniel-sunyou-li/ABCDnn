import os
from json import loads as load_json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument( "-m", "--tag", required = True )
parser.add_argument( "-e", "--epoch", required = True )
args = parser.parse_args()

if os.path.exists( "Results/{}.index".format( args.tag ) ) and os.path.exists( "Results/{}.data-00000-of-00001".format( args.tag ) ):
  if os.path.exists( "Results/{}_EPOCH{}.index".format( args.tag, args.epoch ) ) and os.path.exists( "Results/{}_EPOCH{}.data-00000-of-00001".format( args.tag, args.epoch ) ):
    os.system( "rm Results/{}.index".format( args.tag ) )
    os.system( "rm Results/{}.data-00000-of-00001".format( args.tag ) )
    print( "[INFO] Making {}_EPOCH{} the applied model.".format( args.tag, args.epoch ) )
    os.system( "mv Results/{0}_EPOCH{1}.index Results/{0}.index".format( args.tag, args.epoch ) )
    os.system( "mv Results/{0}_EPOCH{1}.data-00000-of-00001 Results/{0}.data-00000-of-00001".format( args.tag, args.epoch ) )
    os.system( "rm Results/{}_EPOCH*".format( args.tag ) )
  else:
    quit( "[ERROR] The epoch you specified does not exist, quitting..." )

else:
  quit( "[ERROR] The tag you specified does not exist, quitting..." )
