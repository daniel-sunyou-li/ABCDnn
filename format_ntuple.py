# this script takes an existing step3 ROOT file and formats it for use in the ABCDnn training script
# last modified April 25, 2022 by Daniel Li

import os, sys, ROOT
from array import array
from argparse import ArgumentParser
import config
import tqdm

parser = ArgumentParser()
parser.add_argument( "-y",  "--year", default = "2017", help = "Year for sample" )
parser.add_argument( "-sO", "--sOut", default = "source_ttbar", help = "Output name of source ROOT file" )
parser.add_argument( "-tO", "--tOut", default = "target_data", help = "Output name of target ROOT file" )
parser.add_argument( "-v",  "--variables", nargs = "+", default = [ "AK4HT", "DNN_1to40_3t" ], help = "Variables to transform" )
parser.add_argument( "-p",  "--pEvents", default = 100, help = "Percent of events (0 to 100) to include from each file." )
parser.add_argument( "-l",  "--location", default = "LPC", help = "LPC,BRUX" )
parser.add_argument( "--doMC", action = "store_true" )
parser.add_argument( "--doData", action = "store_true" )
parser.add_argument( "--JECup", action = "store_true", help = "Create an MC dataset using the JECup shift for ttbar" )
parser.add_argument( "--JECdown", action = "store_true", help = "Create an MC dataset using the JECdown shift for ttbar" )
args = parser.parse_args()

if args.location not in [ "LPC", "BRUX" ]: quit( "[ERR] Invalid -l (--location) argument used. Quitting..." )

ttbar_xsec = 831.76
target_lumi = {
  "2016APV": 19520.,
  "2016":    16810.,
  "2017":    41480.,
  "2018":    59832.
}
BR = {
  "Hadronic": 0.457,
  "SemiLeptonic": 0.438,
  "2L2Nu": 0.105,
  "HT500Njet9": 0.438 * 0.00617938417682763
}
num_MC = {
  "2016APV": {
    "Hadronic":     93137016.0,
    "SemiLeptonic": 133661764.0,
    "2L2Nu":        40819800.0,
    "HT500Njet9":   4443604.0
  },
  "2016": {
    "Hadronic":     207187204.,
    "SemiLeptonic": 148086112.,
    "2L2Nu":        47141720.,
    "HT500Njet9":   4603338.
  },
  "2017": {
    "Hadronic":     233815417.,
    "SemiLeptonic": 683741954.,
    "2L2Nu":        105697364.,
    "HT500Njet9":   17364358.
  },
  "2018": {
    "Hadronic":     322629460.,
    "SemiLeptonic": 547148148.,
    "2L2Nu":        126685058.,
    "HT500Njet9":   16122362.
  }
}

weight_ttbar = {}
for finalstate in BR:
  weight_ttbar[ finalstate ] = ttbar_xsec * target_lumi[ args.year ] * BR[ finalstate ] / num_MC[ args.year ][ finalstate ]
  
ROOT.gInterpreter.Declare("""
    float compute_weight( float triggerXSF, float triggerSF, float pileupWeight, float lepIdSF, float isoSF, float L1NonPrefiringProb_CommonCalc, float MCWeight_MultiLepCalc, float xsecEff, float tthfWeight, float btagDeepJetWeight, float btagDeepJet2DWeight_HTnj ){
      return 3 * triggerXSF * triggerSF * pileupWeight * lepIdSF * isoSF * L1NonPrefiringProb_CommonCalc * ( MCWeight_MultiLepCalc / abs( MCWeight_MultiLepCalc ) ) * xsecEff * tthfWeight * btagDeepJetWeight * btagDeepJet2DWeight_HTnj;
  }
""")

class ToyTree:
  def __init__( self, name, trans_var ):
    # trans_var is transforming variables
    self.name = name
    self.rFile = ROOT.TFile.Open( "{}.root".format( name ), "RECREATE" )
    self.rTree = ROOT.TTree( "Events", name )
    self.variables = { # variables that are used regardless of the transformation variables
      "xsecWeight": { "ARRAY": array( "f", [0] ), "STRING": "xsecWeight/F" }
    }
    for variable in config.variables.keys():
      if not config.variables[ variable ][ "TRANSFORM" ]:
        self.variables[ variable ] = { "ARRAY": array( "i", [0] ), "STRING": str(variable) + "/I" }
    
    for variable in trans_var:
      self.variables[ variable ] = { "ARRAY": array( "f", [0] ), "STRING": "{}/F".format( variable ) }
   
    self.selection = config.selection 
    
    for variable in self.variables:
      self.rTree.Branch( str( variable ), self.variables[ variable ][ "ARRAY" ], self.variables[ variable ][ "STRING" ] )
    
  def Fill( self, event_data ):
    for variable in self.variables:
      self.variables[ variable ][ "ARRAY" ][0] = event_data[ variable ]
    self.rTree.Fill()
  
  def Write( self ):
      print( ">> Writing {} entries to {}.root".format( self.rTree.GetEntries(), self.name ) )
      self.rFile.Write()
      self.rFile.Close()
      
def format_ntuple( inputs, output, trans_var, weight = None ):
  sampleDir = config.sampleDir[ args.year ]
  if ( args.JECup or args.JECdown ) and "data" in output:
    print( "[WARNING] Ignoring JECup and/or JECdown arguments for data" )
  elif args.JECup and not args.JECdown:
    print( "[INFO] Running with JECup samples" )
    sampleDir = sampleDir.replace( "nominal", "JECup" )
    output = output.replace( "mc", "mc_JECup" )
  elif args.JECdown and not args.JECup:
    print( "[INFO] Running with JECdown samples" )
    sampleDir = sampleDir.replace( "nominal", "JECdown" )
    output = output.replace( "mc", "mc_JECdown" )
  elif args.JECdown and args.JECup:
    sys.exit( "[WARNING] Cannot run with both JECup and JECdown options. Select only one or none. Quitting..." )
  
  ntuple = ToyTree( output, trans_var )
  for input in inputs:
    print( ">> Processing {}".format( input.split( "/" )[-1] ) )
    if args.location == "LPC":
      rPath = os.path.join( config.sourceDir[ "LPC" ], sampleDir, input )
    elif args.location == "BRUX":
      rPath = os.path.join( config.sourceDir[ "BRUX" ].replace( "/isilon/hadoop", "" ), sampleDir, input )
    rDF = ROOT.RDataFrame( "ljmet", rPath )
    sample_total = rDF.Count().GetValue()
    filter_string = ""
    for variable in ntuple.selection: 
      for i in range( len( ntuple.selection[ variable ][ "CONDITION" ] ) ):
        if filter_string == "": 
          filter_string += "( {} {} {} ) ".format( variable, ntuple.selection[ variable ][ "CONDITION" ][i], ntuple.selection[ variable ][ "VALUE" ][i] )
        else:
          filter_string += "&& ( {} {} {} ) ".format( variable, ntuple.selection[ variable ][ "CONDITION" ][i], ntuple.selection[ variable ][ "VALUE" ][i] )
    rDF_filter = rDF.Filter( filter_string )
    rDF_weight = rDF_filter.Define( "xsecWeight", "compute_weight( triggerXSF, triggerSF, pileupWeight, lepIdSF, isoSF, L1NonPrefiringProb_CommonCalc, MCWeight_MultiLepCalc, xsecEff, tthfWeight, btagCSVWeight, btagCSVRenormWeight )" )
    sample_pass = rDF_filter.Count().GetValue()
    dict_filter = rDF_weight.AsNumpy( columns = list( ntuple.variables.keys() ) )
    del rDF, rDF_filter, rDF_weight
    n_inc = int( sample_pass * float( args.pEvents ) / 100. )
 
    for n in tqdm.tqdm( range( n_inc ) ):
      event_data = {}
      for variable in dict_filter:
        if str( variable ) != "xsecWeight":
          event_data[ variable ] = dict_filter[ variable ][n] 
      
      event_data[ "xsecWeight" ] = 1
      if weight is not None:
        event_data[ "xsecWeight" ] = dict_filter[ "xsecWeight" ][n]

      ntuple.Fill( event_data )

    print( ">> {}/{} events saved...".format( n_inc, sample_total ) )
  ntuple.Write()
  
if args.doMC:
  format_ntuple( inputs = config.samples_input[ args.year ][ "MC" ], output = args.sOut + "_mc" , weight = weight_ttbar, trans_var = args.variables )
if args.doData:
  format_ntuple( inputs = config.samples_input[ args.year ][ "DATA" ], output = args.tOut + "_data" , weight = None, trans_var = args.variables )
                                                                                                
