# this script takes an existing step3 ROOT file and formats it for use in the ABCDnn training script
# last modified September 29, 2021 by Daniel Li

import os, ROOT
from array import array
from argparse import ArgumentParser
import config

parser = ArgumentParser()
parser.add_argument( "-y",  "--year", default = "2017", help = "Year for sample" )
parser.add_argument( "-sO", "--sOut", default = "source_ttbar", help = "Output name of source ROOT file" )
parser.add_argument( "-tO", "--tOut", default = "target_data", help = "Output name of target ROOT file" )
parser.add_argument( "-v",  "--variables", nargs = "+", default = [ "AK4HT", "DNN_5j_1to50_S2B10" ], help = "Variables to transform" )
parser.add_argument( "-p",  "--pEvents", default = 100, help = "Percent of events (0 to 100) to include from each file." )
parser.add_argument( "-l",  "--location", default = "LPC", help = "LPC,BRUX" )
parser.add_argument( "--doMC", action = "store_true" )
parser.add_argument( "--doData", action = "store_true" )
args = parser.parse_args()

if args.location not in [ "LPC", "BRUX" ]: quit( "[ERR] Invalid -l (--location) argument used. Quitting..." )

ttbar_xsec = 831.76
target_lumi = {
  "2016": 35867.,
  "2017": 41530.,
  "2018": 59970.
}
BR = {
  "Hadronic": 0.457,
  "SemiLeptonic": 0.438,
  "2L2Nu": 0.105,
  "HT500Njet9": 0.438 * 0.00617938417682763
}
num_MC = {
  "2016": {
    "Hadronic":     67963984,
    "SemiLeptonic": 106736180,
    "2L2Nu":        67312164,
    "HT500Njet9":   8912888
  },
  "2017": {
    "Hadronic":     233815417,
    "SemiLeptonic": 352212660,
    "2L2Nu":        44580106 + 28515514 + 51439568 + 47012288 + 25495972,
    "HT500Njet9":   10179200
  },
  "2018": {
    "Hadronic":     132368556,
    "SemiLeptonic": 100579948,
    "2L2Nu":        63791474,
    "HT500Njet9":   8398387
  }
}

weight_ttbar = {}
for finalstate in BR:
  weight_ttbar[ finalstate ] = ttbar_xsec * target_lumi[ args.year ] * BR[ finalstate ] / num_MC[ args.year ][ finalstate ]
  
ROOT.gInterpreter.Declare("""
    float compute_weight( float triggerXSF, float pileupWeight, float lepIdSF, float isoSF, float L1NonPrefiringProb_CommonCalc, float MCWeight_MultiLepCalc, float xsecEff, float tthfWeight, float btagCSVWeight, float btagCSVRenormWeight ){
      return 3 * triggerXSF * pileupWeight * lepIdSF * isoSF * L1NonPrefiringProb_CommonCalc * ( MCWeight_MultiLepCalc / abs( MCWeight_MultiLepCalc ) ) * xsecEff * tthfWeight * btagCSVWeight * btagCSVRenormWeight;
  }
""")

class ToyTree:
  def __init__( self, name, trans_var ):
    # trans_var is transforming variables
    self.name = name
    self.rFile = ROOT.TFile( "{}.root".format( name ), "RECREATE" )
    self.rTree = ROOT.TTree( "Events", name )
    self.variables = { # variables that are used regardless of the transformation variables
      "NJets_JetSubCalc": { "ARRAY": array( "i", [0] ), "STRING": "NJets_JetSubCalc/I" },
      "NJetsCSV_MultiLepCalc": { "ARRAY": array( "i", [0] ), "STRING": "NJetsCSV_MultiLepCalc/I" },
      "xsecWeight": { "ARRAY": array( "f", [0] ), "STRING": "xsecWeight/F" }
    }
    
    for variable in trans_var:
      self.variables[ variable ] = { "ARRAY": array( "f", [0] ), "STRING": "{}/F".format( variable ) }
    
    self.selection = { # edit these accordingly
      "NJets_JetSubCalc": { "VALUE": [ 4 ], "CONDITION": [ ">=" ] },
      "NJetsCSV_MultiLepCalc": { "VALUE": [ 2 ], "CONDITION": [ ">=" ] },
      "corr_met_MultiLepCalc": { "VALUE": [ 60. ], "CONDITION": [ ">" ] },
      "MT_lepMet": { "VALUE": [ 60 ], "CONDITION": [ ">" ] },
      "minDR_lepJet": { "VALUE": [ 0.4 ], "CONDITION": [ ">" ] },
      "AK4HT": { "VALUE": [ 500. ], "CONDITION": [ ">" ] },
      "DataPastTriggerX": { "VALUE": [ 1 ], "CONDITION": [ "==" ] },
      "MCPastTriggerX": { "VALUE": [ 1 ], "CONDITION": [ "==" ] },
      "isTraining": { "VALUE": [ "1 || 2" ], "CONDITION": [ "==" ] },
    }
    
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
      
def format_ntuple( output, inputs, trans_var, weight = None ):
  ntuple = ToyTree( output, trans_var )
  for input in inputs:
    print( ">> Processing {}".format( input.split( "/" )[-1] ) )
    if args.location == "LPC":
      rPath = os.path.join( config.sourceDir[ "LPC" ], config.sampleDir[ args.year ], input )
    elif args.location == "BRUX":
      rPath = os.path.join( config.sourceDir[ "BRUX" ], config.sampleDir[ args.year ], input )
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
    rDF_weight = rDF_filter.Define( "xsecWeight", "compute_weight( triggerXSF, pileupWeight, lepIdSF, isoSF, L1NonPrefiringProb_CommonCalc, MCWeight_MultiLepCalc, xsecEff, tthfWeight, btagCSVWeight, btagCSVRenormWeight )" )
    sample_pass = rDF_filter.Count().GetValue()
    dict_filter = rDF_weight.AsNumpy( columns = list( ntuple.variables.keys() ) )
    del rDF, rDF_filter, rDF_weight
    n_inc = int( sample_pass * float( args.pEvents ) / 100. )
 
    for n in range( n_inc ):
      event_data = {}
      for variable in dict_filter:
        if str( variable ) != "xsecWeight":
          event_data[ variable ] = dict_filter[ variable ][n] 
      
      event_data[ "xsecWeight" ] = 1
      if weight is not None:
        event_data[ "xsecWeight" ] = dict_filter[ "xsecWeight" ][n]

      ntuple.Fill( event_data )

    print( ">> {}/{} events passed...".format( n_inc, sample_total ) )
  ntuple.Write()
  
if args.doMC:
  format_ntuple( inputs = config.samples_input[ args.year ][ "MC" ], output = args.sOut + "_mc" , weight = weight_ttbar, trans_var = args.variables )
if args.doData:
  format_ntuple( inputs = config.samples_input[ args.year ][ "DATA" ], output = args.tOut + "_data" , weight = None, trans_var = args.variables )
                                                                                                
