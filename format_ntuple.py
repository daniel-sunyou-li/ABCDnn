# this script takes an existing step3 ROOT file and formats it for use in the ABCDnn training script
# formats three types of samples: data (no weights), major MC background (no weights), and minor MC backgrounds (weights)
# last modified April 11, 2023 by Daniel Li

import os, sys, ROOT
from array import array
from argparse import ArgumentParser
import config
import tqdm
import xsec

parser = ArgumentParser()
parser.add_argument( "-y",  "--year", default = "2017", help = "Year for sample" )
parser.add_argument( "-v",  "--variables", nargs = "+", default = [ "AK4HT", "DNN_1to40_3t" ], help = "Variables to transform" )
parser.add_argument( "-p",  "--pEvents", default = 100, help = "Percent of events (0 to 100) to include from each file." )
parser.add_argument( "-l",  "--location", default = "BRUX", help = "Location of input ROOT files: LPC,BRUX" )
parser.add_argument( "--doMajorMC", action = "store_true", help = "Major MC background to be weighted using ABCDnn" )
parser.add_argument( "--doMinorMC", action = "store_true", help = "Minor MC background to be weighted using traditional SF" )
parser.add_argument( "--doClosureMC", action = "store_true", help = "Closure MC background weighted using traditional SF" )
parser.add_argument( "--doData", action = "store_true" )
parser.add_argument( "--JECup", action = "store_true", help = "Create an MC dataset using the JECup shift for ttbar" )
parser.add_argument( "--JECdown", action = "store_true", help = "Create an MC dataset using the JECdown shift for ttbar" )
args = parser.parse_args()

if args.location not in [ "LPC", "BRUX" ]: quit( "[ERR] Invalid -l (--location) argument used. Quitting..." )

print( "[INFO] Evaluating cross section weights." )
weightXSec = {}
if args.doMinorMC:
  files = config.samples_input[ args.year ][ "MINOR MC" ] 
elif args.doMajorMC: 
  files = config.samples_input[ args.year ][ "MAJOR MC" ]
elif args.doClosureMC:
  files = config.samples_input[ args.year ][ "CLOSURE" ] 
elif args.doData:
  files = config.samples_input[ args.year ][ "DATA" ]
else:
  quit( "[ERR] No valid option used, choose from doMajorMC, doMinorMC, doClosureMC, doData" )
for f in files:
  if args.doMinorMC or args.doMajorMC or args.doClosureMC:
    nMC = 0
    if "TTToSemiLeptonic" in f and "HT0Njet0_ttjj" in f:
      rF_ = ROOT.TFile.Open( os.path.join( config.sourceDir[ "BRUX" ], config.sampleDir[ args.year ], "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_tt1b_hadd.root" ).replace( "step3", "step1hadds" ) )
      rT_ = rF_.Get( "NumTrueHist" ).Clone( "NumTrueHist" )
      nMC = rT_.Integral()
    elif "TTTT" in f and args.year == "2018":
      for i in range(1,4):
        rF_ = ROOT.TFile.Open( os.path.join( config.sourceDir[ "BRUX" ], config.sampleDir[ args.year ], "TTTT_TuneCP5_13TeV-amcatnlo-pythia8_{}_hadd.root".format(i) ).replace( "step3", "step1hadds" ) )
        rT_ = rF_.Get( "NumTrueHist" ).Clone( "NumTrueHist" )
        nMC += rT_.Integral()
    else:
      rF_ = ROOT.TFile.Open( os.path.join( config.sourceDir[ "BRUX" ], config.sampleDir[ args.year ], f ).replace( "step3", "step1hadds" ) )
      rT_ = rF_.Get( "NumTrueHist" ).Clone( "NumTrueHist" )
      nMC = rT_.Integral()
    weightXSec[f] = xsec.lumi[ args.year ] * xsec.xsec[f] / nMC
    print( ">> {}: {:.1f} x {:.5f} = {:.1f}".format( f.split("_TuneCP5_")[0], nMC, weightXSec[f], xsec.lumi[ args.year ] * xsec.xsec[f] ) )
    rF_.Close()
  else:
    weightXSec[f] = 1.

  
ROOT.gInterpreter.Declare("""
    float compute_weight( float scale, float triggerXSF, float triggerSF, float pileupWeight, float lepIdSF, float EGammaGsfSF, float isoSF, float L1NonPrefiringProb_CommonCalc, float MCWeight_MultiLepCalc, float xsecEff, float btagDeepJetWeight, float btagDeepJet2DWeight_HTnj ){
      return scale * triggerXSF * triggerSF * pileupWeight * lepIdSF * EGammaGsfSF * isoSF * L1NonPrefiringProb_CommonCalc * ( MCWeight_MultiLepCalc / abs( MCWeight_MultiLepCalc ) ) * xsecEff * btagDeepJetWeight * btagDeepJet2DWeight_HTnj;
  }
""")

ROOT.gInterpreter.Declare("""
    float compute_weight_ttbar( float scale, float topPtWeight13TeV, float triggerXSF, float triggerSF, float pileupWeight, float lepIdSF, float EGammaGsfSF, float isoSF, float L1NonPrefiringProb_CommonCalc, float MCWeight_MultiLepCalc, float xsecEff, float btagDeepJetWeight, float btagDeepJet2DWeight_HTnj ){
      return scale * topPtWeight13TeV * triggerXSF * triggerSF * pileupWeight * lepIdSF * EGammaGsfSF * isoSF * L1NonPrefiringProb_CommonCalc * ( MCWeight_MultiLepCalc / abs( MCWeight_MultiLepCalc ) ) * xsecEff * btagDeepJetWeight * btagDeepJet2DWeight_HTnj;
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
  for f in inputs:
    print( ">> Processing {}".format( f ) )
    if args.location == "LPC":
      rPath = os.path.join( config.sourceDir[ "LPC" ], sampleDir, f )
    elif args.location == "BRUX":
      rPath = os.path.join( config.sourceDir[ "BRUX" ].replace( "/isilon/hadoop", "" ), sampleDir, f )
    rDF = ROOT.RDataFrame( "ljmet", rPath )
    sample_total = rDF.Count().GetValue()
    filter_string = "" 
    scale = 1. / ( int( args.pEvents ) / 100. ) # isTraining == 3 is 20% of the total dataset
    for variable in ntuple.selection: 
      for i in range( len( ntuple.selection[ variable ][ "CONDITION" ] ) ):
        if filter_string == "": 
          filter_string += "( {} {} {} ) ".format( variable, ntuple.selection[ variable ][ "CONDITION" ][i], ntuple.selection[ variable ][ "VALUE" ][i] )
        else:
          filter_string += "&& ( {} {} {} ) ".format( variable, ntuple.selection[ variable ][ "CONDITION" ][i], ntuple.selection[ variable ][ "VALUE" ][i] )
    if args.year == "2018" and args.doData:
      filter_string += " && ( leptonEta_MultiLepCalc > -1.3 || ( leptonPhi_MultiLepCalc < -1.57 || leptonPhi_MultiLepCalc > -0.87 ) )"
    rDF_filter = rDF.Filter( filter_string )
    if f.startswith( "TTTo" ):
      rDF_weight = rDF_filter.Define( "xsecWeight", "compute_weight_ttbar( {}, topPtWeight13TeV, triggerXSF, triggerSF, pileupWeight, lepIdSF, EGammaGsfSF, isoSF, L1NonPrefiringProb_CommonCalc, MCWeight_MultiLepCalc, {}, btagDeepJetWeight, btagDeepJet2DWeight_HTnj )".format(
        scale, weight[f]
      ) )
    else:
      rDF_weight = rDF_filter.Define( "xsecWeight", "compute_weight( {}, triggerXSF, triggerSF, pileupWeight, lepIdSF, EGammaGsfSF, isoSF, L1NonPrefiringProb_CommonCalc, MCWeight_MultiLepCalc, {}, btagDeepJetWeight, btagDeepJet2DWeight_HTnj )".format(
        scale, weight[f]
      ) )
    sample_pass = rDF_filter.Count().GetValue()
    dict_filter = rDF_weight.AsNumpy( columns = list( ntuple.variables.keys() + [ "xsecWeight" ] ) )
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
  
if args.doMajorMC:
  format_ntuple( inputs = config.samples_input[ args.year ][ "MAJOR MC" ], output =  "source_" + args.year + "_mc_p" + args.pEvents, weight = weightXSec, trans_var = args.variables )
elif args.doMinorMC:
  format_ntuple( inputs = config.samples_input[ args.year ][ "MINOR MC" ], output = "minor_" + args.year + "_mc_p" + args.pEvents, weight = weightXSec, trans_var = args.variables )
elif args.doClosureMC:
  format_ntuple( inputs = config.samples_input[ args.year ][ "CLOSURE" ], output =  "closure_" + args.year + "_mc_p" + args.pEvents, weight = weightXSec, trans_var = args.variables )
if args.doData:
  format_ntuple( inputs = config.samples_input[ args.year ][ "DATA" ], output =  "target_" + args.year + "_data_p" + args.pEvents, weight = weightXSec, trans_var = args.variables )
