import os
import numpy as np

data_path = os.path.join( os.getcwd() )
results_path = os.path.join( os.getcwd(), "Results" )
eosUserName = "dali"
postfix = "3t"

condorDir = "root://cmseos.fnal.gov//store/user/{}/".format( eosUserName ) 

sourceDir = {
  "LPC": "root://cmseos.fnal.gov//store/group/{}/".format( "lpcljm" ),
  "BRUX": "root://brux30.hep.brown.edu:1094//store/user/{}/".format( eosUserName )
}

targetDir = {
  "LPC": "root://cmseos.fnal.gov//store/group/lpcljm/",
  "BRUX": "root://brux30.hep.brown.edu:1094//store/user/{}/".format( eosUserName )
}

sampleDir = {
  year: "FWLJMET106XUL_singleLep{}UL_RunIISummer20_{}_step3/nominal/".format( year, postfix ) for year in [ "2016APV", "2016", "2017", "2018" ]
}

variables = {
  "thirdcsvb_bb": {
    "CATEGORICAL": False,
    "TRANSFORM": True,
    "LIMIT": [0.,1.],
    "LATEX": "DeepJet\ 3"
  },
  #"minDR_lepBJet": {
  #  "CATEGORICAL": False,
  #  "TRANSFORM": True,
  #  "LIMIT": [0.,5.],
  #  "LATEX": "DNN"
  #},
  #"AK4HT": {
  #  "CATEGORICAL": False,
  #  "TRANSFORM": True,
  #  "LIMIT": [0.,3000.],
  #  "LATEX": "H_T\ (GeV)"
  #},
  "DNN_1to40_3t": {
    "CATEGORICAL": False,
    "TRANSFORM": True,
    "LIMIT": [0.,1.],
    "LATEX": "DNN"
  },
  "NJetsCSV_JetSubCalc": {
    "CATEGORICAL": True,
    "TRANSFORM": False,
    "LIMIT": [0,2],
    "LATEX": "N_b"
  },
  "NJets_JetSubCalc": {
    "CATEGORICAL": True,
    "TRANSFORM": False,
    "LIMIT": [4,5],
    "LATEX": "N_j"
  },
}

selection = { # edit these accordingly
  "NJets_JetSubCalc": { "VALUE": [ 4 ], "CONDITION": [ ">=" ] },
  "NJetsCSV_JetSubCalc": { "VALUE": [ 0 ], "CONDITION": [ ">=" ] },
  "corr_met_MultiLepCalc": { "VALUE": [ 20. ], "CONDITION": [ ">" ] },
  "MT_lepMet": { "VALUE": [ 0 ], "CONDITION": [ ">" ] },
  "minDR_lepJet": { "VALUE": [ 0.4 ], "CONDITION": [ ">" ] },
  "AK4HT": { "VALUE": [ 350. ], "CONDITION": [ ">" ] },
  "DataPastTriggerX": { "VALUE": [ 1 ], "CONDITION": [ "==" ] },
  "MCPastTriggerX": { "VALUE": [ 1 ], "CONDITION": [ "==" ] },
  #"DNN_1to40_3t": { "VALUE": [ 0. ], "CONDITION": [ ">" ] },
  #"isTraining": { "VALUE": [ "1" ], "CONDITION": [ "==" ] },
}

regions = {
  "X": {
    "VARIABLE": "NJetsCSV_JetSubCalc",
    "INCLUSIVE": True,
    "MIN": 0,
    "MAX": 2,
    "SIGNAL": 2
  },
  "Y": {
    "VARIABLE": "NJets_JetSubCalc",
    "INCLUSIVE": True,
    "MIN": 4,
    "MAX": 5,
    "SIGNAL": 5
  }
}

params = {
  "EVENTS": {
    "MCWEIGHT": None
  },
  "MODEL": { # parameters for setting up the NAF model
    "NODES_COND": 16,
    "HIDDEN_COND": 1,
    "NODES_TRANS": 8,
    "LRATE": 5e-3,
    "DECAY": 0.1,
    "GAP": 500,
    "DEPTH": 1,
    "REGULARIZER": "ALL", # DROPOUT, BATCHNORM, ALL, NONE
    "INITIALIZER": "RandomNormal", # he_normal, RandomNormal
    "ACTIVATION": "softplus", # softplus, relu, swish
    "BETA1": 0.98,
    "BETA2": 0.999,
    "MMD SIGMAS": [1.0],
    "MMD WEIGHTS": None,
    "MINIBATCH": 2**11,
    "RETRAIN": True,
    "PERMUTE": False,
    "SEED": 101, # this can be overridden when running train_abcdnn.py
    "SAVEDIR": "./Results/",
    "VERBOSE": False  
  },
  "TRAIN": {
    "EPOCHS": 2500,
    "PATIENCE": 100000,
    "MONITOR": 100,
    "MONITOR THRESHOLD": 1500,  # only save model past this epoch
    "PERIODIC SAVE": True,   # saves model at each epoch step according to "MONITOR" 
    "SHOWLOSS": True,
    "EARLY STOP": False,      # early stop if validation loss begins diverging
  },
  "PLOT": {
    "RATIO": [ 0.25, 2.0 ], # y limits for the ratio plot
    "YSCALE": "linear",   # which y-scale plots to produce
    "NBINS": 31,            # histogram x-bins
    "ERRORBARS": True,      # include errorbars on hist
    "NORMED": True,         # normalize histogram counts/density
    "SAVE": False,          # save the plots as png
    "PLOT_KS": True,        # include the KS p-value in plots
  }
}
        
hyper = {
  "OPTIMIZE": {
    "NODES_COND": ( [8,16,32,64,128], "CAT" ),
    "HIDDEN_COND": ( [1,4], "INT" ),
    "NODES_TRANS": ( [1,8,16,32,64,128], "CAT" ),
    "LRATE": ( [1e-5,1e-4,1e-3], "CAT" ),
    "DECAY": ( [1,1e-1,1e-2], "CAT" ),
    "GAP": ( [100,500,1000,5000], "CAT" ),
    "DEPTH": ( [1,4], "INT" ),
    "REGULARIZER": ( ["L1","L2","L1+L2","None"], "CAT" ),
    "ACTIVATION": ( ["swish","relu","elu","softplus"], "CAT" ),
    "BETA1": ( [0.90,0.99,0.999], "CAT" ),
    "BETA2": ( [0.90,0.99,0.999], "CAT" ),
    "SIGMA": ( [0.05,0.35], "FLOAT" )
  },
  "PARAMS": {
    "PATIENCE": 10000,
    "EPOCHS": 2000,
    "N_RANDOM": 20,
    "N_CALLS": 30,
    "MINIBATCH": 2**12,
    "VERBOSE": True
  }
}

samples_input = {
  "2016APV": {
    "DATA": [
"SingleElectron_hadd.root",
"SingleMuon_hadd.root",
    ],
    "MC": [
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_tt1b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_tt2b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttbb_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttcc_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_1_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_2_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_3_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_4_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_5_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_6_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_7_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_8_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_9_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_10_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_tt1b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_tt2b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttbb_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttcc_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttjj_hadd.root",
    ]
  },
  "2016": {
    "DATA": [
"SingleElectron_hadd.root",
"SingleMuon_hadd.root",
    ],
    "MC": [
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_tt1b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_tt2b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttbb_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttcc_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_1_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_2_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_3_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_4_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_5_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_6_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_7_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_8_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_9_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_10_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_tt1b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_tt2b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttbb_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttcc_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttjj_hadd.root",
    ]
  },
  "2017": {
    "DATA": [
      "SingleElectron_hadd.root",
      "SingleMuon_hadd.root",
    ],
    "MC": [
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_tt1b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_tt2b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttbb_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttcc_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_1_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_2_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_3_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_4_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_5_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_6_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_7_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_8_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_9_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_10_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_tt1b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_tt2b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttbb_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttcc_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttjj_hadd.root",
    ]
  },
  "2018": {
    "DATA": [
"EGamma_hadd.root",
"SingleMuon_hadd.root",
    ],
    "MC": [
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_tt1b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_tt2b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttbb_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttcc_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_1_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_2_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_3_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_4_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_5_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_6_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_7_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_8_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_9_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_10_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_tt1b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_tt2b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttbb_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttcc_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttjj_hadd.root",
    ]
  }
}

samples_apply = {
  "2016APV": [
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_tt1b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_tt2b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttbb_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttcc_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_1_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_2_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_3_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_4_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_5_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_6_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_7_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_8_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_9_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_10_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_tt1b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_tt2b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttbb_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttcc_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttjj_hadd.root",
  ],
  "2016": [
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_tt1b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_tt2b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttbb_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttcc_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_1_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_2_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_3_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_4_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_5_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_6_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_7_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_8_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_9_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_10_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_tt1b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_tt2b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttbb_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttcc_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttjj_hadd.root",
  ],
  "2017": [
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_tt1b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_tt2b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttbb_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttcc_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_1_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_2_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_3_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_4_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_5_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_6_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_7_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_8_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_9_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_10_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_tt1b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_tt2b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttbb_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttcc_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttjj_hadd.root",
#"QCD_HT200to300_TuneCP5_13TeV-madgraph-pythia8_hadd.root",
#"QCD_HT300to500_TuneCP5_13TeV-madgraph-pythia8_hadd.root",
#"QCD_HT500to700_TuneCP5_13TeV-madgraph-pythia8_hadd.root",
#"QCD_HT700to1000_TuneCP5_13TeV-madgraph-pythia8_hadd.root",
#"QCD_HT1000to1500_TuneCP5_13TeV-madgraph-pythia8_hadd.root",
#"QCD_HT1500to2000_TuneCP5_13TeV-madgraph-pythia8_hadd.root",
#"QCD_HT2000toInf_TuneCP5_13TeV-madgraph-pythia8_hadd.root"
  ],
  "2018": [
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_tt1b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_tt2b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttbb_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttcc_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_1_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_2_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_3_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_4_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_5_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_6_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_7_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_8_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_9_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_10_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_tt1b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_tt2b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttbb_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttcc_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttjj_hadd.root",
#"QCD_HT200to300_TuneCP5_13TeV-madgraphMLM-pythia8_hadd.root",
#"QCD_HT300to500_TuneCP5_13TeV-madgraphMLM-pythia8_hadd.root",
#"QCD_HT500to700_TuneCP5_13TeV-madgraphMLM-pythia8_hadd.root",
#"QCD_HT700to1000_TuneCP5_13TeV-madgraphMLM-pythia8_hadd.root",
#"QCD_HT1000to1500_TuneCP5_13TeV-madgraphMLM-pythia8_hadd.root",
#"QCD_HT1500to2000_TuneCP5_13TeV-madgraphMLM-pythia8_hadd.root",
#"QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8_hadd.root"
  ]
}
