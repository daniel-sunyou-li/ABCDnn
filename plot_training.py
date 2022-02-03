# this script is run on the condor node for applying the trained ABCDnn model to ttbar samples
# last updated 11/15/2021 by Daniel Li

import numpy as np
import os
import imageio
import uproot
import abcdnn
from argparse import ArgumentParser
from json import loads as load_json
from array import array
import ROOT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import config

parser = ArgumentParser()
parser.add_argument( "-s", "--source", required = True )
parser.add_argument( "-t", "--target", required = True )
parser.add_argument( "-m", "--tag", required = True )

args = parser.parse_args()

folder = "Results/"
folder_contents = os.listdir( folder )

def get_region( x, y ):
  x_region = np.linspace( config.regions[ "X" ][ "MIN" ], config.regions[ "X" ][ "MAX" ], config.regions[ "X" ][ "MAX" ] - config.regions[ "X" ][ "MIN" ] + 1 )
  y_region = np.linspace( config.regions[ "Y" ][ "MIN" ], config.regions[ "Y" ][ "MAX" ], config.regions[ "Y" ][ "MAX" ] - config.regions[ "Y" ][ "MIN" ] + 1 )
  if x == x_region[0] and y == y_region[0]: return "X"
  elif x == x_region[0] and y >= y_region[1]: return "Y"
  elif x == x_region[1] and y == y_region[0]: return "A"
  elif x == x_region[1] and y >= y_region[1]: return "C"
  elif x >= x_region[2] and y == y_region[0]: return "B"
  return "D"

# load in json file
print( ">> Reading in {}.json for hyper parameters...".format( args.tag ) )
with open( os.path.join( folder, args.tag + ".json" ), "r" ) as f:
  params = load_json( f.read() )

print( ">> Setting up NAF model..." )

print( ">> Loading checkpoint weights from {} with tag: {}".format( folder, args.tag ) )
checkpoints = [ name.split( "." )[0] for name in folder_contents if ( "EPOCH" in name and args.tag in name and name.endswith( "index" ) ) ]

print( ">> Load the data" )
sFile = uproot.open( args.source )
tFile = uproot.open( args.target )
sTree = sFile[ "Events" ]
tTree = tFile[ "Events" ]

variables = []
v_in = []
categorical = []
lowerlimit = []
upperlimit = []
for variable in sorted( list( config.variables.keys() ) ):
  if config.variables[ variable ][ "TRANSFORM" ] == True: v_in.append( variable )
  variables.append( variable )
  categorical.append( config.variables[ variable ][ "CATEGORICAL" ] )
  upperlimit.append( config.variables[ variable ][ "LIMIT" ][1] )
  lowerlimit.append( config.variables[ variable ][ "LIMIT" ][0] )

_onehotencoder = abcdnn.OneHotEncoder_int( categorical, lowerlimit = lowerlimit, upperlimit = upperlimit )
print( ">> Found {} variables: ".format( len( variables ) ) )
for variable in sorted( variables ):
  print( "  + {}".format( variable ) )

variables_transform = [ variable for variable in config.variables.keys() if config.variables[ variable ][ "TRANSFORM" ] ]

inputs_src = sTree.pandas.df( variables )
inputs_src_enc = _onehotencoder.encode( inputs_src.to_numpy( dtype = np.float32 ) )

print( ">> Applying normalization to source inputs and splitting into regions" )
inputmeans = np.hstack( [ float( mean ) for mean in params[ "INPUTMEANS" ] ] )
inputsigma = np.hstack( [ float( sigma ) for sigma in params[ "INPUTSIGMAS" ] ] )
normedinputs_src = ( inputs_src_enc[:,:-1] - inputmeans ) / inputsigma

x_range = np.linspace( config.regions[ "X" ][ "MIN" ], config.regions[ "X" ][ "MAX" ], config.regions[ "X" ][ "MAX" ] - config.regions[ "X" ][ "MIN" ] + 1 )
y_range = np.linspace( config.regions[ "Y" ][ "MIN" ], config.regions[ "Y" ][ "MAX" ], config.regions[ "Y" ][ "MAX" ] - config.regions[ "Y" ][ "MIN" ] + 1 )
inputs_src_region = { region: [] for region in [ "X", "Y", "A", "B", "C", "D" ] }
inputs_mc_region = { region: [] for region in [ "X", "Y", "A", "B", "C", "D" ] }
print( ">> Found {} total source entries".format( inputs_src.shape[0] ) )
for i in range( inputs_src.shape[0] ):
  var_x = inputs_src.iloc[i][ config.regions[ "X" ][ "VARIABLE" ] ]
  var_y = inputs_src.iloc[i][ config.regions[ "Y" ][ "VARIABLE" ] ]
  if var_x > max(x_range) and not config.regions[ "X" ][ "INCLUSIVE" ]: continue
  if var_y > max(y_range) and not config.regions[ "Y" ][ "INCLUSIVE" ]: continue 
  inputs_src_region[ get_region( var_x, var_y ) ].append( normedinputs_src[i] )
  inputs_mc_region[ get_region( var_x, var_y ) ].append( [ 
    inputs_src.iloc[i][ variables_transform ][1], 
    inputs_src.iloc[i][ variables_transform ][0]
  ] )

inputs_tgt = tTree.pandas.df( variables ) 
inputs_tgt_region = { region: [] for region in [ "X", "Y", "A", "B", "C", "D" ] }
print( ">> Found {} total target entries".format( inputs_tgt.shape[0] ) )
for i in range( inputs_tgt.shape[0] ):
  var_x = inputs_tgt.iloc[i][ config.regions[ "X" ][ "VARIABLE" ] ]
  var_y = inputs_tgt.iloc[i][ config.regions[ "Y" ][ "VARIABLE" ] ]
  if var_x > max(x_range) and not config.regions[ "X" ][ "INCLUSIVE" ]: continue
  if var_y > max(y_range) and not config.regions[ "Y" ][ "INCLUSIVE" ]: continue
  inputs_tgt_region[ get_region( var_x, var_y ) ].append( [
    inputs_tgt.iloc[i][ variables_transform ][1],
    inputs_tgt.iloc[i][ variables_transform ][0]
  ] )

for region in inputs_src_region:
  print( ">> Region {}: MC = {}, DATA = {}".format( region, len( inputs_src_region[ region ] ), len( inputs_tgt_region[ region ] ) ) )

print( ">> Processing checkpoints" )
predictions = {}
for i, checkpoint in enumerate( sorted( checkpoints ) ):
  epoch = checkpoint.split( "EPOCH" )[1]
  NAF = abcdnn.NAF( 
    inputdim = params["INPUTDIM"],
    conddim = params["CONDDIM"],
    activation = params["ACTIVATION"], 
    regularizer = params["REGULARIZER"],
    nodes_cond = params["NODES_COND"],
    hidden_cond = params["HIDDEN_COND"],
    nodes_trans = params["NODES_TRANS"],
    depth = params["DEPTH"],
    permute = True
  )
  NAF.load_weights( os.path.join( folder, checkpoint ) )
  print( "  + {}".format( checkpoint ) )
  
  predictions[ int( epoch ) ] = { region: [] for region in [ "X", "Y", "A", "B", "C", "D" ] }
  for region in predictions[ int( epoch ) ]:
    for prediction in NAF.predict( np.asarray( inputs_src_region[ region ] ) ):
      predictions[ int( epoch ) ][ region ].append( [
        prediction[0] * inputsigma[0] + inputmeans[0],
        prediction[1] * inputsigma[1] + inputmeans[1]
      ] )
  

print( ">> Generating images of plots" )
region_key = {
  0: {
    0: "X",
    1: "Y"
  },
  1: {
    0: "A",
    1: "B" 
  },
  2:{
    0: "B",
    1: "D"
  }
}
images = { variable: [] for variable in variables_transform }

def ratio_err( x, xerr, y, yerr ):
  return np.sqrt( ( yerr * x / y**2 )**2 + ( xerr / y )**2 )

def plot_hist( ax, variable, x, y, epoch, mc_pred, mc_true, data, bins ):
  mc_pred_hist = { 
    "TRUE": np.histogram( mc_pred, bins = bins, density = False ),
    "NORM": np.histogram( mc_pred, bins = bins, density = True )
  }
  mc_pred_scale = mc_pred_hist[ "TRUE" ][0] / mc_pred_hist[ "NORM" ][0]
  
  mc_true_hist = {
    "TRUE": np.histogram( mc_true, bins = bins, density = False ),
    "NORM": np.histogram( mc_true, bins = bins, density = True )
  }
  mc_true_scale = mc_true_hist[ "TRUE" ][0] / mc_true_hist[ "NORM" ][0]

  data_hist = {
    "TRUE": np.histogram( data, bins = bins, density = False ),
    "NORM": np.histogram( data, bins = bins, density = True )
  }
  data_scale = data_hist[ "TRUE" ][0] / data_hist[ "NORM" ][0]
  
  # plot the data first
  ax.errorbar(
    0.5 * ( data_hist[ "NORM" ][1][1:] + data_hist[ "NORM" ][1][:-1] ),
    data_hist[ "NORM" ][0], yerr = np.sqrt( data_hist[ "TRUE" ][0] ) / data_scale,
    label = "Data",
    marker = "o", markersize = 3, markerfacecolor = "black", markeredgecolor = "black",
    elinewidth = 1, ecolor = "black" , capsize = 2, lw = 0 
  )
  # plot the mc
  ax.errorbar(
    0.5 * ( mc_true_hist[ "NORM" ][1][1:] + mc_true_hist[ "NORM" ][1][:-1] ),
    mc_true_hist[ "NORM" ][0], yerr = np.sqrt( mc_true_hist[ "TRUE" ][0] ) / mc_true_scale,
    label = "MC",
    marker = ",", drawstyle = "steps-mid", lw = 2, alpha = 0.7, color = "green"
  )
  # plot the predicted
  ax.hist(
    mc_pred, bins = bins, density = True,
    label = "ABCDnn",
    facecolor = "red", ec = "black", histtype = "stepfilled", alpha = 0.5,
  )
  ax.fill_between(
    0.5 * ( mc_pred_hist[ "NORM" ][1][1:] + mc_pred_hist[ "NORM" ][1][:-1] ),
    y1 = mc_pred_hist[ "NORM" ][0] + np.sqrt( mc_pred_hist[ "TRUE" ][0] ) / mc_pred_scale,
    y2 = mc_pred_hist[ "NORM" ][0] - np.sqrt( mc_pred_hist[ "TRUE" ][0] ) / mc_pred_scale,
    interpolate = False, step = "mid",
    color = "red", alpha = 0.2
  )
  
  ax.set_xlim( config.variables[ variable ][ "LIMIT" ][0], config.variables[ variable ][ "LIMIT" ][1] )
  y_max = max( [ max( data_hist[ "NORM" ][0] ), max( mc_true_hist[ "NORM" ][0] ) ] )
  ax.set_ylim( 0, 1.3 * y_max )
  ax.axes.yaxis.set_visible(False)
  ax.axes.xaxis.set_visible(False)

  x_val = np.linspace( config.regions[ "X" ][ "MIN" ], config.regions[ "X" ][ "MAX" ], config.regions[ "X" ][ "MAX" ] - config.regions[ "X" ][ "MIN" ] + 1 )[x]
  y_val = np.linspace( config.regions[ "Y" ][ "MIN" ], config.regions[ "Y" ][ "MAX" ], config.regions[ "Y" ][ "MAX" ] - config.regions[ "Y" ][ "MIN" ] + 1 )[y]
  title_text = "$"
  if config.regions[ "X" ][ "INCLUSIVE" ] and x_val == config.regions[ "X" ][ "MAX" ]: 
    title_text += "{}\geq {}, ".format( config.variables[ config.regions[ "X" ][ "VARIABLE" ] ][ "LATEX" ], x_val ) 
  else:
    title_text += "{}={}, ".format( config.variables[ config.regions[ "X" ][ "VARIABLE" ] ][ "LATEX" ], x_val ) 
  if config.regions[ "Y" ][ "INCLUSIVE" ] and y_val == config.regions[ "Y" ][ "MAX" ]:
    title_text += "{}\geq {}".format( config.variables[ config.regions[ "Y" ][ "VARIABLE" ] ][ "LATEX" ], y_val )
  else:
    title_text += "{}={}".format( config.variables[ config.regions[ "Y" ][ "VARIABLE" ] ][ "LATEX" ], y_val ) 
  title_text += "$"
  ax.set_title( "Region {}: {}".format( get_region( x_val, y_val ), title_text ), ha = "right", x = 1.0, fontsize = 10 )
  ax.text(
    0.02, 0.95, str(epoch),
    ha = "left", va = "top", transform = ax.transAxes
  )
  ax.legend( loc = "upper right" ,fontsize = 8 )


def plot_ratio( ax, variable, x, y, mc_pred, mc_true, data, bins):
  mc_pred_hist = {
    "TRUE": np.histogram( mc_pred, bins = bins, density = False ),
    "NORM": np.histogram( mc_pred, bins = bins, density = True )
  }
  mc_pred_scale = mc_pred_hist[ "TRUE" ][0] / mc_pred_hist[ "NORM" ][0]

  mc_true_hist = {                                                                                                                        "TRUE": np.histogram( mc_true, bins = bins, density = False ),                                                                        "NORM": np.histogram( mc_true, bins = bins, density = True )
  }
  mc_true_scale = mc_true_hist[ "TRUE" ][0] / mc_true_hist[ "NORM" ][0]

  data_hist = {
    "TRUE": np.histogram( data, bins = bins, density = False ),
    "NORM": np.histogram( data, bins = bins, density = True )
  }
  data_scale = data_hist[ "TRUE" ][0] / data_hist[ "NORM" ][0]  
  
  ratio = []
  ratio_std = []
  for i in range( len( data_hist[ "TRUE" ][0] ) ):
    if data_hist[ "TRUE" ][0][i] == 0 or mc_pred_hist[ "TRUE" ][0][i] == 0: 
      ratio.append(0)
      ratio_std.append(0)
    else:
      ratio.append( ( mc_pred_hist[ "TRUE" ][0][i] * mc_pred_scale[i] ) / ( data_hist[ "TRUE" ][0][i] * data_scale[i] ) )
      ratio_std.append( ratio_err(
        mc_pred_hist[ "TRUE" ][0][i],
        np.sqrt( mc_pred_hist[ "TRUE" ][0][i] ),
        data_hist[ "TRUE" ][0][i],
        np.sqrt( data_hist[ "TRUE" ][0][i] )
      ) * ( data_scale[i] / mc_pred_scale[i] ) )

  ax.scatter(
    0.5 * ( data_hist[ "NORM" ][1][1:] + data_hist[ "NORM" ][1][:-1] ),
    ratio, 
    linewidth = 0, marker = "o", 
    color = "black", zorder = 3
  )
  ax.fill_between(
    0.5 * ( data_hist[ "NORM" ][1][1:] + data_hist[ "NORM" ][1][:-1] ),
    y1 = np.array( ratio ) + np.array( ratio_std ),
    y2 = np.array( ratio ) - np.array( ratio_std ),
    interpolate = False, step = "mid",
    color = "gray", alpha = 0.2
  )
  ax.axhline(
    y = 1.0, color = "r", linestyle = "-", zorder = 1
  )

  ax.set_xlabel( "${}$".format( config.variables[ variable ][ "LATEX" ] ), ha = "right", x = 1.0, fontsize = 10 )
  ax.set_xlim( config.variables[ variable ][ "LIMIT" ][0], config.variables[ variable ][ "LIMIT" ][1] )
  ax.set_ylabel( "ABCDnn/Data", ha = "right", y = 1.0, fontsize = 8 )
  ax.set_ylim( 0, 2 )
  ax.set_yticks( [ 0.5, 1.0, 1.5 ] )
  ax.tick_params( axis = "both", labelsize = 8 )
  if x != 2: ax.axes.xaxis.set_visible(False)


for epoch in sorted( predictions.keys() ):
  print( "  + Generating image for epoch {}".format( epoch ) )
  for i, variable in enumerate( sorted( variables_transform ) ): 
    bins = np.linspace( config.variables[ variable ][ "LIMIT" ][0], config.variables[ variable ][ "LIMIT" ][1], 31 )
    fig, axs = plt.subplots( 6, 2, figsize = (9,12), gridspec_kw = { "height_ratios": [3,1,3,1,3,1] } )
    for x in range( 6 ):
      for y in range( 2 ):
        if x % 2 == 0: 
          plot_hist( 
            ax = axs[x,y], 
            variable = variable,
            x = int( x / 2 ), y = y,
            epoch = epoch,
            mc_pred = np.asarray( predictions[ epoch ][ region_key[int(x/2)][y] ] )[:,i],
            mc_true = np.asarray( inputs_mc_region[ region_key[int(x/2)][y] ] )[:,i],
            data = np.asarray( inputs_tgt_region[ region_key[int(x/2)][y] ] )[:,i],
            bins = bins
          )
        else: 
          plot_ratio( 
            ax = axs[x,y],
            variable = variable,
            x = int((x-1)/2), y = y, 
            mc_pred = np.asarray( predictions[ epoch ][ region_key[int((x-1)/2)][y] ] )[:,i],
            mc_true = np.asarray( inputs_mc_region[ region_key[int((x-1)/2)][y] ] )[:,i],
            data = np.asarray( inputs_tgt_region[ region_key[int((x-1)/2)][y] ] )[:,i],
            bins = bins 
          )
          position_old = axs[x,y].get_position()
          position_new = axs[x-1,y].get_position()
          points_old = position_old.get_points()
          points_new = position_new.get_points()
          points_old[1][1] = points_new[0][1]
          position_old.set_points( points_old )
          axs[x,y].set_position( position_old )

    plt.savefig( "Results/{}_{}_EPOCH{}.png".format( args.tag, variable, epoch ) )
    images[ variable ].append( imageio.imread( "Results/{}_{}_EPOCH{}.png".format( args.tag, variable, epoch ) ) )
    del fig, axs

print( "[DONE] {} training GIF completed: {}_{}.gif, {}_{}.gif".format( args.tag, args.tag, variables_transform[0], args.tag, variables_transform[1] ) )
for variable in variables_transform:
  imageio.mimsave( "Results/{}_{}.gif".format( args.tag, variable ), images[ variable ], duration = 1 )
  for epoch in sorted( predictions.keys() ):
    if epoch != sorted( predictions.keys() )[-1]:
      os.system( "rm Results/{}_{}_EPOCH{}.png".format( args.tag, variable, epoch ) )
