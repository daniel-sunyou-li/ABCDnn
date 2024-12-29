# methods and classes used for constructing the ABCDnn module
# last updated 10/25/2021 by Daniel Li

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as regularizers
import tensorflow_probability as tfp
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
import numpy as np
import os, uproot, pickle
from scipy import stats
from json import dumps as write_json

def invsigmoid( x ):
# inverse sigmoid function for transformer
  #xclip = tf.clip_by_value( x, 1e-6, 1.0 - 1e-6 )
  return tf.math.log( x / ( 1.0 - x ) )

def NAF( inputdim, conddim, activation, regularizer, initializer, nodes_cond, hidden_cond, nodes_trans, depth, permute ):
# neural autoregressive flow is a chain of MLP networks used for the conditioner and transformer parts of the flow
# inputdim = number of transformed inputs (should be 2)
# conddim = conditional categories ( should be 5, 3 for X and 2 for Y control variables )
  keras.constraints.NonNeg()
  activation_key = { # edit this if you add options to config.hyper dict with more activation functions
    "swish": tf.nn.swish,
    "softplus": tf.nn.softplus,
    "elu": tf.nn.elu,
    "relu": tf.nn.relu
  }
  
  xin = layers.Input( shape = ( inputdim + conddim, ), name = "INPUT_LAYER" ) # the expected input data shape should be 7
  xcondin = xin[ :, inputdim: ]   # expected conditional categories layer
  xfeatures = xin[ :, :inputdim ] # expected transformed input layer
  nextfeature = xfeatures
  
  for idepth in range( int( depth ) ):
    if permute: # mix up the input order of the consecutive neural networks in the flow
      randperm = np.random.permutation( inputdim ).astype( "int32" )
      permutation = tf.constant( randperm )
    else:
      permutation = tf.range( inputdim, dtype = "int32" )

    # tfp.bijectors.Permute permutes rightmost dimension of a tensor
    #   - permutation is an int-like vector-shaped tensor representing the permutation to apply
    # if you feed in permuter.forward([-1.,0,1.]) --> [1.,0.,-1.]

    permuter = tfp.bijectors.Permute( permutation = permutation) # initialize the permuter
    xfeatures_permuted = permuter.forward( nextfeature ) # apply permutation to transformation variables
    outlist = [] # stores two concurrent, but independent models for the two transforming inputs
    for i in range( inputdim ): 
      x = tf.reshape( xfeatures_permuted[ :, i ], [ -1, 1 ] )
      condnet = xcondin
      for iv in range( hidden_cond ):
        if regularizer.upper() in [ "BATCHNORM", "ALL" ]:
          condnet = layers.BatchNormalization( name = "BATCHNORM_{}_{}_{}".format( idepth, i, iv ) )( condnet )
        condnet = layers.Dense( nodes_cond, activation = activation_key[ activation ], kernel_initializer = initializer, kernel_regularizer = regularizers.L1L2(), name = "COND_DENSE_{}_{}_{}".format( idepth, i, iv ) )( condnet )
        if regularizer.upper() in [ "DROPOUT", "ALL" ]:
          condnet = layers.Dropout( 0.3, name = "DROPOUT_{}_{}_{}".format( idepth, i, iv ) )( condnet )
      w1 = layers.Dense( nodes_trans, activation = tf.nn.softplus, kernel_initializer = initializer, kernel_regularizer = regularizers.L1L2(), name = "SIGMOID_WEIGHT_{}_{}".format( idepth, i ) )( condnet ) # has to be softplus for output to be >0
      b1 = layers.Dense( nodes_trans, activation = None, name = "SIGMOID_BIAS_{}_{}".format( idepth, i ) )( condnet )
      del condnet

      # apply sigmoidal transformation
      sig = tf.nn.sigmoid( w1 * x + b1,  name = "SIGMOID_{}_{}".format( idepth, i ) ) 
       
      # inverse conditioner network
      condnet = xcondin
      #condnet = layers.Dense( nodes_cond, activation = activation_key[ activation ], name = "INV_COND_DENSE_{}_{}_{}".format( idepth, i, iv ) )( condnet )
      w2 = layers.Dense( nodes_trans, activation = tf.nn.softplus, kernel_initializer = initializer, kernel_regularizer = regularizers.L1L2(), name = "INV_SIGMOID_WEIGHT_{}_{}".format( idepth, i ) )( condnet )
      w2 = w2 / ( 1e-12 + tf.reduce_sum( w2, axis = 1, keepdims = True ) ) # normalize the transformer output for softmax weighting to retain normalization in sigmoidal space
      
      # inverse transformer network
      sigflow = invsigmoid( tf.reduce_sum( sig *  w2, axis = 1, keepdims = True ) )
      
      outlist.append( sigflow )
      xcondin = tf.concat( [ xcondin, x ], axis = 1 )
      
    outputlayer_permuted = tf.concat( outlist, axis = 1 )
    outputlayer = permuter.inverse( outputlayer_permuted )
    nextfeature = outputlayer
    
  return keras.Model( xin, outputlayer )
      
def mix_rbf_mmd2( X, Y, sigmas = ( 1, ), wts = None ):
  def mix_rbf_kernel( X, Y, sigmas, wts ):
    if wts is None: wts = [1] * len( sigmas )
    
    # matmul is matrix multiplication between X and Y where X is the target batch and Y is the generated batch
    XX = tf.matmul( X, X, transpose_b = True )
    XY = tf.matmul( X, Y, transpose_b = True )
    YY = tf.matmul( Y, Y, transpose_b = True )
    
    X_sqnorms = tf.linalg.diag_part( XX )
    Y_sqnorms = tf.linalg.diag_part( YY )
    
    r = lambda x: tf.expand_dims( x, 0 )
    c = lambda x: tf.expand_dims( x, 1 )
    
    K_XX, K_XY, K_YY = 0, 0, 0
    
    for sigma, wt in zip( sigmas, wts ):
      gamma = 1 / ( 2 * sigma**2 )
      K_XX += wt * tf.exp( -gamma * ( -2 * XX + c(X_sqnorms) + r(X_sqnorms) ) )
      K_XY += wt * tf.exp( -gamma * ( -2 * XY + c(X_sqnorms) + r(Y_sqnorms) ) )
      K_YY += wt * tf.exp( -gamma * ( -2 * YY + c(Y_sqnorms) + r(Y_sqnorms) ) )
    
    # reduce_sum() computes sum of elements across dimensions of a tensor
    return K_XX, K_XY, K_YY, tf.reduce_sum( wts )
  
  K_XX, K_XY, K_YY, d = mix_rbf_kernel( X, Y, sigmas, wts )
  m = tf.cast( tf.shape( K_XX )[0], tf.float32 )
  n = tf.cast( tf.shape( K_YY )[0], tf.float32 )
  
  mmd2 = ( tf.reduce_sum( K_XX ) / ( m * m ) + tf.reduce_sum( K_YY ) / ( n * n ) - 2 * tf.reduce_sum( K_XY ) / ( m * n ) )
  return mmd2
  
# Onehot encoding class
class OneHotEncoder_int( object ):
  def __init__( self, categorical_features, lowerlimit = None, upperlimit = None ):
    self.iscategorical = categorical_features
    self.ncolumns = len(categorical_features)
    self.categories_per_feature = []

    self.ncatgroups = 0
    for b in categorical_features:
      if b:
        self.ncatgroups += 1
    self.lowerlimit = lowerlimit 
    self.upperlimit = upperlimit
    self.categories_fixed = False
    pass

  def applylimit( self, categoricalinputdata ):
    if self.lowerlimit is None:
      self.lowerlimit = np.min( categoricalinputdata, axis = 0 )
    
    if self.upperlimit is None:
      self.upperlimit = np.max( categoricalinputdata, axis = 0 )

    lowerlimitapp = np.maximum( categoricalinputdata, self.lowerlimit )
    limitapp = np.minimum( lowerlimitapp, self.upperlimit )
    return limitapp

  def encode( self, inputdata ):
  # encoding done in prepdata()
    cat_limited = self.applylimit( inputdata ) - self.lowerlimit
    # one hot encoding information
    if not self.categories_fixed:
      for icol, iscat in zip( range( self.ncolumns ), self.iscategorical ):
        if iscat:
          ncats = int( self.upperlimit[icol] - self.lowerlimit[icol] + 1 )
          self.categories_per_feature.append( ncats )
        else:
          self.categories_per_feature.append( 0 )
      self.categories_fixed = True

    # the encoding part
    arraylist = []
    for icol, ncat_feat in zip( range( self.ncolumns ), self.categories_per_feature ):
      if ncat_feat > 0:
        res = np.eye( ncat_feat )[ cat_limited[ :,icol ].astype(int) ]
        arraylist.append( res )
      else:
        arraylist.append( inputdata[:,icol].reshape( ( inputdata.shape[0], 1 ) ) )
    encoded = np.concatenate( tuple( arraylist ), axis = 1 ).astype( np.float32 )
    return encoded

  def decode( self, onehotdata ):
    current_col = 0 # start from column 0
    arraylist = []
    for ifeat, ncats in zip( range( len( self.categories_per_feature ) ), self.categories_per_feature ):
      if ncats > 0:
        datatoconvert = onehotdata[ :, current_col:current_col+ncats ]
        converted = np.argmax( datatoconvert, axis = 1 ) + self.lowerlimit[ ifeat ]
        converted = np.reshape( converted, newshape = ( converted.shape[0], 1 ) )
        arraylist.append( converted )
        current_col += ncatgroups
      else:
        arraylist.append( onehotdata[:, current_col].reshape((onehotdata.shape[0], 1) ) )
        current_col += 1
    decoded = np.concatenate( tuple( arraylist ), axis = 1 )
    return decoded
  
class SawtoothSchedule( LearningRateSchedule ):
  def __init__(self, start_learning_rate = 1e-4, end_learning_rate = 1e-6, cycle_steps = 100, random_fluctuation = 0.0, name = None ):
    super( SawtoothSchedule, self ).__init__()
    self.start_learning_rate = start_learning_rate 
    self.end_learning_rate = end_learning_rate
    self.cycle_steps = cycle_steps
    self.random_fluctuation = random_fluctuation
    self.name = name
  pass

  def __call__( self, step ):
    phase = step % self.cycle_steps
    lr = tf.gather( np.geomspace( self.start_learning_rate, self.end_learning_rate, self.cycle_steps ), tf.cast( phase, tf.int32 ) ) 
    if ( self.random_fluctuation > 0 ):
      lr *= np.random.normal( 1.0, self.random_fluctuation )
    return lr

  def get_config( self ):
    return {
        "start_learning_rate": self.start_learning_rate,
        "end_learning_rate": self.end_learning_rate,
        "cycle_step": self.cycle_steps,
        "random_fluctuation": self.random_fluctuation,
        "name": self.name
    }
  
def unweight(pddata):
# by default, this isn't used, need to use option prepdata( mc_weight = "unweight" )
  nrows = pddata.shape[0]
  minweight = pddata['xsecWeight'].min()
  maxweight = pddata['xsecWeight'].max()
  selrows = pddata['xsecWeight']>minweight # identify all entries above the minimum xsec weight

  datatoexpand = pddata[selrows].sort_values(by=['xsecWeight']) # sort selected entries by xsec weight
  nselected = datatoexpand.shape[0] 
  idx = 0
  while idx < nselected-1: # loop through all selected entries
    thisweight = datatoexpand.iloc[idx]['xsecWeight']
    multfactor = int(thisweight//minweight)
    selectrows = (datatoexpand['xsecWeight']==thisweight)
    matches = datatoexpand[selectrows]
    nmatches = matches.shape[0]
    pddata = pddata.append([matches]*multfactor)
    idx += nmatches
  return pddata

def prepdata( rSource, rTarget, variables, regions, closure ):
# mc_weight(str) = option for weighting MC by the xsec
# rSource (str) = source ROOT file
# rTarget (str) = target ROOT file
# selection (dict) = event selection with key: value
# variables (dict) = list of all variables considered and associated parameters

  # set up one-hot encoder
  vNames = [ str( key ) for key in sorted( variables ) if variables[key]["TRANSFORM"] ]
  if regions["Y"]["VARIABLE"] in variables and regions["X"]["VARIABLE"] in variables:
    vNames.append( regions["Y"]["VARIABLE"] )
    vNames.append( regions["X"]["VARIABLE"] )
  else:
    sys.exit( "[ERROR] Control variables are not listed in config.variables, please include. Exiting..." )
  categorical = [ variables[ vName ][ "CATEGORICAL" ] for vName in vNames ]
  upperlimit  = [ variables[ vName ][ "LIMIT" ][1] for vName in vNames ]
  lowerlimit  = [ variables[ vName ][ "LIMIT" ][0] for vName in vNames ]
  
  _onehotencoder = OneHotEncoder_int( categorical, lowerlimit = lowerlimit, upperlimit = upperlimit )

  # read MC and data
  fMajor = uproot.open( rSource )
  fTarget = uproot.open( rTarget )
  tMajor = fMajor[ 'Events' ]
  tTarget = fTarget[ 'Events' ]
  
  dfMajor = tMajor.arrays( vNames, library = "pd" )
  dfMajor = dfMajor.loc[ ( dfMajor[ regions[ "X" ][ "VARIABLE" ] ] >= regions[ "X" ][ "MIN" ] ) & ( dfMajor[ regions[ "Y" ][ "VARIABLE" ] ] >= regions[ "Y" ][ "MIN" ] ) ]
  dfTarget = tTarget.arrays( vNames, library = "pd" )
  dfTarget = dfTarget.loc[ ( dfTarget[ regions[ "X" ][ "VARIABLE" ] ] >= regions[ "X" ][ "MIN" ] ) & ( dfTarget[ regions[ "Y" ][ "VARIABLE" ] ] >= regions[ "Y" ][ "MIN" ] ) ]

  inputRawMajor = dfMajor
  inputEncMajor = _onehotencoder.encode( inputRawMajor.to_numpy( dtype=np.float32 ) )
  
  inputRawTarget = dfTarget
  inputEncTarget = _onehotencoder.encode( inputRawTarget.to_numpy( dtype=np.float32 ) )

  ncats = _onehotencoder.ncatgroups
  ncat_per_feature = _onehotencoder.categories_per_feature

  meanslist = []
  sigmalist = []
  currentcolumn = 0

  # normalize data
  for ncatfeat in ncat_per_feature:
    if ncatfeat == 0: # for float features, get mean and sigma
      mean = np.mean( inputRawTarget.to_numpy( dtype = np.float32 )[:, currentcolumn], axis=0, dtype=np.float32 ).reshape( 1, 1 )
      meanslist.append( mean )
      sigma = np.std( inputRawTarget.to_numpy( dtype = np.float32 )[:, currentcolumn], axis=0, dtype=np.float32 ).reshape( 1, 1 )
      sigmalist.append( sigma )
      currentcolumn += 1
    else: # categorical features do not get changed
      mean = np.zeros( shape = ( 1, ncatfeat ), dtype=np.float32 ) 
      meanslist.append( mean )
      sigma = np.ones( shape = ( 1, ncatfeat ), dtype=np.float32 )
      sigmalist.append( sigma )
      currentcolumn += ncatfeat

  inputMean = np.hstack( meanslist )
  inputSigma = np.hstack( sigmalist )

  inputNormTarget = ( inputEncTarget - inputMean ) / inputSigma        # normed Data
  inputNormMajor = ( inputEncMajor - inputMean ) / inputSigma              # normed MC  
  
  return inputRawTarget, inputRawMajor, inputNormTarget, inputNormMajor, inputMean, inputSigma, vNames, ncat_per_feature
  
  
# construct the ABCDnn model here
class ABCDnn(object):
  def __init__( self, variables, regions,inputdim_categorical_list, inputdim, nodes_cond, hidden_cond,
               nodes_trans, minibatch, activation, regularizer, initializer,
               depth, lr, gap, conddim, beta1, beta2, mmd_sigmas, mmd_weights, decay,
               retrain, savedir, savefile, disc_tag, closure,
               seed, permute, verbose, model_tag ):
    self.variables = variables
    self.regions = regions
    self.inputdim_categorical_list = inputdim_categorical_list
    self.inputdim = inputdim
    self.inputdimcat = int( np.sum( inputdim_categorical_list ) )
    self.inputdimreal = inputdim - self.inputdimcat
    self.minibatch = minibatch
    self.nodes_cond = nodes_cond
    self.hidden_cond = hidden_cond
    self.nodes_trans = nodes_trans
    self.activation = activation
    self.regularizer = regularizer
    self.initializer = initializer
    self.closure = closure
    self.depth = depth
    self.lr = lr 
    self.decay = decay
    self.gap = gap
    self.conddim = conddim
    self.beta1 = beta1
    self.beta2 = beta2
    self.mmd_sigmas = mmd_sigmas
    self.mmd_weights = mmd_weights
    self.retrain = retrain
    self.savedir = savedir
    self.savefile = savefile
    self.disc_tag = disc_tag,
    self.global_step = tf.Variable( 0, name = "global_step" )
    self.monitor_record = []
    self.seed = seed 
    self.permute = permute
    self.verbose = verbose
    self.model_tag = model_tag
    self.minloss = 0
    self.setup()

  def setup( self ):
    np.random.seed( self.seed )
    tf.random.set_seed( self.seed )
    self.createmodel()
    self.checkpoint = tf.train.Checkpoint( global_step = self.global_step, model = self.model, optimizer = self.optimizer )
    self.checkpointmgr = tf.train.CheckpointManager( self.checkpoint, directory = self.savedir, max_to_keep = 1 )
    if (not self.retrain) and os.path.exists(self.savedir):
      status = self.checkpoint.restore(self.checkpointmgr.latest_checkpoint)
      status.assert_existing_objects_matched()
      print( ">> Loaded model from checkpoint" )
      if os.path.exists(os.path.join(self.savedir, self.savefile)):
        print( ">> Reading monitor file" )
        self.load_training_monitor()
      print( ">> Resuming from step {}".format( self.global_step ) )
    elif not os.path.exists(self.savedir):
      os.mkdir(self.savedir)
    pass

  def createmodel( self ):
    self.model = NAF( 
      inputdim = self.inputdim,  
      conddim = self.conddim, 
      activation = self.activation,
      regularizer = self.regularizer,
      initializer = self.initializer,
      nodes_cond = self.nodes_cond, 
      hidden_cond = self.hidden_cond,
      nodes_trans = self.nodes_trans, 
      depth = self.depth, 
      permute = self.permute
    )
    if self.verbose: self.model.summary()
    self.optimizer = keras.optimizers.legacy.Adam(
      learning_rate = SawtoothSchedule( self.lr, self.lr * self.decay, self.gap, 0 ),  
      beta_1 = self.beta1, beta_2 = self.beta2, 
      epsilon = 1e-7, 
    )
    pass

  def category_sorted(self, numpydata, verbose ):
    categoricals, categorical_cats, unique_counts = np.unique( numpydata[:, self.inputdimreal:], axis=0, return_inverse = True, return_counts = True)
    print( categoricals )
    if verbose: 
      print( "Data has {} unique categorical features. The counts in categories are".format( categoricals ) )
      print( unique_counts )

    # store indices separately for easy access later
    categorical_indices_grouped = []
    for icat in range( len( categoricals ) ):
      cat_indices = np.where( categorical_cats == icat )[0]
      categorical_indices_grouped.append( cat_indices )
    
    return categoricals, categorical_indices_grouped

  def setrealdata( self, numpydata, verbose ):
    self.numpydata = numpydata
    self.ntotalevents = numpydata.shape[0]
    self.datacounter = 0
    self.randorder = np.random.permutation( self.numpydata.shape[0] )
    # following is dummy
    self.dataeventweight = np.ones((self.ntotalevents, 1), np.float32)
    # find unique occurrenes of categorical features in data
    self.categoricals_data, self.categorical_data_indices_grouped = self.category_sorted( self.numpydata, verbose )
    pass

  def setmcdata(self, numpydata, verbose ):
    self.mcnumpydata = numpydata
    self.mcntotalevents = numpydata.shape[0]
    self.mcdatacounter = 0
    self.mcrandorder = np.random.permutation( self.mcnumpydata.shape[0] )
    self.mceventweight = np.ones( ( self.mcntotalevents, 1 ) , np.float32 )
    self.categoricals_mc, self.categorical_mc_indices_grouped = self.category_sorted( self.mcnumpydata, verbose )
    pass

  def savehyperparameters(self, inputs, means, sigmas ):
    """Write hyper parameters into file
    """
    means_list = [ str( mean ) for mean in means[0] ]
    sigmas_list = [ str( sigma ) for sigma in sigmas[0] ]
    inputs_list = [ input_ for input_ in inputs ]
    params = {
      "INPUTDIM": self.inputdim,
      "CONDDIM": self.conddim,
      "NODES_COND": self.nodes_cond, 
      "HIDDEN_COND": self.hidden_cond, 
      "NODES_TRANS": self.nodes_trans,
      "LRATE": self.lr, 
      "DECAY": self.decay, 
      "GAP": self.gap,
      "DEPTH": self.depth,
      "PERMUTE": self.permute,
      "REGULARIZER": self.regularizer,
      "INITIALIZER": self.initializer,
      "ACTIVATION": self.activation,
      "BETA1": self.beta1, 
      "BETA2": self.beta2, 
      "MINIBATCH": self.minibatch,
      "DISC TAG": self.disc_tag,
      "INPUTS": inputs_list,
      "SIGMAS": self.mmd_sigmas,
      "INPUTMEANS": means_list,
      "INPUTSIGMAS": sigmas_list,
      "VARIABLES": self.variables,
      "REGIONS": self.regions,
      "EPOCHS": self.steps,
      #"TRANSFER": transfer,
      "CLOSURE": self.closure
    }
    
    with open( os.path.join( self.savedir, "{}.json".format( self.model_tag ) ), "w" ) as f:
      f.write( write_json( params, indent = 2 ) )

  def monitor( self, step, glossv_trn, mmdloss_trn ):
    self.monitor_record.append( [ 
      step, 
      glossv_trn,  
      mmdloss_trn
      ]
    )

  def save_training_monitor(self):
    pickle.dump( self.monitor_record, open(os.path.join( self.savedir, self.savefile ), 'wb' ))
    pass

  def load_training_monitor(self):
    fullfile = os.path.join( self.savedir, self.savefile )
    if os.path.exists(fullfile):
      self.monitor_record = pickle.load(open(fullfile, 'rb'))
      self.epoch = self.monitor_record[-1][0] + 1
    pass

  def find_condmatch(self, size, conditional):
    
    """[Find data and MC batches matching conditional category]

    Args:
        conditional ([numpy]): [single conditional]
    """
    idx_cond = ( ( self.categoricals_data == conditional ).all( axis = 1 ).nonzero()[0])[0]
    # Data
    data_for_cond = self.categorical_data_indices_grouped[idx_cond]
    nextdatabatchidx = np.random.permutation(data_for_cond)[0:size]
    target_b = self.numpydata[nextdatabatchidx]

    # MC
    idx_cond = ((self.categoricals_mc == conditional).all(axis=1).nonzero()[0])[0]
    mc_for_cond = self.categorical_mc_indices_grouped[idx_cond]
    mcnextbatchidx = np.random.permutation(mc_for_cond)[0:size]
    source_b = self.mcnumpydata[mcnextbatchidx]
    weight_b = self.mceventweight[mcnextbatchidx]
    return target_b, source_b, weight_b
  
  def get_next_batch( self, size=None ):
    """Return minibatch from random ordered numpy data
    """
    if size is None:
      size = int( self.minibatch )

    # reset counter if no more entries left for current batch
    if self.datacounter + size >= self.ntotalevents:
      self.datacounter = self.datacounter + size - self.ntotalevents
      self.randorder = np.random.permutation( self.numpydata.shape[0] )

    batchbegin = self.datacounter
    rChoice = np.random.choice( [ "X", "Y", "A", "C", "B" ], 1 )
    if rChoice == "X": nextconditional = np.array( [ 1, 0, 1, 0, 0 ] )
    if rChoice == "Y": nextconditional = np.array( [ 0, 1, 1, 0, 0 ] )
    if rChoice == "A": nextconditional = np.array( [ 1, 0, 0, 1, 0 ] )
    if rChoice == "C": nextconditional = np.array( [ 0, 1, 0, 1, 0 ] )
    if rChoice == "B": nextconditional = np.array( [ 1, 0, 0, 0, 1 ] )
    #nextconditional = self.numpydata[ self.randorder[batchbegin], self.inputdim: ]
    target_b, source_b, weight_b = self.find_condmatch( size, nextconditional )
    
    while len( target_b ) != len( source_b ):
      self.datacounter += size
      if self.datacounter >= self.ntotalevents:
        self.datacounter = self.datacounter - self.ntotalevents
        self.randorder = np.random.permutation( self.numpydata.shape[0] )
      batchbegin = self.datacounter
      nextconditional = self.numpydata[ self.randorder[ batchbegin ], self.inputdim: ]
      target_b, source_b, weight_b = self.find_condmatch( size, nextconditional )
    self.datacounter += size
    self.this_source = source_b
    self.this_target = target_b
    self.this_weight = weight_b

    return source_b, target_b, weight_b

  def get_batch_region( self, region, size=None ):
    """Return minibatch from region of choice"""
    if size is None:
      size = int( self.minibatch )
      
    # reset counter if no more entries left for current batch
    if self.datacounter + size >= self.ntotalevents:
      self.datacounter = self.datacounter + size - self.ntotalevents
      self.randorder = np.random.permutation( self.numpydata.shape[0] )
    
    batchbegin = self.datacounter
    if region == "X": nextconditional = np.array( [ 1, 0, 1, 0, 0 ] )
    if region == "Y": nextconditional = np.array( [ 0, 1, 1, 0, 0 ] )
    if region == "A": nextconditional = np.array( [ 1, 0, 0, 1, 0 ] )
    if region == "C": nextconditional = np.array( [ 0, 1, 0, 1, 0 ] )
    if region == "B": nextconditional = np.array( [ 1, 0, 0, 0, 1 ] )
    target_b, source_b, weight_b = self.find_condmatch( size, nextconditional )
    
    while len( target_b ) != len( source_b ):
      self.datacounter += size
      if self.datacounter >= self.ntotalevents:
        self.datacounter = self.datacounter - self.ntotalevents
        self.randorder = np.random.permutation( self.numpydata.shape[0] )
      batchbegin = self.datacounter
      nextconditional = self.numpydata[ self.randorder[ batchbegin ], self.inputdim: ]
      target_b, source_b, weight_b = self.find_condmatch( size, nextconditional )
      
    return source_b, target_b, weight_b

  # eqvuialent to train_step = tf.function( train_step )
  @tf.function
  def train_step( self, source, target, sourceweight=1. ):
    with tf.GradientTape() as gtape:
      # get the predicted values from the current trained model
      generated = tf.concat( 
	    [
          self.model( source ),
          target[:, -self.conddim:]
        ],
        axis = -1
      )
      mmdloss = mix_rbf_mmd2( target[:, :self.inputdim], generated[:, :self.inputdim], self.mmd_sigmas, self.mmd_weights )

    gradient = gtape.gradient( mmdloss, self.model.trainable_variables )
    self.optimizer.apply_gradients( zip( 
        gradient, 
        self.model.trainable_variables 
      ) 
    )

    glossv = tf.reduce_mean( mmdloss )

    return glossv, mmdloss

  def get_loss( self, source, target ):
    with tf.GradientTape() as gtape:
      generated = tf.concat(
        [
          self.model( source ),
          target[:, -self.conddim:]
        ],
        axis = -1
      )
      mmdloss = mix_rbf_mmd2( target[:, :self.inputdim], generated[:,:self.inputdim], self.mmd_sigmas, self.mmd_weights )

    return mmdloss

  def train( self, steps = 10000, monitor = 1000, patience = 100, early_stopping = True, monitor_threshold = 0, hpo = False, periodic_save = False ):
    # need to update this to use a validation set
    print( "{:<5} / {:<9} / {:<9} / {:<9} / {:<16} / {:<6} / {:<10}".format( "Epoch", "MMD", "Min MMD", "Avg MMD", "Avg MMD Global", "Region", "L Rate" ) ) 
    impatience = 0      # don't edit
    stop_train = False  # don't edit
    self.minepoch = 0
    self.steps = steps
    save_counter = 0    # edit so you aren't saving every time the model improves --> takes too much time
    losses = []
    global_losses = []
    for i in range( steps ):
      source, target, batchweight = self.get_next_batch()
      # train the model on this batch
      glossv, mmdloss  = self.train_step( source, target, batchweight )
      mmdloss_global = []
      for region in [ "A", "B", "C", "X", "Y" ]:
        source_r, target_r, weight_r = self.get_batch_region( region )
        mmdloss_global.append( self.get_loss( source_r, target_r ) )
      avg_mmdloss_global = np.mean( mmdloss_global )
      # currently not using glossv because mmdloss is a scalar and glossv is the mean of mmdloss
      # generator update
      global_losses.append( avg_mmdloss_global )
      losses.append( mmdloss )
      if i == 0: 
        self.minloss = avg_mmdloss_global
      # early stopping on validation implementation
      else: 
        if avg_mmdloss_global < self.minloss:
          self.minepoch = i
          impatience = 0 # reset the impatience counter
          if not hpo: # don't save models during hyper parameter optimization training to save time
            if save_counter > monitor: # reduce number of saved models
              save_counter = 0
              self.checkpointmgr.save()
              self.model.save_weights( "./Results/{}".format( self.model_tag ) )
          self.minloss = avg_mmdloss_global
        elif impatience > patience and early_stopping:
          print( "[WARN] Early stopping after {} epochs without improvement in loss (min loss = {:.3e})".format( i, self.minloss ) )
          stop_train = True
        else:
          impatience += 1
      if i % monitor == 0:
        cArr = target[:,-self.conddim:][0]
        if cArr[0] == 1 and cArr[2] == 1: category_ = "X"
        if cArr[1] == 1 and cArr[2] == 1: category_ = "Y"
        if cArr[0] == 1 and cArr[3] == 1: category_ = "A"
        if cArr[1] == 1 and cArr[3] == 1: category_ = "C"
        if cArr[0] == 1 and cArr[4] == 1: category_ = "B"
        print( "{:<5}   {:<9.2e}   {:<9.2e}   {:<9.2e}   {:<16.2e}   {:<6}     {:<10.2e}".format( 
          self.checkpoint.global_step.numpy(),
          mmdloss.numpy(),
          self.minloss,
          np.mean( losses[-monitor:] ),
          np.mean( global_losses[-monitor:] ),
          category_,
          self.optimizer._decayed_lr( tf.float32 )
         ) )
        self.monitor( 
          self.checkpoint.global_step.numpy(), 
          glossv.numpy(),
          mmdloss
        )
        if periodic_save and i >= monitor_threshold: self.model.save_weights( "./Results/{}_EPOCH{}".format( self.model_tag, i ) )
      self.checkpoint.global_step.assign_add(1) # increment counter
      save_counter += 1
      if stop_train:
        if periodic_save: self.model.save_weights( "./Results/{}_EPOCH{}".format( self.model_tag, i ) ) 
        break
    if not early_stopping: 
      self.checkpointmgr.save()
      self.model.save_weights( "./Results/{}".format( self.model_tag ) )
    print( ">> Minimum loss of {:.3e} on epoch {}".format( self.minloss, self.minepoch ) )
    os.system( "cp -v ./Results/{}.data-00000-of-00001 ./Results/{}_EPOCH{}.data-00000-of-00001".format( self.model_tag, self.model_tag, self.minepoch ) )
    os.system( "cp -v ./Results/{}.index ./Results/{}_EPOCH{}.index".format( self.model_tag, self.model_tag, self.minepoch ) )
    self.save_training_monitor()

class ABCDnn_training(object):
  def __init__( self ):
    pass

  def setup_events( self, rSource, rTarget, selection, variables, regions, closure ):
    # obtain the normalized data (plus others)
    self.rSource = rSource        # (str) path to source ROOT file
    self.rTarget = rTarget        # (str) path to target ROOT file
    self.variables = variables    # (dict) variable limits and type
    self.regions = regions        # (dict) control and signal regions

    self.sourceSF = 1. / ( float( self.rSource.split( "p" )[-1].split( ".root" )[0] ) / 100. )
    self.targetSF = 1. / ( float( self.rTarget.split( "p" )[-1].split( ".root" )[0] ) / 100. )

    rawinputs, rawinputsmc, normedinputs, normedinputsmc, inputMean, \
      inputSigma, inputnames, ncat_per_feature = prepdata( rSource,  rTarget,
      variables, regions, closure )

    self.rawinputs = rawinputs                # unnormalized data tree after event selection
    self.rawinputsmc = rawinputsmc            # unnormalized mc major tree after event selection
    self.normedinputs = normedinputs          # normalized data tree after event selection
    self.normedinputsmc = normedinputsmc      # normalized mc tree after event selection
    self.inputs = inputnames                  # name of inputs
    self.inputMean = inputMean              # mean of unnormalized data
    self.inputSigma = inputSigma              # rms of unnormalized data

    self.inputdim = len( list( variables.keys() ) ) - 2 # the total number of transformed variables
    self.ncat_per_feature = ncat_per_feature[0:self.inputdim] # number of categories per categorical feature
    self.conddim = self.normedinputs.shape[1] - self.inputdim # ?

    # Data and MC in control region

    self.CV_x = self.regions["X"]["VARIABLE"] # First control variable name
    self.CV_y = self.regions["Y"]["VARIABLE"] # Second control variable name

    # apply region selection per event
    if regions[ "X" ][ "INCLUSIVE" ]:
      self.sig_select = ( self.rawinputs[ self.CV_x ] >= regions[ "X" ][ "SIGNAL" ] )
      self.sig_select_mc = ( self.rawinputsmc[ self.CV_x ] >= regions[ "X" ][ "SIGNAL" ] )
    else:
      self.sig_select = ( self.rawinputs[ self.CV_x ] == regions[ "X" ][ "SIGNAL" ] )
      self.sig_select_mc = ( self.rawinputsmc[ self.CV_x ] == regions[ "X" ][ "SIGNAL" ] )
    if regions[ "Y" ][ "INCLUSIVE" ]:
      self.sig_select &= ( self.rawinputs[ self.CV_y ] >= regions[ "Y" ][ "SIGNAL" ] )
      self.sig_select_mc &= ( self.rawinputsmc[ self.CV_y ] >= regions[ "Y" ][ "SIGNAL" ] )
    else:
      self.sig_select &= ( self.rawinputs[ self.CV_y ] == regions[ "Y" ][ "SIGNAL" ] )
      self.sig_select_mc &= ( self.rawinputsmc[ self.CV_y ] == regions[ "Y" ][ "SIGNAL" ] )

    self.bkg_select = ~self.sig_select
    self.normedinputs_bkg = self.normedinputs[ self.bkg_select ]

    self.bkg_select_mc = ~self.sig_select_mc
    self.normedinputsmc_bkg = self.normedinputsmc[ self.bkg_select_mc ]
  
    pass

  def setup_model( self, 
          nodes_cond = 8, hidden_cond = 1, nodes_trans = 8, minibatch = 64, lr = 5.0e-3,
          depth = 1, activation = "swish", regularizer = "None", initializer = "RandomNormal", decay = 1e-1,
          gap = 1000., beta1 = 0.9, beta2 = 0.999, mmd_sigmas = (0.1,1.0), mmd_weights = None,
          savedir = "/ABCDNN/", savefile = "abcdnn.pkl", disc_tag = "ABCDnn", closure = 0.03, mc_weight = None, 
          retrain = False, seed = 100, permute = False, model_tag = "best_model", verbose = False
        ):
    self.nodes_cond = nodes_cond    # (int) number of nodes in conditional layer(s)
    self.hidden_cond = hidden_cond  # (int) number of conditional layers
    self.nodes_trans = nodes_trans  # (int) number of nodes in transformer layer
    self.minibatch = minibatch      # (int) batch size for training
    self.lr = lr                    # (float) learning rate for training
    self.decay = decay              # (float) final learning rate multiplier
    self.gap = gap                  # (float) the gap in epochs for learning rate step increment values
    self.depth = depth              # (int) number of permutations in creating layers
    self.beta1 = beta1              # (float) first moment decay rate for Adam
    self.beta2 = beta2              # (float) second moment decay rate for Adam
    self.mmd_sigmas = mmd_sigmas    # (tuple of floats) defines the reach of the RBF kernel during training
    self.mmd_weights = mmd_weights  # (tuple of floats, or none) defines the weight assigned to mmd loss associated with different sigma
    self.regularizer = regularizer  # (str) use of L1, L2, L1+L2 or None for training regularizers
    self.initializer = initializer
    self.activation = activation    # (str) activation function for conditional layers
    self.savedir = savedir          # (str) save directory for output results
    self.savefile = savefile        # (str) save file name for output results
    self.disc_tag = disc_tag        # (str) tag to add to transformed variable name
    self.closure = closure          # (float) multiplicative factor to vary the ABCDnn output
    self.seed = seed                # (int) RNG seed
    self.retrain = retrain          # (bool) start a new training or continue from where left-off
    self.mc_weight = mc_weight      # (str) weight MC values according to xsec
    self.permute = permute          # (bool) run random permutations of the conditional inputs
    self.model_tag = model_tag

    self.model = ABCDnn( 
      variables = self.variables,
      regions = self.regions,
      inputdim_categorical_list = self.ncat_per_feature, 
      inputdim = self.inputdim,
      conddim = self.conddim,
      nodes_cond = self.nodes_cond,
      hidden_cond = self.hidden_cond,
      nodes_trans = self.nodes_trans, 
      minibatch = self.minibatch,
      activation = self.activation,
      regularizer = self.regularizer,
      initializer = self.initializer,
      depth = self.depth, 
      lr = self.lr,
      decay = self.decay, 
      gap = self.gap,
      beta1 = self.beta1, 
      beta2 = self.beta2,
      mmd_sigmas = self.mmd_sigmas,
      mmd_weights = self.mmd_weights,
      retrain = self.retrain, 
      savedir = self.savedir, 
      savefile = self.savefile,
      disc_tag = self.disc_tag,
      closure = self.closure,
      seed = self.seed, 
      permute = self.permute, 
      model_tag = self.model_tag,
      verbose = verbose
    )

    self.model.setrealdata( self.normedinputs_bkg, verbose = verbose )
    self.model.setmcdata( self.normedinputsmc_bkg, verbose = verbose )

  def train( self, steps = 10000, monitor = 1000, patience = 100, early_stopping = True, display_loss = False, monitor_threshold = 0, hpo = False, periodic_save = False ):
    self.model.train( steps = steps, monitor = monitor, patience = patience, early_stopping = early_stopping, monitor_threshold = monitor_threshold, hpo = hpo, periodic_save = periodic_save )

  def validate( self ):
    self.model.checkpoint.restore( self.savedir )
    self.normedinputsmc_sig = self.normedinputsmc[ self.sig_select_mc ]
    self.n_mc_batches = int( self.normedinputsmc_sig.shape[0] / self.minibatch )
    self.fakelist = []
    for i in range( n_mc_batches ):
      xin = self.normedinputsmc[ i * minibatch: ( i + 1 ) * minibatch, : ]
      xgen = self.model.predict( xin )
      self.fakelist.append( xgen )
    self.fakedata = np.vstack( self.fakelist )
    self.fakedata = self.fakedata * self.inputSigma[ :, :self.inputdim ] + self.inputMean[ :, :self.inputdim ]

  def evaluate_regions( self, hpo = False, verbose = True ):
    self.select = {
        "DATA": {},
        "MC": {},
        "MINOR": {},
    }
    self.count = {
        "DATA": {},
        "MC": {},
        "MINOR": {}
    }
    self.fakedata = []
    self.rawdata = []
    self.rawmc = []
    self.mcweight = []
    self.plottext = []
    self.region = [ "X", "Y", "A", "C", "B", "D" ]

    i = 0
    for x in np.linspace( self.regions[ "X" ][ "MIN" ], self.regions[ "X" ][ "MAX" ], self.regions[ "X" ][ "MAX" ] - self.regions[ "X" ][ "MIN" ] + 1 ):
      for y in np.linspace( self.regions[ "Y" ][ "MIN" ], self.regions[ "Y" ][ "MAX" ], self.regions[ "Y" ][ "MAX" ] - self.regions[ "Y" ][ "MIN" ] + 1 ):
        if hpo and self.region[i] != "D": 
          if verbose: print( "[OPT] Running in HPO mode, skipping evaluation of Region {}".format( self.region[i] ) )
          i += 1
          continue
        if x == self.regions[ "X" ][ "SIGNAL" ] and self.regions[ "X" ][ "INCLUSIVE" ]:
          self.select[ "DATA" ][ self.region[i] ] = ( self.rawinputs[ self.regions[ "X" ][ "VARIABLE" ] ] >= x )
          self.select[ "MC" ][ self.region[i] ] = ( self.rawinputsmc[ self.regions[ "X" ][ "VARIABLE" ] ] >= x )
        else: 
          self.select[ "DATA" ][ self.region[i] ] = ( self.rawinputs[ self.regions[ "X" ][ "VARIABLE" ] ] == x )
          self.select[ "MC" ][ self.region[i] ] = ( self.rawinputsmc[ self.regions[ "X" ][ "VARIABLE" ] ] == x )
        if y == self.regions[ "Y" ][ "SIGNAL" ] and self.regions[ "Y" ][ "INCLUSIVE" ]:
          self.select[ "DATA" ][ self.region[i] ] &= ( self.rawinputs[ self.regions[ "Y" ][ "VARIABLE" ] ] >= y )
          self.select[ "MC" ][ self.region[i] ] &= ( self.rawinputsmc[ self.regions[ "Y" ][ "VARIABLE" ] ] >= y )
        else:
          self.select[ "DATA" ][ self.region[i] ] &= ( self.rawinputs[ self.regions[ "Y" ][ "VARIABLE" ] ] == y )
          self.select[ "MC" ][ self.region[i] ] &= ( self.rawinputsmc[ self.regions[ "Y" ][ "VARIABLE" ] ] == y )

        self.count[ "DATA" ][ self.region[i] ] = np.count_nonzero( self.select[ "DATA" ][ self.region[i] ] )
        self.count[ "MC" ][ self.region[i] ] = np.count_nonzero( self.select[ "MC" ][ self.region[i] ] )
        x_eq = ">=" if ( self.regions[ "X" ][ "INCLUSIVE" ] ) and ( x == self.regions[ "X" ][ "MAX" ] ) else "=="
        y_eq = ">=" if ( self.regions[ "Y" ][ "INCLUSIVE" ] ) and ( y == self.regions[ "Y" ][ "MAX" ] ) else "=="
        if verbose:
          print( "Region {} ({} {} {}, {} {} {}): MC = {}, DATA = {}, MINOR = {:.1f}".format(
              self.region[i], 
              self.regions[ "X" ][ "VARIABLE" ], x_eq, int( x ),
              self.regions[ "Y" ][ "VARIABLE" ], y_eq, int( y ),
              int( self.count[ "MC" ][ self.region[i] ] ), int( self.count[ "DATA" ][ self.region[i] ] ), self.count[ "MINOR" ][ self.region[i] ]
          ) )

        # text for plots
        text = "$"
        if self.regions[ "X" ][ "INCLUSIVE" ] == True and x == self.regions[ "X" ][ "MAX" ]:
          text += "{}\geq {}, ".format( self.variables[ self.regions[ "X" ][ "VARIABLE" ] ][ "LATEX" ], int(x) )
        else:
          text += "{}={}, ".format( self.variables[ self.regions[ "X" ][ "VARIABLE" ] ][ "LATEX" ], int(x) )
        
        if self.regions[ "Y" ][ "INCLUSIVE" ] == True and y == self.regions[ "Y" ][ "MAX" ]:
          text += "{}\geq {}".format( self.variables[ self.regions[ "Y" ][ "VARIABLE" ] ][ "LATEX" ], int(y) )
        else:
          text += "{}={}".format( self.variables[ self.regions[ "Y" ][ "VARIABLE" ] ][ "LATEX" ], int(y) )
        text += "$"
        self.plottext.append( text )
      
        i += 1

  def save_hyperparameters( self ):
    print( ">> Saving hyper parameters" )
    self.model.savehyperparameters( self.inputs, self.inputMean, self.inputSigma )
