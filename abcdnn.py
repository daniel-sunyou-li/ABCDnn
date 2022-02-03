# methods and classes used for constructing the ABCDnn module
# last updated 10/25/2021 by Daniel Li

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow_probability as tfp
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
import numpy as np
import os, uproot, pickle
from scipy import stats
from sklearn.model_selection import train_test_split
from json import dumps as write_json

def invsigmoid( x ):
# inverse sigmoid function for transformer
  xclip = tf.clip_by_value( x, 1e-6, 1.0 - 1e-6 )
  return tf.math.log( xclip / ( 1.0 - xclip ) )

def NAF( inputdim, conddim, activation, regularizer, nodes_cond, hidden_cond, nodes_trans, depth, permute ):
# neural autoregressive flow is a chain of MLP networks used for the conditioner and transformer parts of the flow
  activation_key = { # edit this if you add options to config.hyper dict with more activation functions
    "swish": tf.nn.swish,
    "softplus": tf.nn.softplus,
    "relu": tf.nn.relu,
    "elu": tf.nn.elu
  }
  regularizer_key = { 
    "L1": tf.keras.regularizers.l1,
    "L2": tf.keras.regularizers.l2,
    "L1+L2": tf.keras.regularizers.L1L2,
    "None": None
  }
  
  xin = layers.Input( shape = ( inputdim + conddim, ), name = "INPUT_LAYER" )
  xcondin = xin[ :, inputdim: ]
  xfeatures = xin[ :, :inputdim ]
  nextfeature = xfeatures
  
  for idepth in range( int( depth ) ):
    if permute: # mix up the input order of the consecutive neural networks in the flow
      randperm = np.random.permutation( inputdim ).astype( "int32" )
      permutation = tf.constant( randperm )
    else:
      permutation = tf.range( inputdim, dtype = "int32" )
    permuter = tfp.bijectors.Permute( permutation = permutation )
    xfeatures_permuted = permuter.forward( nextfeature )
    outlist = []
    for i in range( inputdim ):
      x = tf.reshape( xfeatures_permuted[ :, i ], [ -1, 1 ] )
      
      # conditioner network
      net1 = x
      condnet = xcondin
      for iv in range( hidden_cond ):
        condnet = layers.Dense( nodes_cond, activation = activation_key[ activation ], kernel_regularizer = regularizer_key[ regularizer ], name = "COND_DENSE_{}_{}_{}".format( idepth, i, iv ) )( condnet )
      w1 = layers.Dense( nodes_trans, activation = tf.nn.softplus, name = "SIGMOID_WEIGHT_{}_{}".format( idepth, i ) )(condnet ) # has to be softplus for output to be >0
      b1 = layers.Dense( nodes_trans, activation = None, name = "SIGMOID_BIAS_{}_{}".format( idepth, i ) )( condnet )
      del condnet
      # transforming layer
      net2 = tf.nn.sigmoid( w1 * net1 + b1,  name = "TRANS_DENSE_{}_{}".format( idepth, i ) ) 
       
      # reverse conditioner network
      condnet = xcondin
      for iv in range( hidden_cond ):
        condnet = layers.Dense( nodes_cond, activation = activation_key[ activation ], kernel_regularizer = regularizer_key[ regularizer ], name = "INV_COND_DENSE_{}_{}_{}".format( idepth, i, iv ) )( condnet )
      w2 = layers.Dense( nodes_trans, activation = tf.nn.softplus, name = "INV_SIGMOID_WEIGHT_{}_{}".format( idepth, i ) )( condnet )
      w2 = w2 / ( 1.0e-6 + tf.reduce_sum( w2, axis = 1, keepdims = True ) ) # normalize the transformer output
      
      # inverse transformer network
      net3 = invsigmoid( tf.reduce_sum( net2 * w2, axis = 1, keepdims = True ) )
      
      outlist.append( net3 )
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
      self.lowerlimit = np.min( categoricalinputdata, axis=0 )
    else:
      self.lowerlimit = np.maximum( self.lowerlimit, np.min( categoricalinputdata, axis = 0 ) )
    
    if self.upperlimit is None:
      self.upperlimit = np.max( categoricalinputdata, axis = 0 )
    else:
      self.upperlimit = np.minimum( self.upperlimit, np.max( categoricalinputdata, axis = 0 ) )

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

  def transform( self, inputdata ):
    return self.encode( inputdata )

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
    lr = self.start_learning_rate + ( self.end_learning_rate - self.start_learning_rate ) * ( phase / self.cycle_steps )
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
  
def prepdata( rSource, rTarget, selection, variables, mc_weight = None ):
# mc_weight(str) = option for weighting MC by the xsec
# rSource (str) = source ROOT file
# rTarget (str) = target ROOT file
# selection (dict) = event selection with key: value
# variables (dict) = list of all variables considered and associated parameters

  # set up one-hot encoder
  vars = [ str( var ) for var in sorted( list( variables.keys() ) ) ]
  categorical = [ variables[ var ][ "CATEGORICAL" ] for var in sorted( list( variables.keys() ) ) ]
  upperlimit  = [ variables[ var ][ "LIMIT" ][1] for var in sorted( list( variables.keys() ) ) ]
  lowerlimit  = [ variables[ var ][ "LIMIT" ][0] for var in sorted( list( variables.keys() ) ) ]
  
  _onehotencoder = OneHotEncoder_int( categorical, lowerlimit = lowerlimit, upperlimit = upperlimit )

  # read MC and data
  mcf = uproot.open( rSource )
  dataf = uproot.open( rTarget )
  tree_mc = mcf[ 'Events' ]
  tree_data = dataf[ 'Events' ]
  mc_df = {}
  data_df = {}
  for var in selection:
    mc_df[ var ] = tree_mc.pandas.df( var ) 
    data_df[ var ] = tree_data.pandas.df( var )
  zero_key = list( selection.keys() )[0]
  mc_select = ( mc_df[ zero_key ][ zero_key ] >= selection[ zero_key ] )
  data_select = ( data_df[ zero_key ][ zero_key ] >= selection[ zero_key ] )
  
  mc_dfs = tree_mc.pandas.df( vars )
  data_dfs = tree_data.pandas.df( vars )
  for var in selection:
    if var != zero_key:
      mc_select &= ( mc_df[ var ][ var ] >= selection[ var ] )
      data_select &= ( data_df[ var ][ var ] >= selection[ var ] )

  # do not consider cross section weight. The weight of the resut file is filled with 1.
  if mc_weight == None:
    inputrawmc = mc_dfs[ mc_select ]
    inputrawmcweight = None

  # The weight of the resut file is filled with 'xsecWeight'
  if mc_weight == "weight":
    inputrawmc = mc_dfs[ mc_select ]
    inputrawmcweight = tree_mc.pandas.df( [ 'xsecWeight' ] )[ mc_select ].to_numpy( dtype=np.float32 )
  
  # duplicate source samples according to the cross section. The weight of the resut file is filled with 1.
  # may take too long if input sample is the combined one with several samples that has different cross section.
  if mc_weight == "unweight":
    inputrawmc = unweight( tree_mc.pandas.df( vars + ['xsecWeight'] )[ mc_select ]).drop( columns=['xsecWeight'] )
    inputrawmcweight = None
  
  inputsmc_enc = _onehotencoder.encode( inputrawmc.to_numpy( dtype=np.float32 ) )

  inputrawdata = data_dfs[ data_select ]
  inputrawdata_np = inputrawdata.to_numpy( dtype=np.float32 )  
  inputrawdata_enc = _onehotencoder.encode( inputrawdata_np )

  inputsdata = _onehotencoder.encode( inputrawdata.to_numpy(dtype=np.float32))

  ncats = _onehotencoder.ncatgroups
  ncat_per_feature = _onehotencoder.categories_per_feature

  meanslist = []
  sigmalist = []
  currentcolumn = 0

  # normalize data
  for ncatfeat in ncat_per_feature:
    if ncatfeat == 0: # for float features, get mean and sigma
      mean = np.mean( inputrawdata_np[:, currentcolumn], axis=0, dtype=np.float32 ).reshape( 1, 1 )
      meanslist.append( mean )
      sigma = np.std( inputrawdata_np[:, currentcolumn], axis=0, dtype=np.float32 ).reshape( 1, 1 )
      sigmalist.append( sigma )
      currentcolumn += 1
    else: # categorical features do not get changed
      mean = np.zeros( shape = ( 1, ncatfeat ), dtype=np.float32 ) 
      meanslist.append( mean )
      sigma = np.ones( shape = ( 1, ncatfeat ), dtype=np.float32 )
      sigmalist.append( sigma )
      currentcolumn += ncatfeat

  inputmeans = np.hstack( meanslist )
  inputsigma = np.hstack( sigmalist )

  normedinputs_data = ( inputrawdata_enc - inputmeans ) / inputsigma        # normed Data
  normedinputs_mc = ( inputsmc_enc - inputmeans ) / inputsigma              # normed MC  
  
  return inputrawdata, inputrawmc, inputrawmcweight, normedinputs_data, normedinputs_mc, inputmeans, inputsigma, ncat_per_feature
  
  
# construct the ABCDnn model here
class ABCDnn(object):
  def __init__( self, inputdim_categorical_list, inputdim, nodes_cond, hidden_cond,
               nodes_trans, minibatch, activation, regularizer,
               depth, lr, gap, conddim, beta1, beta2, decay,
               retrain, savedir, savefile, disc_tag, 
               seed, permute, verbose, model_tag ):
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
    self.depth = depth
    self.lr = lr 
    self.decay = decay
    self.gap = gap
    self.conddim = conddim
    self.beta1 = beta1
    self.beta2 = beta2
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
      nodes_cond = self.nodes_cond, 
      hidden_cond = self.hidden_cond,
      nodes_trans = self.nodes_trans, 
      depth = self.depth, 
      permute = self.permute
    )
    if self.verbose: self.model.summary()
    self.optimizer = keras.optimizers.Adam(
      learning_rate = SawtoothSchedule( self.lr, self.lr * self.decay, self.gap, 0 ),  
      beta_1 = self.beta1, beta_2 = self.beta2, 
      epsilon = 1e-5, 
      name = 'nafopt'
    )
    pass

  def category_sorted(self, numpydata, verbose ):
    categoricals, categorical_cats, unique_counts = np.unique( numpydata[:, self.inputdimreal:], axis=0, return_inverse = True, return_counts = True)
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

  def setmcdata(self, numpydata, verbose, eventweight = None ):
    self.mcnumpydata = numpydata
    self.mcntotalevents = numpydata.shape[0]
    self.mcdatacounter = 0
    self.mcrandorder = np.random.permutation( self.mcnumpydata.shape[0] )
    if eventweight is not None:
      self.mceventweight = eventweight
    else:
      self.mceventweight = np.ones( ( self.mcntotalevents, 1 ) , np.float32 )
    self.categoricals_mc, self.categorical_mc_indices_grouped = self.category_sorted( self.mcnumpydata, verbose )
    pass

  def savehyperparameters(self, means, sigmas ):
    """Write hyper parameters into file
    """
    means_list = [ str( mean ) for mean in means[0] ]
    sigmas_list = [ str( sigma ) for sigma in sigmas[0] ]
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
      "REGULARIZER": self.regularizer,
      "ACTIVATION": self.activation,
      "BETA1": self.beta1, 
      "BETA2": self.beta2, 
      "MINIBATCH": self.minibatch,
      "DISC TAG": self.disc_tag,
      "INPUTMEANS": means_list,
      "INPUTSIGMAS": sigmas_list
    }
    
    with open( os.path.join( self.savedir, "{}.json".format( self.model_tag ) ), "w" ) as f:
      f.write( write_json( params, indent = 2 ) )
    #pickle.dump( params, open( os.path.join( self.savedir, "hyperparams.pkl" ), "wb" ) )

  def monitor( self, step, glossv_trn, glossv_val, mmdloss_trn, mmdloss_val  ):
    self.monitor_record.append( [ 
      step, 
      glossv_trn, glossv_val, 
      mmdloss_trn, mmdloss_val 
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
  
  def get_next_batch( self, split, size=None ):
    """Return minibatch from random ordered numpy data
    """
    if size is None:
      size = int( self.minibatch * ( ( 1. - split ) ** -1 ) )

    # reset counter if no more entries left for current batch
    if self.datacounter + size >= self.ntotalevents:
      self.datacounter = self.datacounter + size - self.ntotalevents
      self.randorder = np.random.permutation( self.numpydata.shape[0] )

    batchbegin = self.datacounter
    nextconditional = self.numpydata[ self.randorder[batchbegin], self.inputdim: ]
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

  # eqvuialent to train_step = tf.function( train_step )
  @tf.function(autograph=False)
  def train_step(self, source, target, split, sourceweight=1. ):
    source_trn, source_val, target_trn, target_val = train_test_split(
      self.this_source, self.this_target, test_size = split
    )
    
    with tf.GradientTape() as gtape:
      # get the predicted values from the current trained model
      generated_trn = tf.concat( [
          self.model( source_trn, training = True ),
          target_trn[:, -self.conddim:]
        ],
        axis = -1
      )

      #mmdloss = mix_rbf_mmd2(targetbatch[:, :self.inputdim], generated[:, :self.inputdim], sigmas=(0.1, 0.2,)) # for 1-D or 2-D
      mmdloss_trn = mix_rbf_mmd2( target_trn[:, :self.inputdim], generated_trn[:, :self.inputdim], sigmas=(0.2, 0.5, 1.0, 2.0, 4.0) ) # not clear how these sigma values were chosen
    
    gradient = gtape.gradient( mmdloss_trn, self.model.trainable_variables )
    self.optimizer.apply_gradients( zip( 
        gradient, 
        self.model.trainable_variables 
      ) 
    )

    generated_val = tf.concat( [
        self.model( source_val, training = False ),
        target_val[ :, -self.conddim:]
      ],
      axis = -1
    )

    mmdloss_val = mix_rbf_mmd2( target_val[:, :self.inputdim], generated_val[:, :self.inputdim], sigmas=(0.2, 0.5, 1.0, 2.0, 4.0) )

    glossv_trn = tf.reduce_mean( mmdloss_trn )
    glossv_val = tf.reduce_mean( mmdloss_val )

    return glossv_trn, glossv_val, mmdloss_trn, mmdloss_val

  def train( self, steps = 10000, monitor = 1000, patience = 100, early_stopping = True, split = 0.25, hpo = False, periodic_save = False ):
    # need to update this to use a validation set
    print( "{:<5} / {:<14} / {:<14} / {:<14}".format( "Epoch", "MMD (Train)", "MMD (Val)", "Min MMD (Val)" ) ) 
    impatience = 0      # don't edit
    stop_train = False  # don't edit
    self.minepoch = 0
    save_counter = 0    # edit so you aren't saving every time the model improves --> takes too much time
    for i in range( steps ):
      source, target, batchweight = self.get_next_batch( split = split )
      # train the model on this batch
      glossv_trn, glossv_val, mmdloss_trn, mmdloss_val  = self.train_step( source, target, split, batchweight )
      # currently not using glossv because mmdloss is a scalar and glossv is the mean of mmdloss
      # generator update
      if i == 0: self.minloss = mmdloss_val
      # early stopping on validation implementation
      else: 
        if mmdloss_val.numpy() < self.minloss.numpy():
          self.minepoch = i
          impatience = 0 # reset the impatience counter
          if not hpo: # don't save models during hyper parameter optimization training to save time
            if save_counter > monitor or mmdloss_val.numpy() < 0.5 * self.minloss.numpy(): # reduce number of saved models
              save_counter = 0
              self.checkpointmgr.save()
              self.model.save_weights( "./Results/{}".format( self.model_tag ) )
          self.minloss = mmdloss_val
        elif impatience > patience and early_stopping:
          print( "[WARN] Early stopping after {} epochs without improvement in loss (min loss = {:.3e})".format( i, self.minloss ) )
          stop_train = True
        else:
          impatience += 1
      if i % monitor == 0:
        print( "{:<5}   {:<14.3e}   {:<14.3e}    {:<14.3e}".format( 
          self.checkpoint.global_step.numpy(),
          mmdloss_trn.numpy(),
          mmdloss_val.numpy(),
          self.minloss
         ) )
        self.monitor( 
          self.checkpoint.global_step.numpy(), 
          glossv_trn.numpy(), glossv_val.numpy(),
          mmdloss_trn, mmdloss_val
        )
        if periodic_save: self.model.save_weights( "./Results/{}_EPOCH{}".format( self.model_tag, i ) )
      self.checkpoint.global_step.assign_add(1) # increment counter
      save_counter += 1
      if stop_train:
        if periodic_save: self.model.save_weights( "./Results/{}_EPOCH{}".format( self.model_tag, i ) ) 
        break
    if not early_stopping: 
      self.checkpointmgr.save()
      self.model.save_weights( "./Results/{}".format( self.model_tag ) )
    print( ">> Minimum loss (validation) of {:.3e} on epoch {}".format( self.minloss, self.minepoch ) )
    self.save_training_monitor()

  def display_training(self):
    # Following section is for creating movie files from trainings

    fig, ax = plt.subplots(1,2, figsize=(6,6))
    monarray = np.array(self.monitor_record)
    x = monarray[0::, 0]
    ax[0].plot(x, monarray[0::, 1], color='r', label='MMD (Training)')
    ax[0].plot(x, monarray[0::, 2], color='b', label='MMD (Validation)')
    ax[0].set_ylabel( "MMD Loss", ha = "right", y = 1.0 )
    ax[0].set_yscale('linear')
    ax[0].legend()
    ax[1].plot(x, monarray[0::, 2], color='r', label='MMD')
    ax[0].set_ylabel( "MMD", ha = "right", y = 1.0 )
    ax[1].set_yscale('log')
    ax[1].legend()

    plt.draw()

    #fig.savefig(os.path.join(self.savedir, 'trainingperf.pdf'))
    pass  

class ABCDnn_training(object):
  def __init__( self ):
    pass

  def setup_events( self, rSource, rTarget, selection, variables, regions, mc_weight ):
    # obtain the normalized data (plus others)
    self.rSource = rSource        # (str) path to source ROOT file
    self.rTarget = rTarget        # (str) path to target ROOT file
    self.selection = selection    # (dict) event selection cuts
    self.variables = variables    # (dict) variable limits and type
    self.regions = regions        # (dict) control and signal regions
    self.mc_weight = mc_weight    # (str) use xsec weights

    rawinputs, rawinputsmc, rawmcweight, normedinputs, normedinputsmc, inputmeans, \
      inputsigma, ncat_per_feature = prepdata( rSource, rTarget, 
      selection, variables, mc_weight )

    self.rawinputs = rawinputs                # unnormalized data tree after event selection
    self.rawinputsmc = rawinputsmc            # unnormalized mc tree after event selection
    self.rawmcweight = rawmcweight            # xsec weighted mc tree after event selection
    self.normedinputs = normedinputs          # normalized data tree after event selection
    self.normedinputsmc = normedinputsmc      # normalized mc tree after event selection
    self.inputmeans = inputmeans              # mean of unnormalized data
    self.inputsigma = inputsigma              # rms of unnormalized data

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

    if self.rawmcweight is None:
      self.bkgmcweight = None
    else:
      self.bkgmcweight = self.rawmcweight[ self.bkg_select_mc ]
  
    pass

  def setup_model( self, 
          nodes_cond = 8, hidden_cond = 1, nodes_trans = 8, minibatch = 64, lr = 5.0e-3,
          depth = 1, activation = "swish", regularizer = "None", decay = 1e-1,
          gap = 1000., beta1 = 0.9, beta2 = 0.999,
          savedir = "/ABCDNN/", savefile = "abcdnn.pkl", disc_tag = "ABCDnn", mc_weight = None, 
          retrain = False, seed = 100, permute = True, model_tag = "best_model", verbose = False
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
    self.regularizer = regularizer  # (str) use of L1, L2, L1+L2 or None for training regularizers
    self.activation = activation    # (str) activation function for conditional layers
    self.savedir = savedir          # (str) save directory for output results
    self.savefile = savefile        # (str) save file name for output results
    self.disc_tag = disc_tag        # (str) tag to add to transformed variable name
    self.seed = seed                # (int) RNG seed
    self.retrain = retrain          # (bool) start a new training or continue from where left-off
    self.mc_weight = mc_weight      # (str) weight MC values according to xsec
    self.permute = permute          # (bool) run random permutations of the conditional inputs
    self.model_tag = model_tag

    self.model = ABCDnn( 
      inputdim_categorical_list = self.ncat_per_feature, 
      inputdim = self.inputdim,
      conddim = self.conddim,
      nodes_cond = self.nodes_cond,
      hidden_cond = self.hidden_cond,
      nodes_trans = self.nodes_trans, 
      minibatch = self.minibatch,
      activation = self.activation,
      regularizer = self.regularizer,
      depth = self.depth, 
      lr = self.lr,
      decay = self.decay, 
      gap = self.gap,
      beta1 = self.beta1, 
      beta2 = self.beta2,
      retrain = self.retrain, 
      savedir = self.savedir, 
      savefile = self.savefile,
      disc_tag = self.disc_tag,
      seed = self.seed, 
      permute = self.permute, 
      model_tag = self.model_tag,
      verbose = verbose
    )

    self.model.setrealdata( self.normedinputs_bkg, verbose = verbose )
    self.model.setmcdata( self.normedinputsmc_bkg, eventweight = self.bkgmcweight, verbose = verbose )
    pass

  def train( self, steps = 10000, monitor = 1000, patience = 100, early_stopping = True, split = 0.25, display_loss = False, save_hp = False, hpo = False, periodic_save = False ):
    self.model.train( steps = steps, monitor = monitor, patience = patience, split = split, early_stopping = early_stopping, hpo = hpo, periodic_save = periodic_save )
    if save_hp: self.model.savehyperparameters( self.inputmeans, self.inputsigma )
    pass

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
    self.fakedata = self.fakedata * self.inputsigma[ :, :self.inputdim ] + self.inputmeans[ :, :self.inputdim ]

    pass

  def evaluate_regions( self, hpo = False, verbose = True ):
    self.select = {
        "DATA": {},
        "MC": {}
    }
    self.count = {
        "DATA": {},
        "MC": {}
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
          self.select[ "DATA" ][ self.region[i] ] &= ( self.rawinputs[ self.regions[ "Y" ][ "VARIABLE" ] ] >= y  )
          self.select[ "MC" ][ self.region[i] ] &= ( self.rawinputsmc[ self.regions[ "Y" ][ "VARIABLE" ] ] >= y )
        else:
          self.select[ "DATA" ][ self.region[i] ] &= ( self.rawinputs[ self.regions[ "Y" ][ "VARIABLE" ] ] == y )
          self.select[ "MC" ][ self.region[i] ] &= ( self.rawinputsmc[ self.regions[ "Y" ][ "VARIABLE" ] ] == y )

        self.count[ "DATA" ][ self.region[i] ] = np.count_nonzero( self.select[ "DATA" ][ self.region[i] ] )
        self.count[ "MC" ][ self.region[i] ] = np.count_nonzero( self.select[ "MC" ][ self.region[i] ] )
        x_eq = ">=" if ( self.regions[ "X" ][ "INCLUSIVE" ] ) and ( x == self.regions[ "X" ][ "MAX" ] ) else "=="
        y_eq = ">=" if ( self.regions[ "Y" ][ "INCLUSIVE" ] ) and ( y == self.regions[ "Y" ][ "MAX" ] ) else "=="
        if verbose:
          print( "Region {} ({} {} {}, {} {} {}): MC = {}, DATA = {} ".format(
              self.region[i], 
              self.regions[ "X" ][ "VARIABLE" ], x_eq, int( x ),
              self.regions[ "Y" ][ "VARIABLE" ], y_eq, int( y ),
              int( self.count[ "MC" ][ self.region[i] ] ), int( self.count[ "DATA" ][ self.region[i] ] )
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
      
        # format the data in a plottable/saveable manner
        input_data = self.normedinputs[ self.select[ "DATA" ][ self.region[i] ] ]
        raw_data = self.rawinputs.loc[ self.select[ "DATA" ][ self.region[i] ] ]
        self.rawdata.append( raw_data )

        input_mc = self.normedinputsmc[ self.select[ "MC" ][ self.region[i] ] ]
        raw_mc = self.rawinputsmc.loc[ self.select[ "MC" ][ self.region[i] ] ]
        self.rawmc.append( raw_mc )

        if self.rawmcweight is None:
          mcweight = [1.] * input_mc.shape[0]
        else:
          mcweight = self.rawmcweight[ self.select[ "MC" ][ self.region[i] ] ].flatten()
        self.mcweight.append( mcweight )

        n_batches = int( input_mc.shape[0] / self.minibatch )
        fake = []
        for j in range( n_batches - 1 ):
          xin = input_mc[ j * self.minibatch: ( j + 1 ) * self.minibatch:, : ]
          xgen = self.model.model.predict( xin )
          fake.append( xgen )
        xin = input_mc[ ( n_batches - 1 ) * self.minibatch:, : ]
        xgen = self.model.model.predict( xin )
        fake.append( xgen )

        this_fakedata = np.vstack( fake )
        this_fakedata = this_fakedata * self.inputsigma[ :, :self.inputdim ] + self.inputmeans[ :, :self.inputdim ]
        n_fakes = this_fakedata.shape[0]
        this_fakedata = np.hstack( ( this_fakedata, 
                                  np.array( [x] * n_fakes ).reshape( ( n_fakes, 1 ) ), 
                                  np.array( [y] * n_fakes ).reshape( ( n_fakes, 1 ) ) 
                                  ) )
        self.fakedata.append( this_fakedata )

        i += 1
      
    

  def extended_ABCD( self ):
    # compute the norm of the raw data
    dA, dB, dC, dD, dX, dY = self.count[ "DATA" ][ "A" ], self.count[ "DATA" ][ "B" ], self.count[ "DATA" ][ "C" ], self.count[ "DATA" ][ "D" ], self.count[ "DATA" ][ "X" ], self.count[ "DATA" ][ "Y" ]
    self.data_norm = dB * dX * dC**2 / ( dA**2 * dY )
    self.sigma_data_norm = np.sqrt( 1/dB + 1/dX + 1/dY + 4/dC + 4/dA ) * self.data_norm
    print( "Ext. ABCD Prediction: {:.2f} pm {:.2f}".format( self.data_norm, self.sigma_data_norm ) )
    print( "True Count: {}".format( dD ) )
    pass

  def plot_ratio( self, 
      plot_data = True, plot_mc = False, plot_cr = False, plot_sr = True, 
      errorbars = True, normed = True, ratio = [ 0.25, 2.0 ], 
      yscales = ["linear"], n_bins = 20, plot_ks = False, save = False
    ):
    vars = [ str( var ) for var in self.variables ]
  
    for yscale in yscales:
      for var in self.variables:
        i = 0
        if self.variables[ var ][ "TRANSFORM" ] == True:
          for _fakedata, _rawdata, _rawmc, _mcweight, _plottext in zip( self.fakedata, self.rawdata, self.rawmc, self.mcweight, self.plottext ):
            if not plot_cr and self.region[i] in [ "A", "B", "C", "X", "Y" ]: continue
            if not plot_sr and self.region[i] == "D": continue
            sNormed = "NORM" if normed else "UNNORM"
            if plot_data:
              plt.figure( figsize = ( 8, 8 ) )
              axes = MplPlotter.ratio_plot( 
                dict(
                  x = _rawdata[ var ], bins = n_bins, label = "Raw Data",
                  range = ( self.variables[ var ][ "LIMIT" ][0], self.variables[ var ][ "LIMIT" ][1] ),
                  errorbars = errorbars, normed = normed, histtype = "marker"
                ),
                dict(
                  x = _fakedata[ :, vars.index( var ) ], bins = n_bins, label = "Fake Data",
                  range = ( self.variables[ var ][ "LIMIT" ][0], self.variables[ var ][ "LIMIT" ][1] ),
                  errorbars = errorbars, normed = normed
                ),
                ratio_range = ( ratio[0], ratio[1] )
              )

              ks_text = ""
              if plot_ks:
                D, p = stats.ks_2samp( _rawdata[ var ], _fakedata[ :, vars.index( var ) ] )
                ks_text = "$p_{}={:.2e}$".format( "{KS}", p )

              axes[0][0].set_yscale( yscale )
              if normed: axes[0][0].set_ylabel( "Normalized Count", ha = "right", y = 1.0 )
              else: axes[0][0].set_ylabel( "Count", ha = "right", y = 1.0 )
              plt.xlabel( "${}$".format( self.variables[ var ][ "LATEX" ] ), ha = "right", x = 1.0 )

              if self.region[i] == "D":
                axes[0][0].set_title( "Data Signal Region {}: {} ({})".format( self.region[i], _plottext, ks_text ), ha = "right", x = 1.0 )
              else:
                axes[0][0].set_title( "Data Control Region {}: {} ({})".format( self.region[i], _plottext, ks_text ), ha = "right", x = 1.0 )
              if save: plt.savefig( os.path.join( self.savedir, "{}_ratio_DATA_r{}_{}.pdf".format( str(var), self.region[i], sNormed ) ) )
              axes[0][0].legend( loc = "best" )
              plt.show()
              plt.close()

            if plot_mc:
              plt.figure( figsize = ( 8, 8 ) )
              axes = MplPlotter.ratio_plot(
                dict(
                  x = _rawmc[ var ], bins = n_bins, label = "MC",
                  range = ( self.variables[ var ][ "LIMIT" ][0], self.variables[ var ][ "LIMIT" ][1] ),
                  errorbars = errorbars, normed = normed, histtype = "marker"
                ),
                dict(
                  x = _fakedata[ :, vars.index( var ) ], bins = n_bins, label = "Fake Data",
                  range = ( self.variables[ var ][ "LIMIT" ][0], self.variables[ var ][ "LIMIT" ][1] ),
                  errorbars = errorbars, normed = normed
                ),
                ratio_range = ( ratio[0], ratio[1] )
              )

              ks_text = ""
              if plot_ks:
                D, p = stats.ks_2samp( _rawdata[ var ], _fakedata[ :, vars.index( var ) ] )
                ks_text = "$p_{}={:.2e}$".format( "{KS}", p )

              axes[0][0].set_yscale( yscale )
              if normed: axes[0][0].set_ylabel( "Normalized Count", ha = "right", y = 1.0 )
              else: axes[0][0].set_ylabel( "Count", ha = "right", y = 1.0 )
              plt.xlabel( "${}$".format( self.variables[ var ][ "LATEX" ] ), ha = "right", x = 1.0 )
              if self.region[i] == "D":
                axes[0][0].set_title( "MC Signal Region {}: {} ({})".format( self.region[i], _plottext, ks_text ), ha = "right", x = 1.0 )
              else:
                axes[0][0].set_title( "MC Control Region {}: {} ({})".format( self.region[i], _plottext, ks_text ), ha = "right", x = 1.0 )
              if save: plt.savefig( os.path.join( self.savedir, "{}_ratio_MC_r{}_{}.pdf".format( str(var), self.region[i], sNormed ) ) )
              axes[0][0].legend( loc = "best" )
              plt.show()

            i += 1
    pass

  def plot_hist( self,
      plot_data = True, plot_mc = False, plot_cr = False, plot_sr = True,
      errorbars = True, normed = True,
      yscales = [ "linear" ], n_bins = 20, save = False 
    ):
    vars = [ str( var ) for var in self.variables ]

    for yscale in yscales:
      for var in vars:
        if self.variables[ var ][ "TRANSFORM" ] == True:
          i = 0
          for _fakedata, _rawdata, _rawmc, _mcweight, _plottext in zip( self.fakedata, self.rawdata, self.rawmc, self.mcweight, self.plottext ):
            if not plot_cr and self.region[i] in [ "A", "B", "C", "X", "Y" ]: continue
            if not plot_sr and self.region[i] == "D": continue
            sNormed = "NORM" if normed else "UNNORM"
            plt.figure( figsize = ( 8, 8 ) )
            if self.variables[ var ][ "TRANSFORM" ] == True:
              plt.figure()
              MplPlotter.hist( 
                  _rawdata[ var ], bins = n_bins, label = "Raw Data",
                  range = ( self.variables[ var ][ "LIMIT" ][0], self.variables[ var ][ "LIMIT" ][1] ),
                  errorbars = errorbars, normed = normed, histtype = "marker", alpha = 0.5
              )
              if self.mc_weight is None:
                if plot_data: MplPlotter.hist(
                  _fakedata[ :, vars.index( var ) ], bins = n_bins, label = "Predicted",
                  range = ( self.variables[ var ][ "LIMIT" ][0], self.variables[ var ][ "LIMIT" ][1] ),
                  errorbars = errorbars, normed = normed, alpha = 0.5
                )
                if plot_mc: MplPlotter.hist(
                  _rawmc[ var ], bins = n_bins, label = "MC",
                  range = ( self.variables[ var ][ "LIMIT" ][0], self.variables[ var ][ "LIMIT" ][1] ),
                  errorbars = errorbars, normed = normed, histtype = "step", alpha = 0.5
                )
              else:
                if plot_data: MplPlotter.hist(
                  _fakedata[ :, vars.index( var ) ], bins = n_bins, label = "Fake Data",
                  range = ( self.variables[ var ][ "LIMIT" ][0], self.variables[ var ][ "LIMIT" ][1] ),
                  errorbars = errorbars, normed = normed, alpha = 0.5, weights = _mcweight
                )
                if plot_mc: MplPlotter.hist(
                  _rawmc[ var ], bins = n_bins, label = "MC",
                  range = ( self.variables[ var ][ "LIMIT" ][0], self.variables[ var ][ "LIMIT" ][1] ),
                  errorbars = errorbars, normed = normed, histtype = "step", alpha = 0.5, weights = _mcweight
                )
              plt.yscale( yscale )
              if normed: plt.ylabel( "Normalized Count", ha = "right", y = 1.0 )
              else: plt.ylabel( "Count", ha = "right", y = 1.0 )
              plt.xlabel( "${}$".format( self.variables[ var ][ "LATEX" ] ), ha = "right", x = 1.0 )
              plt.legend()
              if self.region[i] == "D":
                plt.title( "Signal Region {}: {}".format( self.region[i], _plottext ), ha = "right", x = 1.0 )
              else:
                plt.title( "Control Region {}: {}".format( self.region[i], _plottext ), ha = "right", x = 1.0 )
              if save: plt.savefig( os.path.join( self.savedir, "{}_hist_r{}_e{}_{}_{}.pdf".format( str(var), self.region[i], sNormed, yscale.upper() ) ) )
              plt.show()
              plt.close()
            i += 1

  # plot the correlation between the fake data and raw mc for each variable
  # the expectation is that there should be 
  def plot_correlation( self, n_sample = 5000 ):
    xmin = self.regions[ "X" ][ "MIN" ]
    xmax = self.regions[ "X" ][ "MAX" ]
    xlen = xmax - xmin + 1
    ymin = self.regions[ "Y" ][ "MIN" ]
    ymax = self.regions[ "Y" ][ "MAX" ]
    ylen = ymax - ymin + 1
    vars = [ str( var ) for var in self.variables ]
    for var in vars:
      if self.variables[ var ][ "TRANSFORM" ] == True:
        i = 0
        fig, ax = plt.subplots( int( xlen ), int( ylen ), figsize = ( 8, 9 ) )
        for xi, x in enumerate( np.linspace( xmin, xmax, xlen ) ):
          for yi, y in enumerate( np.linspace( ymin, ymax, ylen ) ):
            lower = self.variables[ var ][ "LIMIT" ][0]
            upper = self.variables[ var ][ "LIMIT" ][1]
            ax[ xi, yi ].plot( 
                self.rawmc[i].iloc[ :n_sample, vars.index( var ) ], 
                self.fakedata[i][ :n_sample, vars.index( var ) ], 
                ",", markersize = 5
                )
            ax[ xi, yi ].set_xlim( lower, upper )
            ax[ xi, yi ].set_ylim( lower, upper )
            ax[ xi, yi ].set_xlabel( "MC", ha = "right", x = 1.0 )
            ax[ xi, yi ].set_ylabel( "PREDICT DATA", ha = "right", y = 1.0 )
            xVar = self.variables[ self.regions[ "X" ][ "VARIABLE" ] ][ "LATEX" ]
            yVar = self.variables[ self.regions[ "Y" ][ "VARIABLE" ] ][ "LATEX" ]
            ax[ xi, yi ].set_title( "Region {} $({},{})=({},{})$: ${}$".format(
              self.region[i], xVar, yVar, int(x), int(y),
              self.variables[ var ][ "LATEX" ]
            ),
            ha = "right", x = 1.0, fontsize = 10 )
            ax[ xi, yi ].plot( [ lower, upper ], [ lower, upper ], "r--" )
            ax[ xi, yi ].grid()

            i += 1
        fig.tight_layout()
        fig.show()
    pass 
  
  def save_root( self, tree_name = "mytree", file_name = "data.root" ):
    self.save_data = {}
    self.branch = {}
    self.fake_data_all = np.vstack( self.fakedata )
    vars = [ str( var ) for var in self.variables ]
    for i, var in enumerate( vars ):
      self.save_data[ var ] = self.fake_data_all[ :, i ]
    self.mcweight_all = np.concatenate( self.mcweight )
    self.save_data[ "weight" ] = self.mcweight_all
    for key, data in self.save_data.items():
      self.branch[ key ] = data.dtype
    tree = uproot.newtree( branches = self.branch )
    with uproot.recreate( file_name ) as f:
      f[ tree_name ] = tree
      f[ tree_name ].extend( self.save_data )
    pass
