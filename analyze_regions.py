# this script takes in multiple ROOT files and parses them for the event multiplicity in a 2D phase space of two control variables
# last modified September 29, 2021 by Daniel Li

import ROOT
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument( "-f", "--files", nargs = "+", required = True, help = "ROOT files to read in" )
parser.add_argument( "-x1", "--x1", default = "NJets_JetSubCalc", help = "First control variable" )
parser.add_argument( "-x2", "--x2", default = "NJetsCSV_MultiLepCalc", help = "Second control variable" )
args = parser.parse_args()

rFiles = {}
rTrees = {}

for file in args.files:
  rFiles[ file ] = ROOT.TFile.Open( file )
  rTrees[ file ] = rFiles[ file ].Get( "Events" )

# get event information
n_event = {}
for file in rTrees:
  branches = [ branch.GetName() for branch in rTrees[ file ].GetListOfBranches() ]
  if args.x1 not in branches:
    print("[ERR] {} is not a valid branch, exiting...".format( args.x1 ) )
    quit()
  if args.x2 not in branches:
    print("[ERR] {} is not a valid branch, exiting...".format( args.x2 ) )
    
  n_event[ file ] = {}
  range_x1 = [0,0]
  range_x2 = [0,0]
  for i in range( rTrees[ file ].GetEntries() ):
    rTrees[ file ].GetEntry(i)
    x1 = getattr( rTrees[ file ], args.x1 )
    x2 = getattr( rTrees[ file ], args.x2 )
    
    if x1 > range_x1[1]: range_x1[1] = x1
    if x2 > range_x2[1]: range_x2[1] = x2
    
    if x1 not in list( n_event[ file ].keys() ):
      n_event[ file ][ x1 ] = {}
    if x2 not in list( n_event[ file ][ x1 ].keys() ):
      n_event[ file ][ x1 ][ x2 ] = 0
    n_event[ file ][ x1 ][ x2 ] += 1
      
  for i in range( range_x1[0], range_x1[1] ):
    if i not in list( n_event[ file ].keys() ):
      n_event[ file ][ i ] = {}
    for j in range( range_x2[0], range_x2[1] ):
      if j not in list( n_event[ file ][ i ].keys() ):
        n_event[ file ][ i ][ j ] = 0
      
  print( ">> ({},{}) event count for {}".format( args.x1, args.x2, file ) )
  print( "" )
  print( "     " ),
  for i in range( range_x1[0], range_x1[1] ):
    print( "{:<8}".format( i ) ),
  print( "" )
  print( "     " ),
  for i in range( range_x1[0], range_x1[1] ):
    print( "{:<8}".format( "________" ) ),
  print( "" )
  for j in range( range_x2[0], range_x2[1] ):
    print( "{:<3} |".format( j ) ),
    for i in range( range_x1[0], range_x1[1]  ):
      print( "{:<8}".format( n_event[ file ][ i ][ j ] ) ),
    print( " " )
   
