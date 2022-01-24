# this script runs the condor job for applying ABCDnn to step3
#!/bin/sh

condorDir=${1}
sampleNameIn=${2}
sampleNameOut=${3}
sampleDir=${4}
weight=${5}
tag=${6}

xrdcp -s $condorDir/CMSSW946_ttt.tgz .
tar -xf ABCDnn.tgz
rm ABCDnn.tgz

cd CMSSW_9_4_6_patch1/src/TTT-singleLep/ABCDnn/

source /cvmfs/cms.cern.ch/cmsset_default.csh
cmsenv
source /cvmfs/sft.cern.ch/lcg/views/LCG_98/x86_64-centos7-gcc8-opt/setup.csh

python remote_abcdnn.py -j Results/hyperparams.json -s $condorDir/$sampleDir/$tag/$sampleNameIn -c Results/$weight 

xrdcp -f $sampleNameOut $condorDir/$sampleDir 
