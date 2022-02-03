# this script runs the condor job for applying ABCDnn to step3
#!/bin/sh

condorDir=${1}
sampleNameIn=${2}
sampleNameOut=${3}
sampleDir=${4}
checkpoints=${5}

xrdcp -s $condorDir/ABCDnn.tgz .
tar -xf ABCDnn.tgz
rm ABCDnn.tgz

cd CMSSW_10_6_19/src/ABCDnn/

source /cvmfs/cms.cern.ch/cmsset_default.sh
cmsenv
source /cvmfs/sft.cern.ch/lcg/views/LCG_98/x86_64-centos7-gcc8-opt/setup.sh

python remote_abcdnn.py -s $sampleDir/$sampleNameIn -c $checkpoints

xrdcp -f $sampleNameOut $condorDir/$sampleDir
