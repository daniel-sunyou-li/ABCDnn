#!/bin/sh

condorDir=${1}
sampleNameIn=${2}
sampleNameOut=${3}
sampleDir=${4}
checkpoints=${5}

xrdcp -f $condorDir/ABCDnn.tgz .
tar -xf ABCDnn.tgz
rm ABCDnn.tgz

mv *.data* *.json *.index CMSSW_10_6_19/src/ABCDnn/

cd CMSSW_10_6_19/src/ABCDnn/

source /cvmfs/cms.cern.ch/cmsset_default.sh
cmsenv
source /cvmfs/sft.cern.ch/lcg/views/LCG_98/x86_64-centos7-gcc8-opt/setup.sh

pip install --user "tensorflow-probability<0.9"

python remote_abcdnn.py -s $sampleDir/$sampleNameIn -c $checkpoints

xrdcp -f $sampleNameOut\.root $condorDir/$sampleDir
