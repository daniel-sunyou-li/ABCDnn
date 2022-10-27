#!/bin/sh

condorDir=${1}
sourceDir=${2}
targetDir=${3}
sampleDir=${4}
sampleNameIn=${5}
sampleNameOut=${6}
checkpoints=${7}
closure=${8}
tag=${9}
year=${10}

#xrdcp -f $condorDir/ABCDnn.tgz .
#tar -xf ABCDnn.tgz
#rm ABCDnn.tgz
echo "condorDir = $condorDir"
echo "sourceDir = $sourceDir"
echo "targetDir = $targetDir"
echo "sampleDir = $sampleDir"
echo "checkpoints = $checkpoints"
echo "closure = $closure"
echo "tag = $tag"
echo "year = $year"

mkdir Results
mv *.data* *.json *.index Results/

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc7_amd64_gcc700
scramv1 project CMSSW CMSSW_10_6_29
cd CMSSW_10_6_29
eval `scramv1 runtime -sh`
source /cvmfs/sft.cern.ch/lcg/views/LCG_98/x86_64-centos7-gcc8-opt/setup.sh
#pip install --user "tensorflow-probability<0.9"
cd -

if [[ $tag  == "JECup" ]]; then
echo "Running $year JECup $sampleNameIn"
python apply_abcdnn.py -y $year -f $sampleNameIn\.root -c $checkpoints -l $sourceDir -s $targetDir --JECup --condor
elif [[ $tag == "JECdown" ]]; then
echo "Running $year JECdown $sampleNameIn"
python apply_abcdnn.py -y $year -f $sampleNameIn\.root -c $checkpoints -l $sourceDir -s $targetDir --JECdown --condor
elif [[ $tag == "nominal" ]]; then
echo "Running $year nominal $sampleNameIn"
python apply_abcdnn.py -y $year -f $sampleNameIn\.root -c $checkpoints --closure $closure -l $sourceDir -s $targetDir --condor
fi

xrdcp -fp samples_ABCDNN_UL$year/$tag/$sampleNameOut\.root $sampleDir
