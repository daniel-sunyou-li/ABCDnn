# ABCDnn

To run the notebook on the LPC (works on CMSSW_9_4_6_patch1 and CMSSW_10_6_29):
1. First, set up the environment with the following commands (can use `cat start.txt` to quickly retrieve):

		source /cvmfs/cms.cern.ch/cmsset_default.csh
		cmsenv
		source /cvmfs/sft.cern.ch/lcg/views/LCG_98/x86_64-centos7-gcc8-opt/setup.csh
    
2. Second, edit the settings in `config.py`:
* `eosUserName` -- Edit this based on your username
* `postfix` -- Used in `sampleDir`
* `sourceDir` -- Edit the paths to your relevant files on LPC or BRUX
* `sampleDir` -- Edit this for the directory containing ROOT files
* `variables` -- Edit this to define your two transformed variables and two control variables (branch names in ROOT file)
* `selection` -- Edit this for event selection, applied in `format_ntuples.py`
* `regions` -- Edit this to define the control region boundaries for each training instance, the minimum and maximum boundaries must match the limits in `variables` for encoding to work
* `params` -- Edit the model architecture you want and training settings
> * `EPOCHS` -- Number of training instances, recommended to test `1000` first
> * `MONITOR` -- Steps between each epoch to report (and save) models, recommended to save only 20 models
> * `MONITOR THRESHOLD` -- Only monitor events after this epoch number
* `samples_input` -- Edit the name of ROOT files to format into a single pickle file used in training
* `samples_apply` -- Should match `samples_input`, but can specify additional files, such as QCD

3. Create your dataset using `format_ntuple.py`, showing example for 2017 as an example,

		python format_ntuple.py -y 2017 -sO ttbar_17 -tO singleLep_17 -v DNN_1to40_3t AK4HT -p 10 -l BRUX --doMC
		python format_ntuple.py -y 2017 -sO ttbar_17 -tO singleLep_17 -v DNN_1to40_3t AK4HT -p 100 -l BRUX --doData
		
You can also include the flags `--JECup` or `--JECdown` to use JEC shifted samples instead of nominal. It is recommended to not include all of the MC events, as there are probably more than you need. Training will be fine if you have at least 10K events per region. 

4. After you have your datasets (named `ttbar_17_mc.rppt` and `singleLep_17_data.root`), you can run training on the dataset. Let's say that you have boundaries of NJ = {4,5,6+} and NB = {2,3+}, then you can use the naming,

		python train_abcdnn.py -y 2017 -s ttbar_17_mc.root -t singleLep_17_data.root -m model_17 -d nJ6pnB3p 
		
There is an argument available for `--hpo` for hyper parameter optimization, but this is not recommended as the optimization does not converge. The flag `--randomize` will choose a different seed for the RNG. The flags `-m` refer to the model name you will see contained in `Results/` and `-d` refers to the postfix of the discriminator you would add to an ABCDnn ntuple.

5. After training, you want to check which model to retain at a given epoch:

		python plot_training -s ttbar_17_mc.root -t singleLep_17_data.root -m model_17
		
After training, you will have two `.gif` files in `Results/` that you can view to see the model performance at each monitored epoch. The recommended way to view is to `xrdcp` to EOS and view through a GUI interface like SWAN if no in-terminal tools are available for viewing directly. After selecting a model at a given epoch to retain, manually change the name of `model_17_EPOCH###.data-00000-of-00001` to `model_17.data-0000-of-00001` and `model_17_EPOCH###.index` to `model_17.index` as the application script automatically chooses the model without the epoch postfix. 

6. To run application and create new ABCDnn ntuples, use:

		python apply_abcdnn.py -c model_17 -y 2017 -l BRUX -s EOS 
		
You can specify multiple checkpoints/models if you want. There are optional flags:
* `--test` -- run on a single file
* `--closure` -- include a branch with closure systematic shift up and down (necessary to manually compute and add to the `.json` file)
* `--JECup` -- Apply the model to the JECup shifted samples
* `--JECdown` -- Apply the model to the JECdown shifted samples

## __Optional Scripts__
* `analyze_regions.py` will print out the event yield in your given ntuple based on two control regions defined
* `evaluate_model.py` will report several statistics such as the Extended ABCD yield estimate, the loss in the signal region, the loss and variance with and without dropout, and percent different in predicted versus observed yield
