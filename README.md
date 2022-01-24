# ABCDnn

To run the notebook on the LPC GPUs:  
1. First, sign-in to a custom `localhost` instance of the LPC where `###` are any three numbers of your choice:

		ssh -L localhost:8###:localhost:8### <username>@cmslpc-sl7.fnal.gov
    
2. Second, start a singularity instance on the LPC:  

		singularity --nv --bind `readlink $HOME` --bind `readlink -f ${HOME}/nobackup/` --bind /cvmfs /cvmfs/unpacked.cern.ch/registry.hub.docker.com/fnallpc/fnallpc-docker:tensorflow-latest-devel-gpu-singularity
		jupyter notebook --no-browser --ip=127.0.0.1 --port=8###
