#### molscontrol

***

An on-the-fly job control system to manage your DFT geometry optimizations for inorganic discovery. With a combination of the dynamic classifier (predictor) and latent space entropy (confidence metric), this job control system can offer an "effective" two-fold acceleration for your calculation by killing simulations that has a low expected success rate. More details see https://pubs.acs.org/doi/10.1021/acs.jctc.9b00057 or https://doi.org/10.26434/chemrxiv.7616009.v1.

##### Requirement

Keras, Tensorflow, scipy, yaml, pillow.

For molSimplify users, please either (re)run "python setup.py install" if you installed molSimplify from the source or download our most recent conda version of molSimplify to activate molscontrol.

##### Usage

molscontrol is a job control system so it needs to be run with a quantum chemistry process in parallel. An example bash script to do geometry optimization with Terachem is

```bash
module load anaconda
source activate mols_keras
module load terachem/tip

terachem terachem_input > dft.out &
PID_KILL=$!
molscontrol $PID_KILL > control.out &
wait
```

where molscontrol takes in the PID of the terachem simulation in and monitors this process on-the-fly. Prediction probability and latent space entropy at each step are recorded in the log file. Besides the PID of your quantum chemistry calculation, molscontrol requires a configure file in your current working directory where the jobscript is submitted. Examples of configure.json can be found in the tests folder and each keyword are explained in detail at dynamic_classifier.py.

##### Interface with quantum chemistry codes

Currently two modes are implemented, "terachem" and "geo".  In "terachem" mode, the dynamic classifier utilizes outputs of gradients, Mulliken charges, bond order matrix and geometry (all in terachem printout format) at each step to make predictions on the final job label. For non-Terachem users, a "geo" mode is available where only the optimization trajectory is required as the input for the dynamic classifier, which should be readily obtained in quantum chemistry packages. In addition, users are also welcomed to implement your own models based on the outputs of your quantum chemistry package of choice by filling out the "costum" dictionary in dynamic_classifier.py.

##### Other useful info

For the motivation of background of molscontrol, please refer to our tutorials https://hjkgrp.mit.edu/content/molsimplify-tutorial-12-using-static-classifier-predict-your-simulation-outcomes-they-waste and https://hjkgrp.mit.edu/content/molsimplify-tutorial-13-molscontrol-intelligent-job-control-system-manage-your-dft-geometry.
