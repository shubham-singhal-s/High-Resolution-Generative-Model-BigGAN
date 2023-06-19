# BigGAN for 256 resolution images

This project is used to train a BigGAN model on a single class of images. The scripts can be used to format the images, train the model, get inference images and make a nifty animated GIF of the training process.

## Usage
This project uses the SLURM scheduling utility to run the model traning on a A100 GPU. The model requires at least 36 GB of GPU memory to run.

### Processing images
The model works on a dataset of 256x256 pixel RGB images. To process a dataset of higher resolution images, the pre-processing script at `utils/compress_images_sqaure` can be used. The path for the original images must be updated before usage. The script stores the processed images under the directory that the model reads its training set from, i.e. `data/compressed`.

### Running the model
The model uses Tensorflow, Keras , Tensorflow Utils and other supporting libraries to create a BigGAN architecture for generating mountain images.
The model can be run using the `BigGAN.py` script, either directly, or scheduled via SLURM, using the `run_gan.slurm` file.

**Directly via Python**   
`python BigGAN.py <folder> <offset>`   
*folder*: A suffix added to the end of all output folders where model outputs (images and weights) are stored. Useful when training multiple models. Can be left blank to avoid the addition of a suffix (with "").   
*offset*: An offset for the epochs, used when one needs to continue the training of the model, once one round of training is finished. If passed, loads weights from the output directory and continues training by adding the offset to the number of epochs.

**Via SLURM**   
Running the script: `sbatch run_gan.slurm`   
Checking Job status: `squeue`   
Stopping the job: `scancel <job_id>` (job_id is returned when using `sbatch` or can be fetched via `squeue`)   

Note: The SLURM parameters and the system arguaments stated above need to be configured propery in the SLURM file before running the job.

### Inference
The script at `utils/generate_outputs_from_weights` can be used to generate output images once the model is trained and the weights are obtained. The script can be run directly or scheduled via slurm   

**Directly via Python**   
`python generate_outputs_from_weights.py <num> <file>`   
*num*: Number of output images to generate, defaults to 1.   
*file*: Name of the file containing the generator weights. Does not need the suffix. Must be a `.h5` file. Deafults to 'gen'.   

**Via SLURM**   
`sbatch generate_outputs_from_weights.slurm`   

### GIF
The script at `utils/make_gif.py` can be used to create a GIF of the training process. 

**Usage**
`python make_gif.py <max_samples> <suffix>`   
*max_samples*: Number of images from the output directory `genned` to include, defaults to 3010 (for 3000 epochs)   
*suffix*: Folder suffix, similar to the one used in training.   

### Addtional Info
The SLURM scripts generate two files, one for standard out (STDOUT) and one for standard errors (STDERR), which contain the outputs produced via the Python scripts. These used for deugging errors with the model.

The model requires a minimum of 36 GB GPU memory, otherwise OOM (Out Of Memory) errors are thrown.

If running on the WSU Wolfe server, please use the SLURM scheduling to run the model on the corresponding GPU node.

**The model is scaled down to fit into the GPU memory, if you need to mimic the original BigGAN architecture, please change the `dim` variable in both generator and discriminator to `64` and the Batch Size to 256+.**

**The model ideally needs more than 2000 images for training, to produce a wide variety of images.**

**The current implementation converges in 3000 epochs, but if the trainig set, batch size or the architecture is altered, the time of convergeeance might change.**
