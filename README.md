<!--<h3><b>Colorful Image Colorization</b></h3>-->
## <b>Colorful Image Colorization</b> [[Project Page]](http://richzhang.github.io/colorization/) <br>
[Richard Zhang](https://richzhang.github.io/), [Phillip Isola](http://web.mit.edu/phillipi/), [Alexei A. Efros](http://www.eecs.berkeley.edu/~efros/). In [ECCV, 2016](http://arxiv.org/pdf/1603.08511.pdf).


### Overview ###
This repository contains:

<b>Colorization-centric functionality</b>
 - (0) a test time script to colorize an image (python script)
 - (1) a test time demonstration (IPython Notebook)
 - (2) code for training a colorization network
 - (3) links to our results on the ImageNet test set, along with a pointer to AMT real vs fake test code

<b>Representation Learning-centric functionality</b>
 - (4) pre-trained AlexNet, used for representation learning tests (Section 3.2)
 - (5) code for training AlexNet with colorization
 - (6) representation learning tests

<b>Appendices</b>
 - (A) Related follow-up work

### Clone this repository ###
Clone the master branch of the respository using `git clone -b master --single-branch https://github.com/richzhang/colorization.git`

### Dependencies ###
This code requires a working installation of [Caffe](http://caffe.berkeleyvision.org/) and basic Python libraries (numpy, pyplot, skimage, scipy). For guidelines and help with installation of Caffe, consult the [installation guide](http://caffe.berkeleyvision.org/) and [Caffe users group](https://groups.google.com/forum/#!forum/caffe-users).
