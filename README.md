Overview
============
A project that trains a LSTM recurrent neural network over a dataset of MIDI files. More information can be found on the [writeup about this project](http://yoavz.com/music_rnn/). This the code for 'Build an AI Composer' on [Youtube](https://youtu.be/S_f2qV2_U00)

Dependencies
============

* Numpy (http://www.numpy.org/)
* Tensorflow (https://github.com/tensorflow/tensorflow)
* Python Midi (https://github.com/vishnubob/python-midi.git)
* Mingus (https://github.com/bspaans/python-mingus)

Use [pip](https://pypi.python.org/pypi/pip) to install any missing dependencies

Installation (Tested on Ubuntu 16.04)
============

* Step 1: Install Tensorflow normally using the instructions at: https://www.tensorflow.org/install/

* Step 2: After installing Tensorflow, you will have to install the missing dependencies:

`pip install matplotlib`

`sudo apt-get install python-tk `

`pip install numpy`

* Step 3:

```
cd ~
git clone https://github.com/vishnubob/python-midi
cd python-midi
python setup.py install
```




```
cd ~
git clone https://github.com/bspaans/python-mingus
cd python-mingus
python setup.py install
```


Basic Usage
===========

1. `mkdir data && mkdir models`
2. run 'python main.py'. This will collect the data, create the chord mapping file in data/nottingham.pickle, and train the model
3. Run `python rnn_sample.py --config_file new_config_file.config` to generate a new MIDI song.

Give it 1-2 hours to train on your local machine, then generate the new song. You don't have to wait for it to finish, just wait until you see the 'saving model' message in terminal. In a future video, I'll talk about how to easily setup cloud GPU training. Likely using www.fomoro.com

Credits
===========
Credit for the vast majority of code here goes to [Yoav Zimmerman](https://github.com/yoavz). I've merely created a wrapper around all of the important functions to get people started.
