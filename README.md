Overview
============
A project that trains a LSTM recurrent neural network over a dataset of MIDI files. More information can be found on the [writeup about this project](http://yoavz.com/music_rnn/). 

Dependencies
============

* Numpy (http://www.numpy.org/)
* Tensorflow (https://github.com/tensorflow/tensorflow)
* Python Midi (https://github.com/vishnubob/python-midi.git)
* Mingus (https://github.com/bspaans/python-mingus)

Use [pip](https://pypi.python.org/pypi/pip) to install any missing dependencies

Basic Usage
===========

1. `mkdir data && mkdir models`
2. run 'python main.py'. This will collect the data, create the chord mapping file in data/nottingham.pickle, and train the model
3. Run `python rnn_sample.py --config_file new_config_file.config` to generate a new MIDI song.


Credits
===========
Credit for the vast majority of code here goes to [Yoav Zimmerman](https://github.com/yoavz). I've merely created a wrapper around all of the important functions to get people started.