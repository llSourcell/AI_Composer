Overview
============
A project that trains recurrent neural networks over datasets of MIDI files. More information can be found on the [writeup about this project](http://yoavz.com/music_rnn/) and on the [paper](http://yoavz.com/music_rnn_paper.pdf) written to accompany this project. 

Dependencies
============

* Numpy (http://www.numpy.org/)
* Tensorflow (https://github.com/tensorflow/tensorflow)
* Python Midi (https://github.com/vishnubob/python-midi.git)
* Mingus (https://github.com/bspaans/python-mingus)

Basic Usage
===========

1. `mkdir data && mkdir models`
2. Download the dataset [Nottingham MIDI dataset](http://www-etud.iro.umontreal.ca/~boulanni/Nottingham.zip) and unzip to `data/Nottingham`
3. Run `python nottingham_util.py` to generate the sequences and chord mapping file to `data/nottingham.pickle`
4. Run `python rnn.py --run_name YOUR_RUN_NAME_HERE` to start training the model 
