### Using Chessputzer (deep net version) on the command line

**Requirements**

Python + the following packages: numpy, opencv, scipy, pillow, tensorflow and keras.

We don't need the gpu version for tensorflow. The cpu is fast enough. 

**Usage**

Put the files, `putzmain.py`, `putzlib.py` and `putznn.py` in a single directory.

Make a directory `./saved_models` and put both files from [here](https://www.dropbox.com/sh/i7d4xor7iclb9mh/AAAaCKKln671uwxa12eDPbzQa?dl=0). 

For a single chessboard image:

python -m putznn -f `imagefile` -o `output txt file`

For a directory of images:

python -m putznn -d `imagedirectory` -o `output txt file`

If no output file is specified, output is written to fens.txt (overwritten if it exists!). In the first case output is a single line with the FEN position. In the second, it is written as line1: filename line 2: FEN... for each of the file(s) in the directory.

*Note*: You can change the `path_to_models` setting in `putznn.py`
