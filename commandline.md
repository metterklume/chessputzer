### Using Chessputzer on the command line

**Requirements**

Python + the following packages: numpy, opencv, scipy, pillow

**Usage**

You need the files: `putzmain.py`, `putzlib.py` and `pbarrs.npz` in a single directory.

For a single chessboard image:

python -m putzmain -f `imagefile` -o `output txt file`

For a directory of images:

python -m putzmain -d `imagedirectory` -o `output txt file`

If no output file is specified, output is written to fens.txt (overwritten if it exists!). In the first case output is a single line with the FEN position. In the second, it is written as line1: filename line 2: FEN... for each of the file(s) in the directory.

*Notes*: Opencv is (finally!) easy to install via conda or pip. The functions needed from scipy and pillow (Python Imaging library) have equivalents in opencv, so folding them in is on the todo list.

