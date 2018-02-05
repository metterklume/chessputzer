**Chessputzer** is designed to recognize positions from books and magazines. Use the [web interface](https://www.ocf.io/abhishek/putz). 

![Examples](boardexamples.png)

For best results, make sure that you include the border of the chessboard and crop close to it.If parts of the diagram are blurry or too hard to read, the algorithm makes the best guess based on the other pieces on the board. Please let me know if you have examples of books and images that don't work well.  

**Note**: the program is intended for fonts that usually appear in published works. Images created by chess software might not work. 

Have fun!

----

**Chessputzer** was inspired by [Fenbot](https://github.com/Elucidation/tensorflow_chessbot) but the internals are quite different. It is designed for scanned images with noise and artifacts.  So we do much less, but tolerate much more. 

**How it works**(*coming soon*)
