# winogradsky
scripts and notebooks for analyzing images of Winogradsky columns

### Usage
`conda env create -f config.yml`

`transform.py` and `layout.py` contain helper functions for doing warp perspective transforms and generating image layouts with the transformed columns, respectively. It is assumed that the input images have some colored marking the edges of the region of interest on the column, and that this color is not represented elsewhere in the image (in order to perform HSV [hue saturation value] color-based masking). 

See the Jupyter notebooks `corner_alignment_EVR.ipynb` and `corner_alignment_WS.ipynb`as example use cases for column images taken in the environmental room and on the windowsill respectively. 
