<span style="font-size:500%"> **README** </span>

<a name="contents"></a>
# Flow chart  
---
input -> program -> output  
CT image and label -> [sliceImage.py](#slice) -> 3D slice images, 2D slice labels and text file  
text file -> [merge.py](#merge) -> text files for training, validation and test  
text file for training and validation -> [buildUnet.py](#UNet) -> file with model and weight and log.  
CT image, label and file with model and weight -> [segmentation.py](#segmentation) -> segmented image   


# File list  
---
>## .sh  
>- [auto2DUnet.sh](#UNet)
>- [autoCaluculateDICE.sh](#caluculateDICE)
>- [autoSegmentation.sh](#segmentation)
>- [autoSliceImage.sh](#slice)
>## .ipynb
>- [plot.ipynb](#ipynb)
>- [processing.ipynb](#ipynb)
>- [memo.ipynb](#ipynb)
>## .csv
>- [dfAboutKidneyAndCnacer.csv](#csv)
>## .py
>- [buildUnet.py](#UNet)
>- [caluculateDICE.py](#caluculateDICE)
>- [caluculateDICEOneByOne.py](#caluculateDICE)
>- [changeSpacing.py](#changeSpacing)
>- [cut.py](#slice)
>- [extractImages.py](#extractImages)
>- [functions.py](#functions)
>- [loader.py](#UNet)
>- [merge.py](#merge)
>- [mergeOneByOne.py](#merge)
>- [saver.py](#UNet)
>- [segmentationUnetAligned.py](#segmentation)
>- [segmentationUnetInputAs3ch.py](#segmentation)
>- [sliceImage.py](#slice)
>## directory
>- [result](#directory)
>- [test](#directory)
>- [fail](#directory)

<a name="slice"></a>
## Make slice images and align the center of gravity of the kidney region in the slice to the center of image and save 3D images and 2D labels.
We extract kidney region in the CT image and slice it in perpendicular axial direction and align the center of gravity of the kidney region in the slice to the center of image. We transfrom image size into outputSize and save images as 3D, labels as 2D and text file which includes image paths and label paths.  

`sliceImage.py` does that for one patient (ct image).  
`autoSliceImage.sh` runs `sliceImage.py` for patients you want.  
`cut.py` has functions to slice images.  

<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>

<a name="merge"></a>
## Merge textfiles.
When we runs `sliceImage.py`, it creates text file which written paths slice images are saved to. To feed these images to U-Net, we enter a text file in U-Net program. But, because `sliceImage.py` creates text file per patient, we can't feed all of images you want to, even if you enter it in U-Net program. So, we read text files per patient and write all of patient's image paths you want to feed to U-Net into the text file. And then, feed it to U-Net program.  
<br>
`merge.py` reads text files and writes all of patient's image paths you want to feed to U-Net into save text file.  
It determines which files are read in this program such as "~/Desktop/data/slice/path/case_00000, ~/Desktop/data/slice/path/case_00001".  
On the other hand, `mergeOneByOne.py` determines read files when we run it.  

<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>

<a name="UNet"></a>
## Run Unet.
`buildUnetAugmentation.py` does machine learning with U-Net.  
`loader.py` has generators.  
`saver.py` has classes to save weight.  
`auto2DUnet.sh` is written a part of file paths to input, to save weight and so on.  
<br>
When we run `auto2DUnet.sh`, you are requied to type GPU ID and suffix. we can select GPU by GPU ID, and suffix is used to determine paths to input, save weight and so on. And, you are asked "Is the weight file's suffix the same as above?[yes/no]". If you choose yes, above suffix is used when saving weight and so on, if you choose no, you are required to feed suffix which is used when saving weight and so on.  

<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>

<a name="segmentation"></a>
## Segment kidneys and kidney cancers from CT image.
We feed CT image to `segmentation*.py` and it preprocesses input image. For preprocessed images, it segments kidneys and kidney cancers. Then, it restores the original size by performing the reverse of the preprocessing.So, We remake `segmentation*.py` when we develop new preprocessing.  
<br>
`segmentationUnetAligned.py` segments images aligned the center of gravity of the kidney for one patient.  
`segmentationUnetNotAligned.py` segments images not aligned the center of gravity of the kidney for one patient.  
`segmentaionUnetInputAs3ch.py` segments aligned images saved as 3ch for one patient.  
`autoSegmentation.sh` runs `segmentation*.py` for patients you want.  

<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>


<a name="caluculateDICE"></a>
## Caluculate DICE. 
`caluculateDICE.py` caluculates DICE score for two labels, for example, label U-Net predicts and true label.   
`caluculateDICE.py` determine which patients are caluculated, such as ~/Desktop/data/kits19/case_00000, ~/Desktop/data/kits19/case_00001.  
On the other hand, `caluculateDICEOneByOne.py` determines labels which are caluculated DICE, when it is run.  
`autoCaluculateDICE.sh` runs `caluculateDICE.py` and write its outpus into text file.    

<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>

<a name="changeSpacing"></a>
## Change spacing CT image has.
CT image has spacing data which is the length between each boxel. physical size is caluculated from Spacing and matrix size.  
`ChangeSpacing.py` transforms CT image matrix size and change spacing with it so that transformed physical size is the same size as original one.  

<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>

<a name="extractImages"></a>
## Extract images you want.
`extractImages.py` extracts images you want from directories so that you can easily watch image and labels by 3Dslicer.  

<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>

<a name="functions"></a>
## functions.py
`functions.py` has functions used in a various of programs.  

<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>


<a name="ipynb"></a>
## The information of .ipynb.
`memo.ipynb` is like a notepad on programming.  
`plot.ipynb` is used when we plot something.  
`preprocessing.ipynb` is used when we want to preprocess.  
These files are divided by usage, but, in fact, `memo.ipynb` and `plot.ipynb` are used similar purposes. And `preprocessing.ipynb` has not been used recently.  

<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>


<a name="csv"></a>
## About csv
`dfAboutKidneyAndCancer.csv` has patient's CT image's statistics per slice image.

<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>


<a name="directory"></a>
## The information of directories.
`result` has text files written the results of machine learning.  
`test` is output destination when we test program. It is not sent to github.
`fail` has text file written the patients execution failed.  

<br>
<div style="text-align: right;"><a href="#contents">To contents</a></div>



