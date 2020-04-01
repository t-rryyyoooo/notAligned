import argparse
import SimpleITK as sitk

args = None

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("imagePath", help="$HOME/Desktop/data/kits19/case_00000")
    parser.add_argument("savePath", help="$HOME/Desktop/data/kits19/case_00000")
    parser.add_argument("--spacing", default=[1.0, 1.0, 1.0], type=float, nargs=3)
    parser.add_argument("--prefix", default="resampled_")

    args = parser.parse_args()
    return args


def changeSpacing(img, spacing, is_label=False):
    # original shape
    inputShape = img.GetSize()
    inputSpacing = img.GetSpacing()
    newShape = [ int(1 + ish * isp / osp) for ish, isp, osp in zip(inputShape, inputSpacing, spacing)]
    print("Change spacing from {} to {}.".format(inputSpacing, spacing))
    print("So, Change shape from {} to {}.".format(inputShape, newShape))
    
    if img.GetNumberOfComponentsPerPixel() == 1:
        minmax = sitk.MinimumMaximumImageFilter()
        minmax.Execute(img)
        minval = minmax.GetMinimum()
    else:
        minval = None

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(newShape)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputSpacing(spacing)
    
    
    if minval is not None:
        resampler.SetDefaultPixelValue(minval)
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
   
    resampled = resampler.Execute(img)
    
    return resampled

def main(args):
    imagePath = args.imagePath + "/imaging.nii.gz"
    labelPath = args.imagePath + "/segmentation.nii.gz"

    image = sitk.ReadImage(imagePath)
    label = sitk.ReadImage(labelPath)
    resampledImage = changeSpacing(image, [*args.spacing])
    resampledLabel = changeSpacing(label, [*args.spacing])

    saveImagePath = args.savePath + "/" + args.prefix + "imaging.nii.gz"
    saveLabelPath = args.savePath + "/" + args.prefix + "segmentation.nii.gz"

    print("saving Image to {}...".format(saveImagePath))
    sitk.WriteImage(resampledImage, saveImagePath)
    print("Done.")
    print("saving Image to {}...".format(saveImagePath))
    sitk.WriteImage(resampledLabel, saveLabelPath)
    print("Done.")

if __name__=="__main__":
    args = parseArgs()
    main(args)
