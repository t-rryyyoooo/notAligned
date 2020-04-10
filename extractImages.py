import os
import shutil
import argparse

#testing =  ['173', '002', '068', '133', '155', '114', '090', '105', '112', '175', '183', '208', '029', '065', '157', '162', '141', '062', '031', '156', '189', '135', '020', '077', '000', '009', '198', '036']

numbers = ['019', '023', '054', '093', '096', '123', '127', '136', '141', '153', '188', '191', '201', '001', '017', '020', '022', '043', '082', '094', '115', '120', '137', '173', '174', '205']
args = None

def parseArags():
    parser = argparse.ArgumentParser()
    parser.add_argument("resultsPath", nargs="*", help="~/Desktop/data/hist/segmentation ~/Desktop/data/patch/original")
    parser.add_argument("-o", "--originalPath", default="~/Desktop/data/kits19", help="~/Desktop/data/kits19")
    parser.add_argument("-s", "--savePath", default="~/Desktop/data/watch", help="~/Desktop/data/watch")
    args = parser.parse_args()

    return args

def main(args):
    originalPath = os.path.expanduser(args.originalPath)
    savePath = os.path.expanduser(args.savePath)

    length = len(args.resultsPath)
    for i in range(length):
        args.resultsPath[i] = os.path.expanduser(args.resultsPath[i])

    
    shutil.rmtree(savePath)
    for x in numbers:
        ctPath = originalPath + "/case_00" + x + "/imaging.nii.gz"
        labelPath = originalPath + "/case_00" + x + "/segmentation.nii.gz"

        savexPath = savePath + "/case_00" + x


        if not os.path.exists(savexPath):
            print("Make ", savexPath)
            os.makedirs(savexPath, exist_ok = True)

        shutil.copy(ctPath, savexPath)
        shutil.copy(labelPath, savexPath)

        for i, resultPath in enumerate(args.resultsPath):
            resultxPath = resultPath + "/case_00" + x + "/label.mha"
            saveResultPath = savexPath + "/label_" + str(i + 1) + ".mha"

            shutil.copy(resultxPath, saveResultPath)

        print("Successfully extracting images in case_00" + x)

if __name__ == "__main__":
    args = parseArags()
    main(args)

    
