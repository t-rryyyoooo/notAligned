import numpy as np
import SimpleITK as sitk
import os
import argparse
from functions import DICE, caluculateAVG

args = None

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('trueLabel', help = '~/Desktop/data/KIDNEY')
    parser.add_argument('resultLabel',help = '~/Desktop/data/hist/segmentation')

    args = parser.parse_args()

    return args


def main(args):
    #testing =  ['001', '017', '020', '022', '043', '082', '094', '115', '120', '137', '173', '174', '205']    
    #testing =  ['019', '023', '054', '093', '096', '123', '127', '136', '141', '153', '188', '191', '201']
    #testing = ['173', '002', '068', '133', '155', '114', '090', '105', '112', '175', '183', '208', '029', '065', '157', '162', '141', '062', '031', '156', '189', '135', '020', '077', '000', '009', '198', '036']
    #testing = ['001', '017', '020', '022', '043', '082', '094', '115', '120', '137', '173', '174', '205','019', '023', '054', '093', '096', '123', '127', '136', '141', '153', '188', '191', '201']
    #testing = ['000', '003', '004', '006', '007', '009', '010', '014', '015', '027', '031', '032', '033', '037', '039', '040', '049', '050', '052', '062', '063', '064', '065', '072', '077', '083']
    testing = ['071', '097', '100', '106', '121', '124', '125', '128', '129', '132', '138', '140', '146', '150', '152', '155', '156', '158', '164', '167', '175', '182', '190', '193', '198', '203']
    wholeDICE=[]
    kidneyDICE = []
    cancerDICE = []
    for x in testing:

        trueLabel = os.path.expanduser(args.trueLabel) + '/case_00' + x + '/segmentation.nii.gz'
        resultLabel = os.path.expanduser(args.resultLabel) + '/case_00' + x + '/label.mha'

        true = sitk.ReadImage(trueLabel)
        result = sitk.ReadImage(resultLabel)

        trueArray = sitk.GetArrayFromImage(true)
        resultArray = sitk.GetArrayFromImage(result)
     
        wholeDICE.append(DICE(trueArray,resultArray))

        trueKid = np.where(trueArray == 1, 1, 0)
        trueCan = np.where(trueArray == 2, 2, 0)

        resultKid = np.where(resultArray == 1, 1, 0)
        resultCan = np.where(resultArray == 2, 2, 0)


        kidneyDICE.append(DICE(trueKid,resultKid))
        cancerDICE.append(DICE(trueCan,resultCan))
        print('case_00' + x)
        print("Average whole: {}  ".format(DICE(trueArray,resultArray)))
        print("Average kidney: {}  ".format(DICE(trueKid,resultKid)))
        print("Average cancer: {}  ".format(DICE(trueCan,resultCan)))

    print("Average whole: {}  ".format(caluculateAVG(wholeDICE)))
    print("Average kidney: {}  ".format(caluculateAVG(kidneyDICE)))
    print("Average cancer: {}  ".format(caluculateAVG(cancerDICE)))
    print()

if __name__ == '__main__':
    args = parseArgs()
    main(args)
