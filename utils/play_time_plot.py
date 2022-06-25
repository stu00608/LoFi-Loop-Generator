import os
from pretty_midi import PrettyMIDI
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

jsfakePath = '../js-fakes'
trainPath = os.path.join(jsfakePath, 'midi')
testPath = os.path.join(jsfakePath, 'jsf-extended')

if __name__ == '__main__':

    totalPlayTimeList = []
    for mid in tqdm(glob(os.path.join(trainPath, '*.mid'))):
        m = PrettyMIDI(mid)
        endTime = m.get_end_time()
        totalPlayTimeList.append(endTime)

    f, ax1 = plt.subplots()

    ax1.plot(list(range(1, len(os.listdir(trainPath))+1)), totalPlayTimeList)
    plt.show()
