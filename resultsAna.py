import os
import re

def getTrainLoss(file):
    with open(file, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        res = [0] * 1200
        for i, line in enumerate(lines):
            if 'Epoch[' in line:
                loc = re.search(r"Epoch\[[0-9]+\]", line).span()
                epoch = line[loc[0]+6: loc[1]-1]

                loc = re.search(r"Loss: [0-9]+\.[0-9]+", line).span()
                loss = line[loc[0]+6: loc[1]]
                # print(epoch, loss)
                
                res[int(epoch)-1] = float(loss)
    return res

def getResults(file):
    with open(file, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        for i, line in enumerate(lines):
            if 'all bn feat concat' in line:
                epoch = lines[i+1].split(':')[-1]
                mAP = lines[i+2].split(':')[-1]
                R1 = lines[i+3].split(':')[-1]
                R5 = lines[i+4].split(':')[-1]
                R10 = lines[i+5].split(':')[-1]
                print(epoch, mAP, R1, R5, R10)

if __name__ == "__main__":
    file = r"outputs_cmp\baseline3\log.txt"
    res1 = getTrainLoss(file)
    # print("+++++++++++"*10)
    file = r"outputs_cmp\ALNU\log.txt"
    res2 = getTrainLoss(file)

    file = r"outputs_cmp\ALNU_CdC\log.txt"
    res3 = getTrainLoss(file)
    
    import matplotlib.pyplot as plt
    # for i in range(100):
    #     plt.scatter(i, res1[i], color='r')
    #     print(res1[i])
    #     plt.scatter(i, res2[i], color='b')
    start, end = 0, 400
    plt.plot(res1[start: end], color='r')
    # print(res1[:100])
    plt.plot(res2[start: end], color='g')
    # plt.plot(res3[:end], color='b')
    # plt.yticks([15, 10, 5, 0])
    plt.show()