import numpy as np
import random
import re
# 使用贝叶斯算法实现垃圾邮件过滤
# 将一个大字符串解析为字符串列表
def textParse(bigString):
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())  # spam文件夹中的邮件全设为1
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())  # ham文件夹中的邮件全设为0
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # 将重复出现的单词删掉
    trainingSet = list(range(50))
    testSet = []
    # 随机选取20封邮件为测试集
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])  # 将测试集从训练集中删除
    trainMat = []
    trainClasses = []
    # 剩下的30封作为训练集
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))  # 将文本转换成向量
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))  # 贝叶斯算法来计算概率
    rightCount = 0
    # 测试集分类精度计算
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        print("the index %d is classified as: %d, the real class is %d" % (
        docIndex, classifyNB(np.array(wordVector), p0V, p1V, pSpam), classList[docIndex]))
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) == classList[docIndex]:
            rightCount += 1
    print('the accuracy rate is: ', float(rightCount) / len(testSet))



def createVocabList(dataSet):
    vocabSet = set([])  # 创建空集合
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 返回不重复的单词集合
        # print(vocabSet)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec



# trainMatrix为输入的词条集合,trainCategory为词条类别
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  # 获取词条长度,即分母变量
    numWords = len(trainMatrix[0])  # 第一段词条中单词个数,即分子变量
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive



# 分类,取概率高的值
# 1是垃圾邮件 0是非垃圾邮件
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    print('p0:', p0)
    print('p1:', p1)
    if p1 > p0:
        return 1
    else:
        return 0

if __name__ == '__main__':
    spamTest()