"""
David Chang
Student ID: 1487883
CSC 594 Homework #3
"""

import sys

tagCounts = {}
wordTagCounts = {}
tagTagCounts = {}


def addToTagCounts(tag):
    """ Add to tag counts dictionary """
    if tag in tagCounts:
        tagCounts[tag] = tagCounts[tag] + 1
    else:
        tagCounts[tag] = 1


def addToWordTagCounts(wordTag):
    """ Add to word tag counts dictionary """
    if wordTag in wordTagCounts:
        wordTagCounts[wordTag] = wordTagCounts[wordTag] + 1
    else:
        wordTagCounts[wordTag] = 1


def addToTagTagCounts(tagTag):
    """ Add to tag to tag transition count dictionary """
    if tagTag in tagTagCounts:
        tagTagCounts[tagTag] = tagTagCounts[tagTag] + 1
    else:
        tagTagCounts[tagTag] = 1


def parseTrainingText(filename):
    """
    Read in training file line by line and count how
    many times tags, words associated with tags, and tag
    transitions occur in the text
    """
    prevTag = ""
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line != "":
                tup = line.split()
                word = tup[0]
                tag = tup[1]
                wordTag = (word, tag)

                if prevTag == "":
                    # Start of sentence
                    addToTagCounts('start')
                    addToTagTagCounts(('start', tag))
                else:
                    addToTagTagCounts((prevTag, tag))

                addToTagCounts(tag)
                addToWordTagCounts(wordTag)
                prevTag = tag
            else:
                # End of sentence
                prevTag = ""


# Read and parse training text
filename = "WSJ-train.txt" if len(sys.argv) == 1 else sys.argv[1]
parseTrainingText(filename)

# Calculate probabilities of word given a tag
wordTagProbs = {}
for wordTag, count in wordTagCounts.items():
    tag = wordTag[1]
    wordTagProbs[wordTag] = count / tagCounts[tag]

# Calculate probabilities of tag to tag transitions
tagTagProbs = {}
for tagTag, count in tagTagCounts.items():
    firstTag = tagTag[0]
    tagTagProbs[tagTag] = count / tagCounts[firstTag]

# Read in the full test file
testFilePath = "test.txt" if len(sys.argv) == 1 else sys.argv[2]
TEST_FILE = open(testFilePath, 'r')
TEST_TEXT = TEST_FILE.read().strip()
TEST_FILE.close()
sentences = TEST_TEXT.split('\n\n')

# For each sentence, use the HMM we created from the probabilites
# and apply the viterbi algorithm to calculate the most likely
# states for each word in the sentence.
for sentence in sentences:
    lines = sentence.strip().split('\n')
    words = []
    for line in lines:
        words.append(line.split()[0])
    x_length = len(words)
    y_length = len(tagCounts)

    # create viterbi matrix
    probMatrix = [[0 for x in range(x_length)] for y in range(y_length)]

    tagArr = list(tagCounts.keys())
    for i in range(len(tagArr)):
        tag = tagArr[i]
        currTuple = (words[0], tag)
        prob = 0

        if currTuple in wordTagProbs and ('start', tag) in tagTagProbs:
            prob = tagTagProbs[('start', tag)] * wordTagProbs[currTuple]

        probMatrix[i][0] = prob

    print(probMatrix)
