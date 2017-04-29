"""
David Chang
Student ID: 1487883
CSC 594 Homework #3
"""

import sys

tagCounts = {}
wordTagCounts = {}
tagTagCounts = {}

wordTagProbs = {}
tagTagProbs = {}


def addToTagCounts(tag):
    if tag in tagCounts:
        tagCounts[tag] = tagCounts[tag] + 1
    else:
        tagCounts[tag] = 1


def addToWordTagCounts(wordTag):
    if wordTag in wordTagCounts:
        wordTagCounts[wordTag] = wordTagCounts[wordTag] + 1
    else:
        wordTagCounts[wordTag] = 1


def addToTagTagCounts(tagTag):
    if tagTag in tagTagCounts:
        tagTagCounts[tagTag] = tagTagCounts[tagTag] + 1
    else:
        tagTagCounts[tagTag] = 1


# Get probabilities of word given a tag
prevTag = ""
filename = "WSJ-train.txt"
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
            prevTag = ""

# Calculate probs of word given a tag
for wordTag, count in wordTagCounts.items():
    tag = wordTag[1]
    wordTagProbs[wordTag] = count / tagCounts[tag]

# Calculate probs of tag to tag transitions
for tagTag, count in tagTagCounts.items():
    firstTag = tagTag[0]
    tagTagProbs[tagTag] = count / tagCounts[firstTag]

TEST_FILE = open(sys.argv[2], 'r')
TEST_TEXT = TEST_FILE.read().strip()
TEST_FILE.close()

sentences = TEST_TEXT.split('\n\n')
for sentence in sentences:
    lines = sentence.strip().split('\n')
    words = []
    for line in lines:
        words.append(line.split()[0])
    x_length = len(words)
    y_length = len(tagCounts)
    probMatrix = [[0 for x in range(x_length)] for y in range(y_length)]

    index = 0
    for tag, count in tagCounts.items():
        currTuple = (words[0], tag)
        prob = 0

        if currTuple in wordTagProbs and ('start', tag) in tagTagProbs:
            prob = tagTagProbs[('start', tag)] * wordTagProbs[currTuple]

        probMatrix[index][0] = prob
        index = index + 1

    print(probMatrix)
