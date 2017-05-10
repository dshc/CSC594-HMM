"""
David Chang
Student ID: 1487883
CSC 594 Homework #3
"""

import sys
import math
import numpy as np

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
    wordTagProbs[wordTag] = math.log(count / tagCounts[tag])

# Calculate probabilities of tag to tag transitions
tagTagProbs = {}
for tagTag, count in tagTagCounts.items():
    firstTag = tagTag[0]
    tagTagProbs[tagTag] = math.log(count / tagCounts[firstTag])

# Read in the full test file
testFilePath = "WSJ-test.txt" if len(sys.argv) == 1 else sys.argv[2]
#testFilePath = "test.txt" if len(sys.argv) == 1 else sys.argv[2]
TEST_FILE = open(testFilePath, 'r')
TEST_TEXT = TEST_FILE.read().strip()
TEST_FILE.close()
sentences = TEST_TEXT.split('\n\n')

# use these to keep track of overall accuracy
overallTotal = 0
overallCorrect = 0

# For each sentence, use the HMM we created from the probabilites
# and apply the viterbi algorithm to calculate the most likely
# states for each word in the sentence.
for sentence in sentences:
  lines = sentence.strip().split('\n')
  words = []
  tags = []
  for line in lines:
    splitLine = line.split()
    words.append(splitLine[0])
    tags.append(splitLine[1])
  x_length = len(words)
  y_length = len(tagCounts)

  # create viterbi matrix
  probMatrix = [[0 for x in range(x_length)] for y in range(y_length)]

  # matrix to keep track of the previous forward path probability
  maxPrevIndMatrix = [[0 for x in range(x_length)] for y in range(y_length)]

  # fill in the first column with probabilities
  # this tag array will contain all unique tags and will remain in a constant order
  tagArr = list(tagCounts.keys())
  for i in range(len(tagArr)):
    tag = tagArr[i]
    currTuple = (words[0], tag)

    # set the probability initiallly to negative infinity. This will be the 
    # default probability in case the transition probability or the state observation
    # likelihood does not appear in the training data
    prob = float("-inf")

    # the 'start' tag was included when populating probabilities
    if currTuple in wordTagProbs and ('start', tag) in tagTagProbs:
      # the transition for the first column will always include the 'start' tag as the
      # previous tag.
      prob = tagTagProbs[('start', tag)] + wordTagProbs[currTuple]

    # populate an item in the first column of the probability matrix
    probMatrix[i][0] = prob

  # the outermost loop covers each word in the sentence
  for wordInd in range(1, len(words)):
    # the second of the 3 level loop iterates through the tag array
    for currTagInd in range(len(tagArr)):
      currTup = (words[wordInd], tagArr[currTagInd])

      #use these variables to keep track of the optimal previous forward path
      maxProbKy = -1
      maxProbVal = None

      # finally, the innermost loop iterates through the tag array again,
      # this time, for the purpose of considering diffrent prior tags for the
      # transition probabilities
      for priorTagInd in range(len(tagArr)):
        tagTup = (tagArr[priorTagInd], tagArr[currTagInd])

        # like before, use a default of negative infinity in case the transition 
        # probability or the state observation likelihood does not appear in 
        # the training data
        prob = float("-inf")

        if currTup in wordTagProbs and tagTup in tagTagProbs:
          # add the transition probability with the state obserevation likelihood
          p = tagTagProbs[tagTup] + wordTagProbs[currTup]
          # also include the previous forward path probability
          prob = p + probMatrix[priorTagInd][wordInd-1]

        # keep track of the maximum previous forward path probability
        if (maxProbVal == None) or (math.exp(prob) > math.exp(maxProbVal)):
          maxProbVal = prob
          maxProbKy = priorTagInd

      maxPrevIndMatrix[currTagInd][wordInd] = maxProbKy
      probMatrix[currTagInd][wordInd] = maxProbVal

  # initialize a numpy array to easily calculate the index containing the 
  # maximum probability from the probability matrix
  a = np.array(probMatrix)
  maxIndLastCol = np.argmax(a[:,x_length-1])

  # this array will contain the tags for each word in the sentence
  t = [tagArr[maxIndLastCol]]

  # backtrack using the indices retrieved from each column to populate
  # the array of predicted tags
  for x in range(x_length-1, 0, -1):
    maxIndLastCol = maxPrevIndMatrix[maxIndLastCol][x]
    t.insert(0, tagArr[maxIndLastCol])

  # calculate the number of correct tag predictions 
  for y in range(len(t)):
    if t[y] == tags[y]:
      overallCorrect += 1
    overallTotal += 1

accuracy = overallCorrect / overallTotal
print(accuracy)




