
# A file of the test setences, plus some other info about them
test_sents_file = open("test_set.txt", "r")

# A file of the supertags assigned to each word in each test sentence
testTags = open("test_tags.txt", "r")

# Create a dictionary that, for each supertag in the MICA
# database, returns the tree properties of that supertag
treeProps = open("d6.treeproperties", "r")
treeDict = {}
for line in treeProps:
	parts = line.split()
	treeDict[parts[0]] = parts[1:]

# Creates a list of all the test sentences, plus the other info about
# each sentence
test_sents = test_sents_file.readlines()

# Creates a list of the supertag lists for all test sentences
test_tags = testTags.readlines()

# Counters that count different PP attachment results. Each count has
# a 2-letter identifier at its end. The first of those 2 letters is 
# the correct answer, in terms of whether the PP should attach to a
# Noun (N) or a Verb (V). The second letter is what the supertagger identified the
# PP as attaching to--either a Noun (N), Verb (V), or Other (O)
countNN = 0
countVV = 0
countNV = 0
countVN = 0
countNO = 0
countVO = 0

# Now, for each test sentence, we figure out what the correct 
# PP attachment answer was, and what answer the supertagger identified
for index, sent in enumerate(test_sents):
    # Identifying the test sentence, the noun before the PP, the 
    # preposition starting the PP, and the correct POS modified
    # by the PP
    parts = sent.split("\t")
    sentence = parts[-1]
    noun = parts[1]
    prep = parts[2]
    desiredPOS = parts[0]
    
    # Identifies the supertag that the supertagger assigned to
    # the preposition
    words = sentence.split()
    indNoun = words.index(noun)
    later = words[indNoun + 1:]
    indPrep = later.index(prep) + indNoun + 1
    tag = test_tags[index].split()[indPrep]

    # Uses the tree properties to determine what the PP is
    # modifying, as judged by the supertagger
    if tag != "tCO":
        actualPOS = treeDict[tag][2]
    else:
        actualPOS = "modif:VP"

    # Consolidating related parts of speech
    if actualPOS == "modif:V":
        actualPOS = "modif:VP"
    if actualPOS == "modif:PRN" or actualPOS == "modif:QP" or actualPOS == "modif:AP":
        actualPOS = "modif:NP"

    # Incrementing the applicable counter
    if actualPOS == "modif:VP" and desiredPOS == "V":
        countVV += 1
    elif actualPOS == "modif:NP" and desiredPOS == "N":
        countNN += 1
    elif actualPOS == "modif:VP" and desiredPOS == "N":
        countNV += 1
    elif actualPOS == "modif:NP" and desiredPOS == "V":
        countVN += 1
    elif desiredPOS == "N":
        countNO += 1
        #print actualPOS, sentence, tag
    elif desiredPOS == "V":
        countVO += 1
        #print actualPOS, sentence, tag
    else:
        print "error"

# Printing the results
print "countVV:", countVV
print "countNN:", countNN
print "countNV:", countNV
print "countVN:", countVN
print "countNO:", countNO
print "countVO:", countVO

# Print the number correct, followed by the number incorrect
print "Number correctly tagged:", countVV + countNN
print "Number incorrectly tagged:", countNV + countVN + countNO + countVO

# Print the accuracy
print "Accuracy:", (countVV + countNN) * 1.0 / (countVV + countNN + countNV + countVN + countNO + countVO)


