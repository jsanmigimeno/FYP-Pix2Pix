import os, sys

class scrapper:
    def __init__(self, rootDir, setName, setType):
        self.rootDir = rootDir
        self.setName = setName
        self.setType = setType

    def extract(self):
        # Get paths to long and short sets
        longPath = os.path.join(self.rootDir, 'Long', self.setName)
        shortPath = os.path.join(self.rootDir, 'Short', self.setName)
        shortElems = os.listdir(shortPath)
        shortIds = [elem[1:5] for elem in shortElems]

        for elem in os.listdir(longPath):
            id = elem[1:5]
            matches = [(i) for i,x in enumerate(shortIds) if x==id]
            matchesNames = [shortElems[i] for i in matches]
            matches = self.filterMatches(matchesNames)
            yield elem, matches

    def filterMatches(self, matches):
        if self.setType == 'Best':
            expTimes = [elem[9:-2] for elem in matches]
            maxTime = max(expTimes)
            maxItems = [i for i, j in enumerate(expTimes) if j == maxTime]
            return [matches[i] for i in maxItems]
        elif self.setType == 'Worst':
            expTimes = [elem[9:-2] for elem in matches]
            minTime = min(expTimes)
            minItems = [i for i, j in enumerate(expTimes) if j == minTime]
            return [matches[i] for i in minItems]
        elif self.setType == 'All':
            return matches
        else:
            sys.exit(2)
