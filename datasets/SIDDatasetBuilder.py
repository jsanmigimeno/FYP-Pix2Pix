import os, sys, getopt, shutil
import scrapper
import combine_A_and_B as cAB

rootFolder = ''
setType = ''
deletePrevDir = False
buildFinal = False
onlyCombine = False

try:
    opts, args = getopt.getopt(sys.argv[1:], "p:t:f", ["path=","type=","build-final","only-combine"])
except getopt.GetoptError:
    print("Error when parsing arguments")
    sys.exit(2)

for opt, arg in opts:
    if opt in ("-p", "--path"):
        rootFolder = arg
        print("Dataset main folder: ", rootFolder)
    elif opt in ("-t", "--type"):
        if not arg in ['Best', 'All', 'Worst']:
            print("Output dataset type '" + arg + "' not recognised.")
            sys.exit(2)
        setType = arg
        print("Set to be generated: ", setType)
    elif opt == "-f":
        deletePrevDir = True
        print("Warning: Existing output directories will be deleted.")
    elif opt == "--build-final":
        buildFinal = True
        print("Dataset will be combined to form the final dataset.")
    elif opt == "--only-combine":
        onlyCombine = True
        print("Skipping split, assuming already done")

# In each of the following 3 folders are expected, with Test, Train and Validation sets
# 'Long' and 'Short' directories are expected
rootList = os.listdir(rootFolder)

# Check directory consistency
if not all(elem in rootList for elem in ['Long', 'Short']):
    print("Error: Either 'Long' or 'Short' folders were not found.")
    sys.exit(2)
for element in ['Long', 'Short']:
    subpath = os.path.join(rootFolder, element)
    if not all(elem in os.listdir(subpath) for elem in ['train', 'val', 'test']):
        print("Error: Mising set in '" + element + "' folder")
        sys.exit(2)


# Create output directory
outputDir = os.path.join(rootFolder, setType)
ADir = os.path.join(outputDir, 'A')
BDir = os.path.join(outputDir, 'B')

if not onlyCombine:
    if os.path.exists(outputDir) and os.path.isdir(outputDir):
        if not deletePrevDir:
            count = 0
            delete = False
            while not delete:
                ans = input("Warning: Output directory already exists. All contents will be erased. Continue? [yes/no]: ")
                if ans == 'yes':
                    delete = True
                elif ans == 'no':
                    sys.exit(0)
                count += 1
                if count > 2 and not delete:
                    sys.exit(0)
        shutil.rmtree(outputDir)

    os.makedirs(outputDir)
    os.makedirs(ADir)
    os.makedirs(BDir)

    for subset in ['train', 'test', 'val']:
        print("Starting '" + subset + "' set.")
        LongOrgDir = os.path.join(rootFolder, 'Long', subset)
        ShortOrgDir = os.path.join(rootFolder, 'Short', subset)
        ADirTemp = os.path.join(ADir, subset)
        BDirTemp = os.path.join(BDir, subset)
        os.makedirs(ADirTemp)
        os.makedirs(BDirTemp)

        nFiles = len(os.listdir(LongOrgDir))
        counter = 0

        sc = scrapper.scrapper(rootFolder, subset, setType)
        for longEl, shortEl in sc.extract():
            longElOrgPath = os.path.join(LongOrgDir, longEl)
            for elem in shortEl:
                longElDestPath = os.path.join(BDirTemp, elem)
                shutil.copyfile(longElOrgPath, longElDestPath)

                shortElOrgPath = os.path.join(ShortOrgDir, elem)
                shortElDesPath = os.path.join(ADirTemp, elem)
                shutil.copyfile(shortElOrgPath, shortElDesPath)

            counter += 1
            if counter % 10 == 0 and not counter == 0:
                print(str(counter) + " of " + str(nFiles) + " done.")
        print("'" + subset + "' done.")    

    print("Finished making sets correctly.")

if buildFinal or onlyCombine:
    combinedDir = os.path.join(outputDir, 'Combined')
    os.makedirs(combinedDir)
    cAB.combine(ADir, BDir, combinedDir)
    shutil.make_archive(combinedDir, 'zip', combinedDir)