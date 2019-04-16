import urllib.request
import requests
import os, sys, getopt, shutil
import zipfile

def getDataset(id):
    if id == 'All_JPEG9':
        return 'https://imperiallondon-my.sharepoint.com/:u:/g/personal/js5415_ic_ac_uk/ET6bdVXRESBGmP65Pag-Ys4BjiiVU8sSDR1l94CAi8xtLw?download=1'
    elif id == 'Best_JPEG9':
        return 'https://imperiallondon-my.sharepoint.com/:u:/g/personal/js5415_ic_ac_uk/EXr7idor-MVKjkp00BK77GoBH-JwVzTUu1QFT8tJdPNtiQ?download=1'
    elif id == 'Worst_JPEG9':
        return ''
    elif id == 'All_PNG':
        return 'https://imperiallondon-my.sharepoint.com/:u:/g/personal/js5415_ic_ac_uk/EaeBvQ97WaJGn2Iis9DwwY4ByAd1yOUViQovn4AXov74zw?download=1'
    elif id == 'Best_PNG':
        return 'https://imperiallondon-my.sharepoint.com/:u:/g/personal/js5415_ic_ac_uk/ETT7TE4LFApAqazKsWGsGgUBVyHVo6GEXFL8Gby_YA42bQ?download=1'
    elif id == 'Worst_PNG':
        return ''
    elif id == 'Downscaled_All_JPEG':
        return 'https://imperiallondon-my.sharepoint.com/:u:/g/personal/js5415_ic_ac_uk/EaUEnsEpMKpOij7fc-8cJgsBzKwXdpufOJsTkT2l02EfuA?download=1'
	elif id == 'baselineScaled':
		return 'https://imperiallondon-my.sharepoint.com/:u:/g/personal/js5415_ic_ac_uk/EQCOAp0ZMGVPghOYwa_nHdMB3Rk0FPrz1MMtVm2V2G9iQw?download=1'
    else:
        return ''

rootFolder = ''
dataset = ''
deletePrevDir = False

try:
    opts, args = getopt.getopt(sys.argv[1:], "p:d:f", ["path="])
except getopt.GetoptError:
    print("Error when parsing arguments")
    sys.exit(2)

for opt, arg in opts:
    if opt in ("-p", "--path"):
        rootFolder = arg
        print("Dataset main folder: ", rootFolder)
    elif opt in ("-d"):
        dataset = arg
        print("Dataset to download: ", dataset)
    elif opt == "-f":
        deletePrevDir = True
        print("Warning: Existing output directories will be deleted.")

downloadDir = os.path.join(rootFolder, dataset)
if os.path.exists(downloadDir) and os.path.isdir(downloadDir):
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
    shutil.rmtree(downloadDir)

os.makedirs(downloadDir)
downloadPath = os.path.join(downloadDir, dataset + '.zip')

print("Downloading file.")
url = getDataset(dataset)
r = requests.get(url)
with open(downloadPath, 'wb') as f:  
    f.write(r.content)

print("Unzipping file.")
with zipfile.ZipFile(downloadPath, 'r') as zip_ref:
    zip_ref.extractall(downloadDir)

print("Removing zip file.")
os.remove(downloadPath)

print("Done.")