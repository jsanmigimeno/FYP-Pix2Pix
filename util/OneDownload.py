import requests

def download_from_onedrive(shareableLink, destination):
    r = requests.get(shareableLink)
    with open(destination, 'wb') as f:  
        f.write(r.content)