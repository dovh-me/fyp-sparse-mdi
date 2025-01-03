import gdown

def download_drive_file(self, file_id, destination):
        url = 'https://drive.google.com/uc?id=' + file_id
        output = destination 
        gdown.download(url, output, quiet=False)