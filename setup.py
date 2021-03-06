import requests, zipfile, os 
import shutil

url = "https://github.com/suddu22/Deep-Learning-Assignments/archive/master.zip"
  
r = requests.get(url) 
  
file_name = "15CS10050_Assignment2.zip"
with open(file_name, 'wb') as f:  
    f.write(r.content) 

f.close()

zip_ref = zipfile.ZipFile(file_name, 'r')
zip_ref.extractall()
zip_ref.close()

src = 'Deep-Learning-Assignments-master'
dst = '15CS10050_Assignment2'

os.rename(src, dst)

src = '15CS10050_Assignment2/weights'
dst = 'weights'
shutil.copytree(src, dst)

shutil.rmtree('15CS10050_Assignment2')