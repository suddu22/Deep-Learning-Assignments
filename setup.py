import requests, zipfile, os 
url = "https://github.com/suddu22/Deep-Learning-Assignments/archive/master.zip"
  
r = requests.get(url) 
  
file_name = "15CS10050_Assignment2.zip"
with open(file_name, 'wb') as f:  
    f.write(r.content) 

zip_ref = zipfile.ZipFile(file_name, 'r')
zip_ref.extractall()
zip_ref.close()

src = 'Deep-Learning-Assignments-master'
dst = '15CS10050_Assignment2'
os.rename(src, dst)