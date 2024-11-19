
import os
import wget

# Download the zipped dataset
url = 'https://archive.ics.uci.edu/static/public/53/iris.zip'
zip_name = "data.zip"
wget.download(url, zip_name)

# Unzip it and standardize the .csv filename
import zipfile
with zipfile.ZipFile(zip_name,"r") as zip_ref:
    zip_ref.filelist[1].filename = 'temp.csv'
    zip_ref.extract(zip_ref.filelist[1])

# Step 1: Read the existing contents of the file
with open("temp.csv", 'r') as file:
    contents = file.readlines()

# Step 2: Prepend the new line to the contents
contents.insert(0, "SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species" + '\n')  # Add a newline character

# Step 3: Write the combined content back to the file
with open("data_raw.csv", 'w') as file:
    file.writelines(contents)

os.remove(zip_name)
os.remove("temp.csv")