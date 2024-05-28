import os
import cv2
import pickle
import numpy as np
import face_recognition
import streamlit as st
from pathlib import Path
import zipfile
import shutil
import tempfile

def save_to_temp(file_content, file_name):
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file_name)
    with open(file_path, "wb") as f:
        f.write(file_content)
    return temp_dir, file_path

def clean_up_temp(temp_dir):
    shutil.rmtree(temp_dir)

def clean_up():
    folders_to_delete = ["Dataset", "People", "output"]
    files_to_delete = ["known_encodings.pickle", "output.zip"]
    
    for folder in folders_to_delete:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    
    for file in files_to_delete:
        if os.path.exists(file):
            os.remove(file)

def reset_opencv():
    cv2.destroyAllWindows()
    cv2.waitKey(1)

# Save encodings
def saveEncodings(encs, names, fname="encodings.pickle"):
    data = [{"name": nm, "encoding": enc} for (nm, enc) in zip(names, encs)]
    encodingsFile = fname
    
    # Dump the facial encodings data to disk
    print("[INFO] serializing encodings...")
    with open(encodingsFile, "wb") as f:
        f.write(pickle.dumps(data))

# Function to read encodings
def readEncodingsPickle(fname):
    with open(fname, "rb") as f:
        data = pickle.loads(f.read())
    encodings = [d["encoding"] for d in data]
    names = [d["name"] for d in data]
    return encodings, names

# Function to create encodings and get face locations
def createEncodings(image):
    face_locations = face_recognition.face_locations(image)
    known_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
    return known_encodings, face_locations

# Function to compare encodings
def compareFaceEncodings(unknown_encoding, known_encodings, known_names):
    matches = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance=0.5)
    face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)
    best_match_index = np.argmin(face_distances)
    distance = face_distances[best_match_index]
    
    if matches[best_match_index]:
        acceptBool = True
        duplicateName = known_names[best_match_index]
    else:
        acceptBool = False
        duplicateName = ""
    
    return acceptBool, duplicateName, distance

# Save Image to new directory
def saveImageToDirectory(image, name, imageName):
    path = f"./output/{name}"
    os.makedirs(path, exist_ok=True)
    cv2.imwrite(f"{path}/{imageName}", image)

# Function for creating encodings for known people
def processKnownPeopleImages(path="./People/", saveLocation="./known_encodings.pickle"):
    known_encodings = []
    known_names = []

    for root, dirs, files in os.walk(path):
        for img in files:
            imgPath = os.path.join(root, img)
            
            # Read image
            image = cv2.imread(imgPath)
            if image is None:
                print(f"[WARN] Skipping file {imgPath}, not a valid image")
                continue
            
            name = os.path.splitext(img)[0]
            
            # Resize
            image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
            
            # Get locations and encodings
            encs, locs = createEncodings(image)
            
            if encs:
                known_encodings.append(encs[0])
                known_names.append(name)
                print(f"[INFO] Processed {imgPath}, found {len(encs)} face(s)")
            else:
                print(f"[WARN] No faces found in {imgPath}")
            
            for loc in locs:
                top, right, bottom, left = loc
                
                # Show Image with rectangle
                cv2.rectangle(image, (left, top), (right, bottom), color=(255, 0, 0), thickness=2)
                cv2.imshow("Image", image)
                cv2.waitKey(1)
                cv2.destroyAllWindows()
    
    if known_encodings:
        saveEncodings(known_encodings, known_names, saveLocation)
        print(f"[INFO] Saved encodings to {saveLocation}")
    else:
        print("[ERROR] No encodings were saved")

# Function for processing dataset images
def processDatasetImages(path="./Dataset/", saveLocation="./dataset_encodings.pickle"):
    if not os.path.exists("./known_encodings.pickle"):
        raise FileNotFoundError("[ERROR] known_encodings.pickle not found. Make sure to process known people images first.")
    
    people_encodings, names = readEncodingsPickle("./known_encodings.pickle")
    
    for root, dirs, files in os.walk(path):
        for img in files:
            imgPath = os.path.join(root, img)
            
            # Read image
            image = cv2.imread(imgPath)
            if image is None:
                print(f"[WARN] Skipping file {imgPath}, not a valid image")
                continue
            
            orig = image.copy()
            
            # Resize
            image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
            
            # Get locations and encodings
            encs, locs = createEncodings(image)
            
            # Save image to a group image folder if more than one face is in image
            if len(locs) > 1:
                saveImageToDirectory(orig, "Group", img)
            
            # Processing image for each face
            knownFlag = 0
            for i, loc in enumerate(locs):
                top, right, bottom, left = loc
                unknown_encoding = encs[i]
                acceptBool, duplicateName, distance = compareFaceEncodings(unknown_encoding, people_encodings, names)
                
                if acceptBool:
                    saveImageToDirectory(orig, duplicateName, img)
                    knownFlag = 1
            
            if knownFlag:
                print("Match Found")
            else:
                saveImageToDirectory(orig, "Unknown", img)
            
            # Show Image
            if locs:
                cv2.rectangle(image, (left, top), (right, bottom), color=(255, 0, 0), thickness=2)
                cv2.imshow("Image", image)
                cv2.waitKey(1)
                cv2.destroyAllWindows()

def zip_output_folder(folder_path, zip_path):
    shutil.make_archive(zip_path, 'zip', folder_path)
    print(f"[INFO] Zipped output folder to {zip_path}.zip")

def main():
    st.title("Face Recognition System")
    
    # Upload folder for People
    people_folder = st.file_uploader("Upload a ZIP file containing images of known people", type="zip")
    if people_folder is not None:
        people_temp_dir, people_zip_path = save_to_temp(people_folder.read(), "people.zip")
    
    # Upload folder for Dataset
    dataset_folder = st.file_uploader("Upload a ZIP file containing dataset images", type="zip")
    if dataset_folder is not None:
        dataset_temp_dir, dataset_zip_path = save_to_temp(dataset_folder.read(), "dataset.zip")
    
    if st.button("Process Images"):
        # Clean up before processing
        clean_up()
        reset_opencv()
        st.cache_data.clear()

        # Unzip the uploaded files if they exist
        if people_folder is not None:
            with zipfile.ZipFile(people_zip_path, 'r') as zip_ref:
                zip_ref.extractall("People")
            clean_up_temp(people_temp_dir)
        if dataset_folder is not None:
            with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
                zip_ref.extractall("Dataset")
            clean_up_temp(dataset_temp_dir)
        
        # Progress bar
        total_steps = 2
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process known people images
        status_text.text("Processing known people images...")
        processKnownPeopleImages(path="./People/")
        progress_bar.progress(1 / total_steps)
        
        # Process dataset images
        status_text.text("Processing dataset images...")
        processDatasetImages(path="./Dataset/")
        progress_bar.progress(2 / total_steps)
        
        # Zip the output folder
        zip_output_folder("./output", "output")
        
        st.success("Images processed and saved to output folder.")
        with open("output.zip", "rb") as f:
            st.download_button(
                label="Download Output",
                data=f,
                file_name="output.zip",
                on_click=clean_up
            )

if __name__ == "__main__":
    main()
