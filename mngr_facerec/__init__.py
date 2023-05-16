import cv2
import sys
import json
import time
import pickle
import random
import threading
import face_recognition
from pathlib import Path
from collections import Counter
from typing import TextIO
from os import walk, listdir, makedirs
from os.path import basename, \
     exists, join as join_paths


class FaceRecognizer():
    """
    A class that performs face recognition from a videoflow
    with options to update user labels and get the detected label as an id
    that can be later parsed from a json file

    Args:
    - folder_name: path to the folder containing images
    - encodings_location: location to the .pkl file
    - json_location: location to the .json file
    - model: the model that performs recognition ("hog"(default) or "cnn")
    - scale: by how much the image needs to be downgraded
    """
    def __init__(self,
                 folder_name: str,
                 encodings_location: str,
                 json_location: str,
                 model: str = "hog",
                 scale: int = 5,
                 ) -> None:
        # Id of the detected employee
        self.__userID = "-2"
        # Assign the passed image folder to the class instance
        self.__img_folder_name = folder_name
        # Empty array for labels
        self.__labels = []
        # Assign the passed model to the class instance
        self.__model = model
        # Assign the passed encodings location to the class instance
        self.__encodings_location = Path(encodings_location)
        # Assign the passed image width to the class instance
        self.__scale = scale
        # Assign the passed json location to the class instance
        self.__json_location = json_location
        
        # Create counters for three situations
        self.__known_count = 0
        self.__noone_count = 0
        self.__unknown_count = 0

        # Initialize the mode for collecting new user
        self.collection_mode = 0
        # Initialize the counter for images that are taken
        self.__taken_pictures = 0
        # Initialize new user id
        self.__new_user_id = None

        # Load data from the json file
        with open(self.__json_location) as f:
            self.__json_data = json.load(f)

        # Load the images folder
        err = self.__load_dir()
        if err:
            sys.stdout.write(f"{err}\n")
            return

        # Create the dict of labels from the provided folder
        self.__load_labels()

    def __load_dir(self) -> str:
        """
        Loads the image folder for finding images
        """

        # Check for the folder actually existing
        tmp_dir = self.__img_folder_name
        if not exists(tmp_dir):
            return "Folder does not exist"

        # Check for subfolders or files in the provided folder
        if len(listdir(tmp_dir)) == 0:
            return "Folder is empty"

        # Set the image directory and return no errors
        self.__images_dir = tmp_dir
        return None

    def __load_labels(self) -> None:
        """
        Fills in a list for the labels provided in the folder with images
        """

        # Reset the list of labels
        self.__labels = []
        for root, _, files in walk(self.__images_dir):
            for file in files:
                # Every file is being checked for being an image
                # (i.e. has a fornat of either .png or .jpg)
                if file.endswith("png") or file.endswith("jpg"):
                    # Set the name of the folder as a label
                    label = basename(root)
                    # If this label is not present in the dict - add it
                    if label not in self.__labels:
                        self.__labels.append(label)

        self.collection_mode = 0
        self.__cleanup_json()

    def __cleanup_json(self) -> None:
        """
        Removes unmatched folder to .json ids
        """

        # Rebase the dict only for those ids that are in the image folder
        self.__json_data = {item: val for (item, val) in self.__json_data.items() if item in self.__labels}

        # Write data to the json file
        with open(self.__json_location, "w") as f:
            json.dump(self.__json_data, f, indent=2)

    def __progress_bar(self,
                       it: list,
                       prefix: str = "",
                       size: int = 60,
                       out: TextIO = sys.stdout,
                       ) -> None:
        """
        Prints the progress bar to the stdout for better visualization

        Args:
        - it: list that goes through a loop
        - prefix: what will be printed before the progress bar
        - size: width of the progress bar in symbols
        - out: a way to print out the progress bar
        """
        count = len(it)
        if count == 0:
            return

        def show(j):
            x = int(size*j/count)
            sys.stdout.write(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count}\r")

        show(0)
        for i, item in enumerate(it):
            yield item
            show(i+1)
        sys.stdout.write("\n")

    def __encode_known_faces(self) -> None:
        """
        Creates encodings for every found label in the provided folder
        """

        # Init the lists for the labels
        # And their encodings
        labels = []
        encodings = []

        # Find every image in the folder
        for root, _, files in walk(self.__images_dir):
            # Separate filed for storing the label name
            label = basename(root)

            # Skip the images/ folder
            if label == basename(self.__images_dir):
                continue

            # Loop every file
            for file in self.__progress_bar(files, prefix=f"Encodings for label {label}: "):
                # If a file ends on the image format (.png/.jpg)
                if file.endswith("png") or file.endswith("jpg"):
                    # Create a path to file from the folder and label's name
                    file_path = join_paths(self.__images_dir, label, file)
                    # Load the image from image path
                    image = face_recognition.load_image_file(file_path)

                    # Resizes the image for faster processing
                    resized_image = cv2.resize(image,
                                               (int(image.shape[1] / self.__scale),
                                                int(image.shape[0] / self.__scale)),
                                               interpolation=cv2.INTER_AREA,
                                               )

                # Find the location of the face
                face_locations = face_recognition.face_locations(
                    resized_image,
                    model=self.__model,
                )
                # Generates encoding for the found face(s)
                face_encodings = face_recognition.face_encodings(
                    resized_image,
                    face_locations,
                )

                # For every found face
                # Append the names of labels and their encodings
                # To the corresponding lists
                for encoding in face_encodings:
                    labels.append(label)
                    encodings.append(encoding)

        # Create the dictionary from generated labels and encodings
        name_encodings = {"names": labels, "encodings": encodings}

        # Save the dictionary
        with self.__encodings_location.open(mode="wb") as f:
            pickle.dump(name_encodings, f)

    def __recognize_one_face(self,
                             uknown_encoding: list,
                             loaded_encodings: dict) -> str:
        """
        Compares the acquired face to the known encodings
        And returns the name of the most matched label

        Args:
        - uknown_encoding: encoding of the found face in the new image
        - loaded_encodings: encodings of the known faces
        """

        # Compares encodings of the known faces and the new image
        # As a array of matches containing
        # True and False for every known encoding
        boolean_matches = face_recognition.compare_faces(
            loaded_encodings["encodings"], uknown_encoding
        )

        # Counts how many of the matched encodings
        # Belong to the certain label
        votes = Counter(
            # Init the name of the label
            name
            # Iterate through every matche in the array
            # Parallel to theirs descriptor
            for match, name in zip(boolean_matches, loaded_encodings["names"])
            # If the matches array has a label name
            # Then increments the mount of matches
            # That belong to a certain label
            if match
        )

        # Returns the label that has got the most matches
        if votes:
            return votes.most_common(1)[0][0]

    def __take_picture(self, img: cv2.Mat) -> None:
        """
        Saves one picture of the user to the folder
        """
        if self.__taken_pictures < 40:
            sys.stdout.write(f"Collecting pictures {self.__taken_pictures+1}/40\n")
            cv2.imwrite(join_paths(self.__images_dir, str(self.__new_user_id), str(self.__taken_pictures+1)+".jpg"), img)
            self.__taken_pictures += 1

            time.sleep(0.1)
        else:
            self.__add_user()
            self.update_users()

    def __add_user(self) -> None:
        """
        Adds the new user to the json file
        """
        self.__json_data[self.__new_user_id] = {"name": "TO SET", "tasks": "TO SET"}

    def __create_user_id(self) -> None:
        """
        Creates a random number to be user id
        """
        self.__new_user_id = str(random.randrange(10**5, 10**6))

        while self.__new_user_id in self.__labels:
            self.__new_user_id = str(random.randrange(10**5, 10**6))

        makedirs(join_paths(self.__images_dir, self.__new_user_id))

    def __wait(self) -> None:
        """
        Waits for some time before the person 
        can be asked again to take pictures of
        """

        time.sleep(10)
        self.collection_mode = 0

    def get_user_id(self) -> str:
        """
        Returns the found face on the videoflow
        """
        if self.__userID in self.__labels:
            return self.__userID
        else:
            sys.stdout.write("There is no info about that user \
                             (it was probably added to the .pkl file previously\
                              but was deleted from training folder)\n")
            return "-2"

    def get_json_data(self) -> dict:
        """
        Returns the data read from the json file
        """
        return self.__json_data

    def update_users(self) -> None:
        """
        Updates the encodings of all the known images
        and generates new if were added
        """
        # Create the dict of labels from the provided folder
        self.__load_labels()

        # Create new encodings
        self.__encode_known_faces()

    def recognize_faces(self, img) -> None:
        """
        Performs the face recognition from the image
        """

        # Leave the method if the user is being taken pictures of
        if self.collection_mode == 1:
            return

        # Leave the method if the file specified does not exist
        if not exists(self.__encodings_location):
            sys.stdout.write("Specified file for encodings is not found\n")
            return

        # Loads the generated dictionary for the labels and encodings
        with self.__encodings_location.open(mode="rb") as f:
            loaded_encodings = pickle.load(f)

        # If the loaded file is empty - leave the method
        if len(loaded_encodings["names"]) == 0 or \
                len(loaded_encodings["encodings"]) == 0:
            sys.stdout.write("Empty encodings.pkl file\n")
            return

        small_frame = cv2.resize(img,
                                    (0, 0),
                                    fx=self.__scale/10.,
                                    fy=self.__scale/10.)

        # Finds the position of the face
        input_face_locations = face_recognition.face_locations(
            small_frame,
            model=self.__model
        )

        # If a face has been detected
        if input_face_locations:
            # Generates an encoding for the detected face(s)
            input_face_encodings = face_recognition.face_encodings(
                small_frame,
                input_face_locations,
            )

            # For every detected positions and encodings
            # Gets the encoding of one face
            for _, unknown_encoding in zip(
                    input_face_locations,
                    input_face_encodings):
                # Get the label of the found face
                name = self.__recognize_one_face(
                    unknown_encoding,
                    loaded_encodings,
                )

                # If nothing has been detected
                if not name:
                    # Set the corresponding name
                    name = "Unknown"
                    # Add to the uknown counter
                    self.__unknown_count += 1
                    # If there happens to be an unkown person
                    # In the frame for some time
                    if self.__unknown_count >= 5:
                        # Set the unknown user code
                        self.__userID = "-1"
                else:
                    # Reset the other counters
                    self.__unknown_count = 0
                    self.__noone_count = 0
                    self.__known_count += 1
                    # If there happens to be a known person
                    # In the frame for some time
                    if self.__known_count >= 10:
                        # Set the id of the recognized person
                        self.__userID = name
        else:
            # Reset the other counters
            self.__unknown_count = 0
            self.__known_count = 0
            self.__noone_count += 1
            # If there happens to be a known person
            # In the frame for some time
            if self.__noone_count >= 10:
                # Set the code for no faces in frame
                self.__userID = "-2"

    def create_user(self, img: cv2.Mat) -> None:
        """
        Gathers pictures from the passed images and creates a new user from them
        """

        # Leave the method if the user has chosen not to
        # Take pictures of them
        if self.collection_mode == 2:
            return
        
        # If user has agreed to take pictures of them
        if self.collection_mode == 1:
            self.__take_picture(img)
            return

        # Print the message
        sys.stdout.write("I don't know you\nCan I take some images of you (y/n)?\n")

        # Wait for the answer
        answer = input()

        # Interpret the answer
        if answer == "y":
            # If yes - collect photos of the user
            self.__create_user_id()
            self.__taken_pictures = 0
            self.collection_mode = 1
        elif answer == "n":
            # If no - do not collect faces of the user
            # Until the program is restarted or
            # Until the update_user method is called
            self.collection_mode = 2
            wait_thread = threading.Thread(target=self.__wait)
            wait_thread.start()
        else:
            return
    
    def _debug_faces(self):
        # Open the videoflow
        cap = cv2.VideoCapture(0)

        import time

        # Loop handling for the videoflow
        while cap.isOpened():
            # Reading every frame of the videoflow
            ret, frame = cap.read()
            st = time.time()
            # If a frame could not be processed
            # Break the loop
            if not ret:
                continue

            SCALE = 2.

            small_frame = cv2.resize(frame, (0, 0), fx=1./SCALE, fy=1./SCALE)

            # Finds the position of the face
            input_face_locations = face_recognition.face_locations(
                small_frame,
                model=self.__model
            )

            # If a face has been detected
            if input_face_locations:
                # Generates an encoding for the detected face
                input_face_encodings = face_recognition.face_encodings(
                    small_frame,
                    input_face_locations,
                )

                # For every detected positions and encodings
                # Gets the encoding of one face
                for bounding_box, _ in zip(input_face_locations,
                                           input_face_encodings):

                    top, right, bottom, left = bounding_box

                    cv2.rectangle(frame,
                                  (int(left*SCALE), int(top*SCALE)),
                                  (int(right*SCALE), int(bottom*SCALE)),
                                  (0, 0, 255), 2
                                  )

            et = time.time()

            iteration = et - st
            milliseconds = int(round(iteration * 1000))
            print(f"Has taken {milliseconds} ms")

            cv2.imshow("", frame)
            if cv2.waitKey(1) and 0xFF == ord('q'):
                break

        # Closes the videoflow
        cap.release()
        cv2.destroyAllWindows()
