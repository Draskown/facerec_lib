import face_recognition
import pickle
import cv2
import sys
from pathlib import Path
from collections import Counter
from typing import TextIO
from os import walk, listdir
from os.path import basename, \
     exists, join as join_paths


class FaceRecognizer():
    """
    A class that performs face recognition from a videoflow
    with options to update user labels and get the detected label as an id
    that can be later parsed from a json file

    Args:
    - folder_name: path to the folder containing images
    - model: the model that performs recognition ("hog"(default) or "cnn")
    - location: location to the .pkl file for the generated encodings
    - scale: by how much the image needs to be downgraded
    """
    def __init__(self,
                 folder_name: str,
                 encodings_location: str,
                 model: str = "hog",
                 scale: int = 5,
                 ) -> None:
        # Id of the detected employee
        self.__userID = 0
        # Assign the passed image folder to the class instance
        self.__img_folder_name = folder_name
        # Empty dict for labels
        self.__labels = {}
        # Assign the passed model to the class instance
        self.__model = model
        # Assign the passed encodings location to the class instance
        self.__encodings_location = Path(encodings_location)
        # Assign the passed image width to the class instance
        self.__scale = scale
        

        # Create counters for three situations
        self.__known_count = 0
        self.__noone_count = 0
        self.__unknown_count = 0

        # Load the images folder
        err = self.__load_dir()
        if err:
            sys.stdout.write(f"{err}\n")
            return

        # Create the dict of labels from the provided folder
        self.__load_labels()

    def get_user_id(self) -> int:
        """
        Returns the found face on the videoflow
        """

        return self.__userID

    def update_users(self) -> None:
        """
        Updates the encodings of all the known images
        and generates new if were added
        """

        # Create the dict of labels from the provided folder
        self.__load_labels()

        # Create new encodings
        self.__encode_known_faces()

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
        Creates a dict for the labels provided in the folder with images
        """

        # Set the first id to zero
        current_id = 0
        for root, _, files in walk(self.__images_dir):
            for file in files:
                # Every file is being checked for being an image
                # (i.e. has a fornat of either .png or .jpg)
                if file.endswith("png") or file.endswith("jpg"):
                    # Set the name of the folder as a label
                    label = basename(root)
                    # If this label is not present in the dict - add it
                    if label not in self.__labels:
                        self.__labels[label] = current_id
                        current_id += 1

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

    def recognize_faces(self, img) -> None:
        """
        Performs the face recognition from the videocapture
        """

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
                        self.__userID = -1
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
                self.__userID = -2

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
