from mngr_facerec.mngr_facerec import FaceRecognizer
import threading
import os.path
import logging
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)


# A function to endlessly get user id
def print_user_id() -> None:
    while True:
        user_id = fr.get_user_id()
        if user_id is not None:
            logging.info(user_id)
            print(user_id)
        else:
            logging.warning(user_id)


if __name__ == "__main__":
    # Create an instance of the recognizer class
    # And pass the image folder
    fr = FaceRecognizer(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "Images"),
        )

    # Create thread for recognizing faces on a videoflow
    recon_thread = threading.Thread(target=fr.recognize_faces)
    # Create thread for printing the recognized user id
    get_thread = threading.Thread(target=print_user_id)
    # Start the threads
    recon_thread.start()
    get_thread.start()
