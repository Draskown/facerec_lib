
# Face recognizer for the manager robot

Program that does face recognition in the context of applying it to the robot assistant to greet employees of Applied Robotics company

## Installation

1. Download the archive
2. Unzip it in your environment's directory
3. Write a the command `pip install -e .` to install the library into your environment
4. Import it as you would any library (i.e. `import mngr_facerec`)

## Usage

In order to use the package you need to create a `FaceRecognizer` class instance and:

- call `recognize_faces` method in order to immediately perform face recognition from the videoflow. Breaks if the encodings.pkl was altered of not generated at all. To generate the .pkl file use the following method.
- call `update_users` method in order to update the .pkl file accodring to the folder where the images are located.
- call `get_user_id` method in order to get the id of the user that was recognized.
- call `get_json_data` method in order to load the data from .json file with which the system operates.
- call `create_user` method in order to start the gathering of the images for the bran`d new id.

## Examples

- The simplest example is provided in the `example.py` file.

```python
from mngr_facerec import FaceRecognizer
import threading
import os.path


# A function to endlessly get user id
def print_user_id() -> None:
    while fr.get_user_id() is not None:
        print(fr.get_user_id())


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

```

- More advanced example with ROS2 integration you can find [here](https://github.com/Draskown/facerec_pkg)