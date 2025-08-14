This project was created by students Alessio A. and Salvatore F. for the **Open Day** event. The goal of the project was to develop a hand gesture recognition system that can detect and visualize hand movements in real-time. Using a combination of computer vision and machine learning techniques, the system processes input from hand landmark detection models and translates these into gesture-based commands. The project also includes a visual feedback mechanism that highlights key points of the hand and displays relevant information on the screen.

## **Main file**

This Python script is designed for controlling a DJI Tello drone, offering the ability to switch between two modes of control: keyboard and gesture-based control. The script leverages the `djitellopy` library to communicate with the drone and OpenCV to handle video processing and visualization. It also integrates threading to ensure the drone operates smoothly without blocking the main control loop. Here’s a formal breakdown of the code's structure and functionality:

### **Argument Parsing: `get_args()`**

The script starts by defining the `get_args()` function, which handles argument parsing using the `configargparse` library. This function allows the user to provide configuration values either through a command-line interface or a configuration file (`config.txt`). The arguments parsed include:

- **Video capture dimensions** (`--width`, `--height`).
- **Control mode** (`--is_keyboard`), determining if keyboard control is enabled.
- **Gesture control settings** (`--min_detection_confidence`, `--min_tracking_confidence`), allowing the user to adjust the sensitivity of gesture recognition.
- **Buffer length** (`--buffer_len`), which sets the number of gestures the system can remember.
- **Static image mode** (`--use_static_image_mode`), enabling the use of pre-captured images for gesture detection instead of live video.

These arguments help configure the program to behave according to the user's preferences.

### **Drone Setup and Control Initialization:**

In the `main()` function, the script initializes a connection to the Tello drone using the `djitellopy` library. The drone's video feed is activated with the `streamon()` method, and the frames are captured using `tello.get_frame_read()`. 

Additionally, two control mechanisms are initialized:
- **`GestureController`**: This class is responsible for controlling the drone using hand gestures.
- **`TelloKeyboardController`**: This class allows the drone to be controlled via keyboard input.

Furthermore, the script initializes a gesture recognition system using the `GestureRecognition` class, which processes incoming frames to detect hand gestures.

### **Control Mode Switching:**

The script provides two primary modes for controlling the drone:
- **Keyboard Control Mode**: Activated when the `keyboard_control` variable is `True`. The user can control the drone with specific keys:
  - `t` for takeoff.
  - `l` for landing.
  - `k` to switch to keyboard control mode.
  - `g` to switch to gesture control mode.
- **Gesture Control Mode**: If `keyboard_control` is `False`, gestures detected from the camera feed are used to control the drone.

Mode switching is handled via key presses (`k`, `g`) and is indicated by changes to the `mode` and `keyboard_control` variables.

### **Threading for Concurrent Operations:**

To ensure smooth operation, the script uses multiple threads to handle different tasks concurrently:
- **Drone Control (`tello_control`)**: This function, executed in a separate thread, decides whether to use keyboard or gesture control based on the mode selected.
- **Battery Status Monitoring (`tello_battery`)**: Battery status is fetched asynchronously in a separate thread to avoid blocking the main loop.

This approach allows for real-time control of the drone without delay from background operations such as battery monitoring or gesture recognition.

### **Video Frame Processing and Visualization:**

The script continuously processes video frames received from the drone’s camera:
- The frames are passed to the gesture recognition system, which detects and recognizes gestures.
- The `gestureBuffer` stores recognized gestures and helps determine the appropriate drone actions.
- Debug information, including gesture IDs and battery status, is drawn onto the frames.
- The processed frames are displayed using OpenCV’s `imshow()` function.

The script also calculates and displays the frames per second (FPS) to give the user feedback on the performance of the gesture recognition and video processing.

### **Exit Condition and Cleanup:**

The main loop continues to run until the user presses the `Esc` key (`27`), at which point:
- If the drone is in flight, it will land automatically.
- OpenCV windows are closed with `cv.destroyAllWindows()`.
- The video stream from the drone is turned off with `tello.streamoff()`.

This ensures that the program exits gracefully and that the drone is safely landed before the application terminates.

### **Key Considerations:**

- **Multithreading**: The script uses threads to perform concurrent tasks (like controlling the drone and monitoring battery status), which helps maintain responsiveness. However, careful consideration must be given to avoid race conditions or conflicts between threads, especially when accessing shared resources.
  
- **Global State Management**: Variables like `gestureBuffer` and `gestureID` are accessed globally, which could lead to potential issues with data consistency. It would be more robust to encapsulate these variables within classes to ensure proper state management.

- **Error Handling**: While the script does include basic error handling for battery retrieval (`tello.get_battery()`), further error handling for other parts of the code (e.g., drone connection failures or frame read errors) could improve the robustness of the program.

### **Conclusion:**

This script provides a versatile and interactive way to control a Tello drone, utilizing both keyboard and gesture-based inputs. By combining real-time video processing with multithreading, the script offers a responsive and flexible drone control interface. However, further improvements could be made in error handling, thread synchronization, and resource management to enhance the script’s reliability and performance.

---

## **GestureController Class**

The provided code snippet defines a class called `GestureController`, which is responsible for controlling a DJI Tello drone based on detected hand gestures. The class interprets gestures and translates them into specific drone movements by setting velocities for different axes (forward/backward, up/down, left/right, and yaw/rotation). These velocities are then sent to the drone using the `djitellopy` library's `send_rc_control()` method.

Here’s a formal explanation of the code's structure and functionality:

### **Class Definition: `GestureController`**
```python
class GestureController:
    def __init__(self, tello: Tello):
        self.tello = tello
        self._is_landing = False

        # RC control velocities
        self.for_back_velocity = 0
        self.up_down_velocity = 0
        self.left_right_velocity = 0
        self.yaw_velocity = 0
```

- **`__init__(self, tello: Tello)`**: This is the constructor of the `GestureController` class. It accepts an instance of the `Tello` drone object as a parameter (`tello`), which is used to send commands to the drone.
  - **`self.tello`**: Stores the Tello drone object for use throughout the class.
  - **`self._is_landing`**: A flag that indicates whether the drone is in the process of landing. Initially set to `False`.
  - **RC Control Velocities**: Four variables are initialized to zero. These represent the velocities for controlling the drone's movements:
    - **`for_back_velocity`**: Controls forward and backward motion.
    - **`up_down_velocity`**: Controls upward and downward motion.
    - **`left_right_velocity`**: Controls left and right movement.
    - **`yaw_velocity`**: Controls the rotation of the drone (yaw).

### **Gesture Control Method: `gesture_control`**
```python
def gesture_control(self, gesture_buffer):
    gesture_id = gesture_buffer.get_gesture()
    print("GESTURE", gesture_id)
```
- **`gesture_control(self, gesture_buffer)`**: This method takes a `gesture_buffer` object as an argument, which contains the most recent gesture ID detected by the gesture recognition system.
  - **`gesture_id = gesture_buffer.get_gesture()`**: Retrieves the current gesture ID from the gesture buffer.
  - **`print("GESTURE", gesture_id)`**: This line prints the gesture ID to the console for debugging purposes.

### **Gesture Processing Logic**
The core of the `gesture_control` method is a series of conditional checks that map each `gesture_id` to a specific movement command for the drone. These gestures are interpreted as follows:

```python
if not self._is_landing:
```
- The condition checks if the drone is not currently in the process of landing. If it is landing, no other gestures can affect the movement until the landing is complete.

#### **Mapping Gesture IDs to Drone Movements**
- **`gesture_id == 0` (Forward)**:
  ```python
  if gesture_id == 0:  # Forward
      self.for_back_velocity = 30
  ```
  - The gesture ID `0` corresponds to a "forward" gesture, setting the forward/backward velocity (`for_back_velocity`) to `30` to move the drone forward.

- **`gesture_id == 1` (Stop)**:
  ```python
  elif gesture_id == 1:  # STOP
      self.for_back_velocity = self.up_down_velocity = self.left_right_velocity = self.yaw_velocity = 0
  ```
  - The gesture ID `1` represents a "stop" gesture, which halts all movement by setting all velocity parameters to `0`.

- **`gesture_id == 5` (Back)**:
  ```python
  if gesture_id == 5:  # Back
      self.for_back_velocity = -30
  ```
  - The gesture ID `5` corresponds to a "backward" gesture, setting the forward/backward velocity (`for_back_velocity`) to `-30`, causing the drone to move backward.

- **`gesture_id == 2` (Up)**:
  ```python
  elif gesture_id == 2:  # UP
      self.up_down_velocity = 25
  ```
  - The gesture ID `2` indicates an "up" gesture, setting the upward/downward velocity (`up_down_velocity`) to `25`, making the drone ascend.

- **`gesture_id == 4` (Down)**:
  ```python
  elif gesture_id == 4:  # DOWN
      self.up_down_velocity = -25
  ```
  - The gesture ID `4` represents a "down" gesture, causing the drone to descend by setting the downward velocity (`up_down_velocity`) to `-25`.

- **`gesture_id == 3` (Land)**:
  ```python
  elif gesture_id == 3:  # LAND
      self._is_landing = True
      self.for_back_velocity = self.up_down_velocity = self.left_right_velocity = self.yaw_velocity = 0
      self.tello.land()
  ```
  - The gesture ID `3` corresponds to a "land" gesture. When this gesture is recognized, the drone stops all movement (all velocities set to `0`) and initiates landing by calling the `land()` method on the `tello` object. The flag `_is_landing` is also set to `True` to indicate the landing process.

- **`gesture_id == 6` (Left)**:
  ```python
  elif gesture_id == 6:  # LEFT
      self.left_right_velocity = 20
  ```
  - The gesture ID `6` is interpreted as a "left" gesture, which sets the left/right velocity (`left_right_velocity`) to `20`, causing the drone to move left.

- **`gesture_id == 7` (Right)**:
  ```python
  elif gesture_id == 7:  # RIGHT
      self.left_right_velocity = -20
  ```
  - The gesture ID `7` represents a "right" gesture, causing the drone to move right by setting the left/right velocity (`left_right_velocity`) to `-20`.

- **`gesture_id == -1` (Invalid Gesture)**:
  ```python
  elif gesture_id == -1:
      self.for_back_velocity = self.up_down_velocity = self.left_right_velocity = self.yaw_velocity = 0
  ```
  - The gesture ID `-1` signifies an invalid or unrecognized gesture. In such a case, all velocities are set to `0`, effectively stopping the drone.

### **Sending RC Control Commands**
```python
self.tello.send_rc_control(
    self.left_right_velocity,
    self.for_back_velocity,
    self.up_down_velocity,
    self.yaw_velocity,
)
```
- **`self.tello.send_rc_control()`**: After determining the correct velocities based on the recognized gesture, this method sends the corresponding control signals to the drone. The parameters represent the velocities for left/right, forward/backward, up/down, and yaw (rotation), controlling the drone’s movement in the air.

### **Conclusion:**
The `GestureController` class enables gesture-based control of the DJI Tello drone. It interprets hand gestures captured by a gesture recognition system and maps them to specific drone movements. The class allows the drone to move in all directions (forward, backward, up, down, left, right), stop, and land, based on the recognized gestures. 

The control is achieved by adjusting the drone's velocities along different axes and sending those control commands to the drone using the `send_rc_control()` method. Additionally, the class ensures that no movements are made while the drone is landing, preventing conflicting commands during the landing process.

---
## **KeyboardController Class**
### Objective:
The objective of the provided code is to control a Tello drone using keyboard input through a Python program. The program defines a `TelloKeyboardController` class that allows the user to control the drone's movement and other functionalities by pressing specific keys on the keyboard.

### Structure of the Code:
1. **TelloKeyboardController Class:**
   - This class is designed to interact with the `Tello` drone (from the `djitellopy` library) and control its movement based on keyboard inputs.
   - The constructor method (`__init__`) initializes the controller with an instance of the Tello drone. The instance is passed as a parameter to the constructor when the controller object is created.
   - The `control` method takes a key (as a string) and executes the corresponding drone command. It maps each key to a specific movement or action (e.g., moving forward, rotating, or landing).

2. **Movement Commands:**
   The `control` method recognizes the following keys and sends commands to the drone accordingly:
   - `'w'`: Move the drone forward by 30 cm.
   - `'s'`: Move the drone backward by 30 cm.
   - `'a'`: Move the drone left by 30 cm.
   - `'d'`: Move the drone right by 30 cm.
   - `'e'`: Rotate the drone clockwise by 30 degrees.
   - `'q'`: Rotate the drone counter-clockwise by 30 degrees.
   - `'r'`: Move the drone upward by 30 cm.
   - `'f'`: Move the drone downward by 30 cm.
   - `'x'`: Trigger an emergency landing.

3. **Listening for Keyboard Input:**
   - The `listen_for_controls` method continuously monitors the keyboard for key presses using the `keyboard.is_pressed` method. 
   - When a valid key is pressed (i.e., one of `'w', 's', 'a', 'd', 'e', 'q', 'r', 'f'` or `'x'`), the corresponding movement function is executed. If the key `'x'` is pressed, the drone will execute an emergency landing and the control loop will terminate.

4. **Tello Drone Connection and Execution:**
   - In the `__main__` block, an instance of the Tello drone is created and connected to the Python program using the `Tello()` constructor and the `connect()` method. 
   - The battery level is retrieved and printed for reference using `tello.get_battery()`.
   - An instance of `TelloKeyboardController` is created and the `listen_for_controls()` method is invoked to start the keyboard control loop.
   - Once the control loop ends (either through the `x` key or other termination), the `end()` method is called to safely disconnect from the drone.

### Key Libraries and Methods:
1. **`djitellopy` Library:**
   The Tello drone is controlled using the `djitellopy` library, which provides an easy interface to communicate with the Tello drone. It supports commands for movement, rotation, and other functionalities like video streaming, battery status, etc.
   - `Tello()`: Creates an instance of the Tello drone.
   - `connect()`: Establishes a connection to the drone.
   - `move_forward(distance)`, `move_back(distance)`, `move_left(distance)`, `move_right(distance)`: Commands that move the drone in various directions by the specified distance in centimeters.
   - `rotate_clockwise(degrees)`, `rotate_counter_clockwise(degrees)`: Commands that rotate the drone by the specified number of degrees.
   - `move_up(distance)`, `move_down(distance)`: Commands that control the altitude of the drone.
   - `get_battery()`: Retrieves the battery level of the drone.
   - `land()`: Initiates an emergency landing, safely bringing the drone to the ground.
   - `end()`: Disconnects the program from the Tello drone and ends the session.

2. **`keyboard` Library:**
   The `keyboard` library is used to capture keyboard input during runtime. The `keyboard.is_pressed` method checks if a specific key is currently pressed.

### Additional Considerations:
1. **Safety and Reliability:**
   - **Emergency Landing**: The `'x'` key is reserved for emergency landing. This functionality ensures that the drone can be safely brought down in case of an unforeseen issue or hazard.
   - **Battery Level**: The program retrieves and prints the battery level of the drone upon connection to help the user monitor the drone's remaining power and avoid mid-flight battery failure.

2. **Keyboard Input Handling:**
   - The program utilizes a continuous loop to monitor for key presses. Once a key is pressed, it triggers the corresponding movement or action. The loop will run until the `'x'` key is pressed, at which point the drone will land and the program will exit.

3. **Control Structure:**
   - The design is modular, where the `TelloKeyboardController` class handles all aspects of drone control, and the input handling is separated into a distinct method (`listen_for_controls`). This structure improves code readability and maintainability.

4. **Extensions and Enhancements:**
   - Additional commands such as hovering, camera control, or more intricate movement patterns could be added by extending the `control` method.
   - The program could also be enhanced by integrating GUI elements (e.g., with `pygame`) to display the drone's status or provide visual feedback to the user.

### Example of Usage:
1. Ensure the Tello drone is powered on and connected to your computer via Wi-Fi.
2. Run the Python script.
3. The program will display the battery level of the drone.
4. Use the following keys to control the drone:
   - `'w'` to move forward
   - `'s'` to move backward
   - `'a'` to move left
   - `'d'` to move right
   - `'r'` to move up
   - `'f'` to move down
   - `'e'` to rotate clockwise
   - `'q'` to rotate counter-clockwise
   - `'x'` to land the drone

Once the `'x'` key is pressed, the drone will land and the program will terminate.

---

### Conclusion:
This program provides a straightforward way to control a Tello drone via keyboard input. The modular design ensures ease of modification and extension, making it a flexible foundation for more complex drone control applications. The code effectively combines basic movement commands with safety features such as emergency landing.

---
## **GestureRecognition Class**

This class seems to be responsible for recognizing and drawing key points and gestures from hand landmark data. It processes hand gestures by drawing lines between key landmarks on the hand and rendering a visual representation of the gesture on an image. Additionally, it handles the display of various hand gesture-related information.

#### Method 1: `_drawHand`

The **`_drawHand`** method is designed to visualize hand landmarks and the connections between them, based on a set of points. 

1. **Hand Landmarks**:
   The method first loops through a list of landmarks, `landmarkPoint`, which likely contains coordinates for each hand keypoint (e.g., wrist, fingers, etc.).

2. **Drawing the Lines Between Landmarks**:
   - For each finger (thumb, index, middle, ring, little), the method draws lines between adjacent joints using OpenCV’s `cv.line` function.
   - It first draws a thick black line, followed by a thinner white line on top. This ensures the line appears prominently against the background.
   - It uses `cv.line(image, start_point, end_point, color, thickness)` where the `start_point` and `end_point` are tuples representing the (x, y) coordinates of each landmark.

3. **Drawing Keypoints**:
   - For each landmark (from wrist to fingertip), the method draws a circle to mark its position.
   - Depending on the importance of the landmark (e.g., wrist, fingertip), the circles have different radii and colors.
   - It uses `cv.circle(image, (x, y), radius, color, thickness)` to draw these keypoints on the image.
   - For example:
     - The wrist landmarks (index 0 and 1) are marked with smaller circles, while the fingertip landmarks are marked with larger circles.

4. **Visualization**:
   - The image with drawn landmarks and lines is returned, allowing for the visualization of the hand and its movements.

#### Method 2: `_drawInfoTest`

The **`_drawInfoTest`** method is responsible for overlaying text and information onto the image, likely to show the classification of the hand gesture.

1. **Drawing a Rectangular Info Box**:
   - A black rectangular box is drawn at the top of the hand’s bounding box to hold text information.
   - The method uses `cv.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), color, thickness)` to draw the rectangle.

2. **Displaying Gesture Information**:
   - The method checks if a `handSignText` is provided (indicating a recognized gesture) and appends it to the `info_text`.
   - It then displays the information inside the rectangle using `cv.putText`, which overlays the gesture text on the image.

3. **Output**:
   - The image with the overlaid information is returned.

#### Method 3: `_drawBoundRect`

The **`_drawBoundRect`** method draws a bounding rectangle around the hand region if the `useRect` flag is set to `True`.

1. **Drawing the Rectangle**:
   - It checks if `useRect` is `True`, and if so, it draws the bounding rectangle using the `cv.rectangle()` function. The rectangle's corners are defined by the coordinates in `rect`.
   
2. **Output**:
   - The image with the bounding rectangle drawn (if applicable) is returned.

---

### Class: **`GestureBuffer`**

The **`GestureBuffer`** class is designed to store a buffer of gestures and analyze the most frequent gesture within a specified window of time. This is useful for recognizing continuous gestures over time, as it can keep track of recent gestures and identify the dominant gesture.

#### Method 1: `__init__`

The constructor initializes the buffer with a specified length (`buffer_len`), which determines how many recent gestures will be stored.

1. **Parameters**:
   - `buffer_len` (default 10): The maximum number of gestures the buffer will store at any given time.
   
2. **Buffer**:
   - A `deque` (from the `collections` module) is used to store the gestures. A `deque` is a double-ended queue that efficiently supports appending and popping from both ends, which is ideal for managing a sliding window of gestures.
   - The `deque` is initialized with a maximum length of `buffer_len`.

#### Method 2: `add_gesture`

This method adds a new gesture ID to the buffer.

1. **Adding the Gesture**:
   - The `gesture_id` is appended to the `deque`, storing the most recent gesture.

#### Method 3: `get_gesture`

The **`get_gesture`** method analyzes the buffer and returns the most frequent gesture if it meets a specific threshold of occurrences.

1. **Analyzing the Buffer**:
   - The method uses `Counter(self._buffer).most_common()` to count the occurrences of each gesture in the buffer and identify the most frequent one.
   
2. **Threshold Check**:
   - If the most frequent gesture appears at least `(buffer_len - 1)` times in the buffer, it is considered the dominant gesture. In this case, the buffer is cleared (`self._buffer.clear()`), and the gesture ID is returned.
   
3. **No Dominant Gesture**:
   - If no gesture meets the frequency threshold (i.e., no gesture occurs enough times in the buffer), the method returns `None`, indicating that no dominant gesture was detected.
   
4. **Use Case**:
   - This approach can be used to smooth out fluctuations or noise in gesture detection by considering only those gestures that occur repeatedly over a series of frames.

---

### Summary:

- **`GestureRecognition` class**: Handles the drawing of landmarks, lines, and keypoints on an image to visualize hand gestures. It also overlays information about recognized gestures, such as the hand sign and finger gestures.
- **`GestureBuffer` class**: Provides a buffer to store and analyze gestures over time. It identifies the most frequent gesture in the buffer and returns it if it meets the necessary frequency threshold, allowing for continuous gesture recognition.

This structure is useful in applications like gesture-based control systems, where continuous and stable gesture recognition is required, and a visual representation of hand landmarks is needed for feedback or debugging.

If you have further questions or need additional clarification on any part, feel free to ask!

---

## **FrameCalculator Class** 

The `FrameCalculator` class is designed to calculate and track the frames per second (FPS) of the video stream in a computer vision application. This can be helpful for performance monitoring, ensuring that the system runs smoothly in real-time. Below is a detailed breakdown of how this class works:

#### 1. **Imports**
```python
from collections import deque
import cv2 as cv
```
- **`deque`**: The `deque` (double-ended queue) is used to store the FPS calculations for a specified number of frames. It helps in calculating the average FPS over a buffer of frames, giving a more stable and accurate reading.
- **`cv2` (OpenCV)**: OpenCV is the primary computer vision library used for handling video and image processing tasks. Here, it is used to access system ticks and frame timing.

#### 2. **Class Definition**
```python
class FrameCalculator(object):
    def __init__(self, buffer_len=1):
```
- The class `FrameCalculator` has an initializer (`__init__`) method that takes an optional parameter `buffer_len`, which defines the number of frames to average for FPS calculation. By default, this is set to `1`, which means the FPS is calculated based on the current frame.

#### 3. **Initialization**
```python
self._start_tick = cv.getTickCount()
self._frequency = 1024.0 / cv.getTickFrequency()
self._differences = deque(maxlen=buffer_len)
```
- **`self._start_tick`**: This variable stores the initial system tick count (i.e., the number of ticks since the system started). `cv.getTickCount()` is a function from OpenCV that returns the system’s tick count, which helps in measuring elapsed time with high precision.
  
- **`self._frequency`**: The frequency of the system ticks is calculated by dividing `1024.0` by `cv.getTickFrequency()`. This gives the time per tick in seconds (or fractions of a second). OpenCV provides `cv.getTickFrequency()` to return the frequency of the clock used by `cv.getTickCount()`, typically in ticks per second.

- **`self._differences`**: This deque is used to store the differences in time (i.e., the time intervals between frames). The size of this deque is determined by `buffer_len`, which specifies how many frame time intervals are retained to compute the average FPS.

#### 4. **Method: `get()`**
```python
def get(self):
    current_tick = cv.getTickCount()
    different_time = (current_tick - self._start_tick) * self._frequency
    self._start_tick = current_tick

    self._differences.append(different_time)

    fps = 1024.0 / (sum(self._differences) / len(self._differences))
    fps_rounded = round(fps, 2)

    return fps_rounded
```
- **`current_tick`**: This variable stores the current system tick count, just like in the initializer. Each time the method `get()` is called, it fetches the current tick count to calculate the time difference between the current and previous frames.
  
- **`different_time`**: The time difference between the current tick and the previous tick is calculated by subtracting `self._start_tick` from `current_tick`, and then multiplying by the frequency (`self._frequency`). This gives the time interval in seconds for the current frame.

- **`self._start_tick = current_tick`**: The current tick count is updated to be the new starting point for the next frame's time calculation.

- **`self._differences.append(different_time)`**: The calculated time difference for the current frame is appended to the deque. If the deque exceeds its maximum length (`buffer_len`), the oldest value is automatically discarded.

- **FPS Calculation**: The FPS is calculated by taking the reciprocal of the average frame time. This is done by summing the time differences in the deque and dividing by the number of stored differences (`len(self._differences)`), then inverting the result to get FPS:
  ```python
  fps = 1024.0 / (sum(self._differences) / len(self._differences))
  ```
  This equation calculates the average FPS by averaging the frame time and converting it to FPS.

- **Rounding**: The FPS is rounded to two decimal places using Python's `round()` function:
  ```python
  fps_rounded = round(fps, 2)
  ```

- **Return**: Finally, the method returns the calculated FPS value (`fps_rounded`).

---

### Summary of Functionality:
The `FrameCalculator` class tracks the time intervals between frames and calculates the FPS by averaging the time per frame over a buffer of several frames. This provides a more stable and accurate representation of the real-time frame rate, which can be helpful for performance monitoring in video processing or computer vision applications.

The class allows for dynamic adjustment of how many frames should be considered in the FPS calculation via the `buffer_len` parameter, making it flexible for various applications.

---

**This project is under GPL License - Provided by Istituto Tecnico Industriale - Enrico Medi**


---

**Created by:** Alessio A. & Salvatore F.

*Thanks to all contributors for thier help*
