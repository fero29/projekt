import cv2
from tflite_runtime.interpreter import load_delegate
from tflite_runtime.interpreter import Interpreter
import numpy as np
from PIL import Image
from time import sleep
import time

def scale_image(frame, new_size=(224, 224)):
  # Get the dimensions
  height, width, _ = frame.shape # Image shape
  new_width, new_height = new_size # Target shape 

  # Calculate the target image coordinates
  left = (width - new_width) // 2
  top = (height - new_height) // 2
  right = (width + new_width) // 2
  bottom = (height + new_height) // 2
  
  image = frame[left: right, top: bottom, :]
  return image

def time_elapsed(start_time,event):
        time_now=time.time()
        duration = (time_now - start_time)*1000
        duration=round(duration,2)
        print (">>> ", duration, " ms (" ,event, ")")
       
      
#-----initialise the Model and Load into interpreter-------------------------

#specify the path of Model and Label file

model_path = "mobilenet_v1_1.0_224_quant_edgetpu.tflite" 
label_path = "labels_mobilenet_quant_v1_224.txt"

top_k_results = 2

with open(label_path, 'r') as f:
    labels = list(map(str.strip, f.readlines()))

# Load TFLite model and allocate tensors
interpreter = Interpreter(model_path=model_path, 
    experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

## Get input size
input_shape = input_details[0]['shape']
#print(input_shape)
size = input_shape[:2] if len(input_shape) == 3 else input_shape[1:3]
#print(size)

#prediction threshold for triggering actions
threshold=0.5


#---------------------------------------------------------

camera = cv2.VideoCapture(0)


#loop continuously (press control + 'c' to exit program)
for i in range(10):
    start_time = time.time()
    
    #----------------------------------------------------
    start_t1=time.time()
    stream = np.empty((480, 640, 3), dtype=np.uint8)
    
    #camera.capture(stream, 'rgb',use_video_port=True)
    ret, frame = camera.read()
    #stream = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = scale_image(stream)
    
    time_elapsed(start_t1,"camera capture")
    #----------------------------------------------------------------
    
    
    #-------------------------------------------------------------
    start_t2=time.time()
    # Add a batch dimension
    input_data = np.expand_dims(img, axis=0)
    
    # feed data to input tensor and run the interpreter
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Obtain results and map them to the classes
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Get indices of the top k results
    top_k_indices = np.argsort(predictions)[::-1][:top_k_results]
    
    
    pred_max=predictions[top_k_indices[0]]/255.0
    lbl_max=labels[top_k_indices[0]]
    
            
            
    if (pred_max >= threshold):
            percent=round(pred_max*100)
            txt= " " + lbl_max + " (" + str(percent) + "%)"
            print(txt)
            
    
    time_elapsed(start_t2,"inference")
    #-------------------------------------------------------------
    
    #time_elapsed(start_time,"overall")
    
    print(lbl_max, pred_max)
    print("********************************")
    time.sleep(1)
        
camera.release()