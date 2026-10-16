import cv2
import numpy as np
def video(frames, folder):
    img_width, img_height = frames[0].size
    
    # Initialize OpenCV VideoWriter (10.0 is the Frames Per Second)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(f'{folder}/video.mp4', fourcc, 5.0, (img_width, img_height))
    
    for frame in frames:
        
        image_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        video_out.write(image_bgr)

    video_out.release()
    print(f"Video successfully saved to {folder}/video.mp4")