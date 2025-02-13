import cv2
import numpy as np
from ultralytics import solutions

# capture  the video
video_path = r'videos and images\5051126-hd_1280_720_30fps.mp4'
cap = cv2.VideoCapture(video_path)

# get original video information
w, h, fps = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FPS)))


# result saving
output_path = r"res\output_video3.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
out = cv2.VideoWriter(output_path, fourcc, fps, (1200, 800))


# get the ultralytics anallytics object
analytics = solutions.Analytics(
    show = True,
    model = 'yolo11m.pt',
    analytics_type  = 'bar',
    classes = [0,2]
)

frame_count = 0
# define the target  classes
target_classes = [0,2]

# define classes colors
green = (0,255,0)
red = (0,0,255)

while cap.isOpened():
    ret , frame = cap.read()
    if not ret:
        break

    # get the chart in the target classes
    frame_count +=1
    analytics_frame = analytics.process_data(frame,frame_count)
    # get the model detection results
    results = analytics.model(frame)
    detections = results[0].boxes

    for detection in detections :
        x1,y1,x2,y2 = map(int,detection.xyxy[0])
        conf = detection.conf[0]# Confidence score
        cls = int(detection.cls[0])  # Class index

        if cls in target_classes :
            # prepare class color
            color = green if cls == 0 else red
            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2),color, 2)

            # Draw the label with the class name and confidence
            label = f"{analytics.model.names[cls]} {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            label_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)




    # Resize Pie chart to fit the top portion of the frame (for example, half of the height)
    pie_height = int(h * 0.4)  # 40% of the video height for the pie chart
    pie_width = w  # Full width of the frame

    # Convert pie chart to a numpy array for overlay
    pie_chart_resized = cv2.resize(analytics_frame, (pie_width, pie_height))

    # Create a blank image with the same width as the frame and total height of both pie chart and video
    combined_frame = np.zeros((h + pie_height, w, 3), dtype=np.uint8)

    # Place the pie chart on top of the combined frame
    combined_frame[:pie_height, :] = pie_chart_resized

    # Place the original video frame below the pie chart in the combined frame
    combined_frame[pie_height:, :] = frame

    # Resize the combined frame to 1200x800
    resized_frame = cv2.resize(combined_frame, (1200, 800))

    out.write(resized_frame)
    cv2.imshow('frame',frame)


    if cv2.waitKey(1) & 0XFF ==ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()