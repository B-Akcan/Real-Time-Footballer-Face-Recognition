import cv2 as cv
from deepface import DeepFace
import pandas as pd

# Set the webcam for recording
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv.CAP_PROP_FPS, 30)

while cap.isOpened():
    # Read the next frame
    success, frame = cap.read()
    
    if not success:
        break

    # Calculate the cosine distance between the frame and all images in the database, select the ones with distance less then threshold
    lst = DeepFace.find(img_path=frame, db_path="images", model_name="SFace", enforce_detection=False, distance_metric="cosine", threshold=0.6)

    # Convert list of dataframes to a single dataframe
    df = pd.DataFrame()
    for i in lst:
        df = df._append(i)

    if not df.empty:
        # Select the entry with the shortest distance
        entry = df[df["distance"] == df["distance"].min()]

        player_name = entry["identity"].tolist()[0].split("\\")[1].upper()
        distance = entry["distance"].tolist()[0]
        x = entry["source_x"].tolist()[0]
        y = entry["source_y"].tolist()[0]
        w = entry["source_w"].tolist()[0]
        h = entry["source_h"].tolist()[0]

        # Put text and draw a rectangle around the player's face
        cv.putText(frame, f"Player: {player_name}", (x, y-30), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        cv.putText(frame, f"Similarity: {(1-distance)*100:.2f}%", (x, y-10), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv.putText(frame, "Press 'q' to quit", (10, 20), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
    cv.imshow("Webcam", frame)

    # If 'q' is pressed, quit
    k = cv.waitKey(1)
    if k == ord("q"):
        break

cap.release()
cv.destroyAllWindows()

