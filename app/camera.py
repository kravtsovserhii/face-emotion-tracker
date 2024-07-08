import cv2

class Camera:
    '''
    Camera class to capture video frames
    '''
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        
    def get_frame(self):
        '''
        Get the video frame
        '''
        success, frame = self.video.read()
        if not success:
            return None
        return frame
    
    def display_frame(self, frame, faces, emotions):
        '''
        Display the frame with faces and emotions
        '''
        for (x, y, w, h), emotion in zip(faces, emotions):
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Display the emotion
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.video.release()
            cv2.destroyAllWindows()
            exit(0)

