In this project is a live visual art piece thats going to rely on processing video feed (webcam or video). We are going to detect facial expressions of a person from the feed and then animate the screen based on the facial expression.

Here is what we need to build first:
  - we need to detect the users's face from a video stream and plot the facial features as points on the screen. The facial features would be detected like in the MediaPipe plug in from google. Then they would be plotted on top of the user's face.
  - next we are going to try to classify the user's emotion based on the features of the face. For example if the user's eyebrows are frowning it could be classified as 'thinking' or 'distress', if they are smiling then 'happy' or laughing.
  - Then we would draw boxes on the screen with text of what might be going through the user's head based on the emotion. We are going to sample 'happy' or 'distressed' thoughts from an llm and print them in the boxes.


Important configurability features of the project:
 - the visual aesthetic of the way we display thoughts is very important. we need to be able to control the line's thickness, the way boxes appear and so on. Think of similar projects in TouchDesigner (but we are not going to use TouchDesigner for this project).