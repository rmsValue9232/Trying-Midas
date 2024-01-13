# Import dependencies
import cv2
import torch
import matplotlib.pyplot as plt 

# Download the MiDaS
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cuda')
midas.eval()

# Input transformation pipeline
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform 

# Hook into OpenCV
print("Opening camera...")
cap = cv2.VideoCapture(0)
i = 1 # frame number, to see the frame update
while cap.isOpened():
    print("Capturing the frame #{}".format(i))
    ret, frame = cap.read()

    # Transform input for midas 
    print("    Transform input for midas") 
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to('cuda')

    # Make a prediction
    print("    Make a prediction")
    with torch.no_grad(): 
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size = img.shape[:2], 
            mode='bicubic', 
            align_corners=False
        ).squeeze()

        output = prediction.cpu().numpy()

        # print(output)
    print("    showing the output image ->")
    plt.imshow(output)
    print("    showing the camera frame now ->")
    cv2.imshow('CV2Frame', frame)
    plt.pause(0.00001)

    i = i + 1

    if cv2.waitKey(10) & 0xFF == ord('q'): 
        cap.release()
        cv2.destroyAllWindows()

plt.show()