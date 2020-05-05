import pyautogui
import matplotlib.pyplot as plt

image = pyautogui.screenshot()
plt.imshow(image)

while(True):
    image = pyautogui.screenshot()
