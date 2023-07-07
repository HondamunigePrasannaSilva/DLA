import glob
import os
from PIL import Image

# Create gif
def make_gif(titolo="prova", k = 10):
    frames = []
    
    for i in range(k):
        gg = glob.glob(f"Lab1/img/forgif/{i}.png")
        frames.append(Image.open(gg[0]))
    
    #frames = [Image.open(image) for image in glob.glob("Lab1/img/forgif/*.png")]
    
    frame_one = frames[0]
    frame_one.save("Lab1/img/"+str(titolo)+".gif", format="GIF", append_images=frames,
               save_all=True, duration=200, loop=0)

    """removing_files = glob.glob('Lab1/img/forgif/*.png')
    for i in removing_files:
      os.remove(i)"""

