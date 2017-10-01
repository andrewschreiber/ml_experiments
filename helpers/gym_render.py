from JSAnimation.IPython_display import display_animation
from matplotlib import animation
import matplotlib.pyplot as plt
from IPython.display import display

def display_frames_as_gif(frames, rects = None, scalar=0):
    """
    Displays a list of frames as a gif, with controls
    """
    width = frames[0].shape[1] / 72.0 + scalar
    height = frames[0].shape[0] / 72.0 + scalar

    plt.figure(figsize=(width, height), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    # Create a Rectangle patch


    def animate(i):
        patch.set_data(frames[i])
        if rects is not None:
            cluster = rects[i]
            rect = patches.Rectangle((cluster[0], cluster[1]),cluster[2]-cluster[0],cluster[3] cluster[1],linewidth=1,edgecolor='r',facecolor='none')
            patch.add_patch(rect)

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=30)
    display(display_animation(anim, default_mode='loop'))
