import matplotlib.pyplot as plt
import random
import math
import numpy as np
from scipy.stats import beta
import seaborn as sns

sns.set(style="darkgrid", font_scale=1.5)

# plt.rcParams.update({"text.usetex": True})

plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = False

# alpha_d = 0.1
# beta_d = 1.0
img = plt.imread("../../pics/maps/pacific_northwest_blank.PNG")
num_points = 500
city_center = [
    [-5, -7],
    [-5,-1],
    [7,3],
    [7,-7]
]

def sample_point(radius, beta_a=1.0, beta_b=1.0):
    # Generate point on perimeter
    a = random.random() * 2 * math.pi
    
    # Generate radius
    r = radius * beta.rvs(beta_a, beta_b) ** 0.5 
    # Get points based on polar coordinates
    x = r * math.cos(a)
    y = r * math.sin(a)
    
    return x, y


def plot_pts(ax, alpha_d=1.0, beta_d=1.0, radius=4):
    
    for i in range(len(city_center)):
        pts = []

        for _ in range(num_points):
            x, y = sample_point(radius, alpha_d, beta_d)

            x += city_center[i][0]
            y += city_center[i][1]

            x = (x + 10) / 20 * img.shape[0]
            y = (y + 10) / 20 * img.shape[0]
            pts.append((x,y))

        ax.scatter(*zip(*pts), s = 4)



fig, axs = plt.subplots(2, 3)

#axs[1].scatter(*zip(*create_pts()), s = 4)

plot_pts(axs[0,0])
plot_pts(axs[0,1], alpha_d=0.1)
plot_pts(axs[0,2], alpha_d=0.1, beta_d=5.0)

plot_pts(axs[1,0], radius=8)
plot_pts(axs[1,1], alpha_d=0.1, radius=8)
plot_pts(axs[1,2], alpha_d=0.1, beta_d=5.0, radius=8)



for i in range(2):
    for j in range(3):
        axs[i, j].imshow(img)
        axs[i, j].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        axs[i, j].grid(False)
        #axs[i, j].set(aspect="equal")
        # for spine in axs[i, j].gca().spines.values():
        #     spine.set_visible(False)



axs[0,0].set_title(r"$\alpha_b = 1.0, \beta_b = 1.0$")
axs[0,1].set_title(r"$\alpha_b = 0.1, \beta_b = 1.0$")
axs[0,2].set_title(r"$\alpha_b = 0.1, \beta_b = 5.0$")

axs[1,0].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

axs[0,0].set_ylabel(r"$radius = 4$", rotation=0, size="medium", labelpad=50)
axs[1,0].set_ylabel(r"$radius = 8$", rotation=0, size="medium", labelpad=50)

fig.set_size_inches(15, 7.5)


plt.tight_layout()
plt.savefig("cities.png", dpi=200)


