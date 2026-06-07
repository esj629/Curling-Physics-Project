import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

class Stone:

    # constructor
    def __init__(self, xvel, yvel, angvel, mass, radius, radiusrun, xpos, ypos, angle, mu):
        self.__xvel = xvel
        self.__yvel = yvel
        self.__angvel = angvel
        self.__mass = mass
        self.__radius = radius
        self.__radiusrun = radiusrun
        self.__xpos = xpos
        self.__ypos = ypos
        self.__angle = angle
        self.__mu = mu
    
    # methods
    def calculate_Acceleration(self, icesheet, angles):
        forces = self.calculate_overall_force(angles, icesheet)
        accelerationx = -self.__mu * 9.81 + forces[0]
        accelerationy = forces[1]
        return accelerationx, accelerationy
    
    def calculate_angular_Acceleration(self):
        angular_acceleration = -2 * self.__mu * 9.81 * self.__radius
        return angular_acceleration
    
    def update_motion(self, dt, icesheet, angles):
        # calculating new velocities
        accelerations = self.calculate_Acceleration(icesheet, angles)
        self.__xvel = self.__xvel+(accelerations[0])*dt
        self.__yvel = self.__yvel+(accelerations[1])*dt
        self.__angvel = self.__angvel+(self.calculate_angular_Acceleration())*dt

        # making sure there are no negative velocities before changing position
        if(self.__xvel<=0):
            self.__xvel=0
        if(self.__angvel<=0):
            self.__angvel=0

        # calculate new position and angle
        self.__xpos = self.__xpos+self.__xvel*dt
        self.__ypos = self.__ypos+self.__yvel*dt
        self.__angle = self.__angle+self.__angvel*dt
    
    def divide_circumference(self, n):
        # divides the circumference into n points and returns the angle for each point
        angles = []
        theta = 0
        n=n
        while(theta<=(360-(360/n))):
            angles.append(theta)
            theta+=(360/n)
        return angles
    
    def calculate_overall_force(self, angles, icesheet):
        n: int=len(angles)
        diff_mu=0
        icesheet = icesheet
        xforce = 0
        yforce = 0
        x = int(np.ceil(n/2))
        for i in range (x):
            # difference in coefficients of friction
            x = int(np.ceil(n/2))
            mu1 = icesheet.find_mu(angles[i], self.__xvel, self.__yvel)
            mu2 = icesheet.find_mu(angles[x+i], self.__xvel, self.__yvel)
            diff_mu = mu1 - mu2
            force = diff_mu * self.__mass * 9.81
            xforce += force*np.sin(angles[i]*(np.pi/180))
            yforce += force*np.cos(angles[i]*(np.pi/180))
        return xforce, yforce
        
    
    # getters
    def get_xpos(self):
        return self.__xpos
    def get_ypos(self):
        return self.__ypos
    def get_xvel(self):
        return self.__xvel
    def get_angularv(self):
        return self.__angvel
    def get_angle(self):
        return self.__angle
    def get_yvel(self):
        return self.__yvel
    
class Ice_Sheet:

    # constructor
    def __init__(self, width, length, temperature, pebbles, scratches):
        self.__width = width
        self.__length = length
        self.__temperature = temperature
        self.__pebbles = pebbles
        self.__scratches = scratches
    
    # methods
    def find_mu(self, angle, xvelocity, yvelocity):
        mu = 0
        rotation = np.degrees(np.arctan2(yvelocity, xvelocity))
        angle2 = angle-rotation
        if(angle2>360):
            #wrap around
            angle2=angle2-360
        elif(angle2<0):
            angle2=angle2+360

        if(angle2>300 or angle2<60):
            mu = 0.012
        elif((angle2>=60 and angle2<=90) or (angle2>=270) and angle2<=300):
            mu = 0.014
        elif((angle2>=90 and angle2<=120) or (angle2>=240 and angle2<=270)):
            mu = 0.016
        elif(angle2>120 and angle2<240):
            mu = 0.018
        return mu

class Pebble:

    # constructor
    def __init__(self, xpos, ypos, radius, height):
        self.__xpos = xpos
        self.__ypos = ypos
        self.__radius = radius
        self.__height = height
    
    # methods

class Scratch:

    # constructor
    def __init__(self, xpos, ypos, depth, direction):
        self.__xpos = xpos
        self.__ypos = ypos
        self.__depth = depth
        self.__direction = direction
    
    # methods

class Model:

    # constructor
    def __init__(self):
        pass

    # methods

# next three are inheritance examples, subclasses of the model class
class Friction_Imbalance_Model(Model):

    # constructor
    def __init__(self):
        super().__init__()
    
    # methods

class Pivot_Slide_Model(Model):

    # constructor
    def __init__(self):
        super().__init__()
    
    # methods

class Scratch_Model(Model):

    # constructor
    def __init__(self):
        super().__init__()
    
    # methods





#-------------------------main bit--------------------------------------------------------
# timestep
dt=0.05
# stone info
x= 0
y=0
t=0
mass=19
v=3
angularv=1.0
g=9.81
angle=0
radius = 0.145
coef = 0.015

# instantiate the ice sheet
icesheet = Ice_Sheet(20, 45, -5, 0, 0)
# instantiate the stone
stone = Stone(v, 0, angularv, mass, radius, 0, x, y, angle, coef)

# graph stuff, labels, animation, limits
fig, axis = plt.subplots() 
xpositions = []
ypositions =[]
animated_path, = axis.plot([], [], '-.', color="red") # path of stone
animated_stone, = axis.plot([], [], 'o', markersize=15, color="blue") # creation of curling stone
animated_pointer, = axis.plot([], [], '-', color="yellow", linewidth=1) # pointer for rotation purposes
axis.set_xlim([0,45])
axis.set_ylim([-30,30])
axis.set_title("Stone Motion w/ No Curl")
axis.set_xlabel("X Displacement")
axis.set_ylabel("Y Displacement")
axis.set_aspect('equal', adjustable='box')
# add legend that will later show all numerical no.s
legend_text = axis.text(
    0.02, 0.95, "", transform=axis.transAxes,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)
)

# update function for the animation
def animate(frame):
    angles = stone.divide_circumference(30)
    stone.update_motion(dt, icesheet, angles)
    # previous positions of the stone
    x=stone.get_xpos()
    y=stone.get_ypos()
    xpositions.append(x)
    ypositions.append(y)

    # update graphics
    x2 = x+(0.90*np.cos(stone.get_angle()))
    y2 = y+(0.90*np.sin(stone.get_angle()))
    animated_stone.set_data([x], [y])
    animated_path.set_data(xpositions, ypositions)
    animated_pointer.set_data([x, x2], [y, y2])

    # update legend, shows velocity and other variables as they are updated
    legend_text.set_text(
        f"X Velocity: {stone.get_xvel():.3f}\n"
        f"Y Velocity: {stone.get_yvel():.3f}\n"
        f"Angular vel: {stone.get_angularv():.3f}")
        #f"X disp: {stone.get_xpos():.3f}\n"
        #f"Y disp: {stone.get_ypos():.3f}")

    return animated_stone, animated_path, animated_pointer

# animation
animation = FuncAnimation ( 
            fig=fig, 
            func=animate,
            frames=2000,
            interval=10, 
            repeat=False
)

# saves animation as a gif
#animation.save("More_Realistic_V2.1.gif", writer=PillowWriter(fps=10))

# plots everything
plt.show()


