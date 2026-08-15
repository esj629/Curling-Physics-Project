import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
import random

# ------------------------------stone-------class-----------------------------------------------------------
class Stone:

    # constructor
    def __init__(self, xvel, yvel, angvel, mass, radius, radiusrun, xpos, ypos, angle, mu, rotation):
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
        self.__rotation = rotation
    
    # methods
    def calculate_Acceleration(self, icesheet, forces):
        xvelocity = self.__xvel
        yvelocity = self.__yvel
        anglev = np.arctan2(yvelocity, xvelocity)

        fx = self.__mu*9.81*np.cos(anglev)
        fy = self.__mu*9.81*np.sin(anglev)
        xthermal = forces[0]
        ythermal = (self.__rotation)*forces[1]

        accelerationx = -xthermal - fx
        accelerationy = -ythermal - fy
        return accelerationx, accelerationy
    
    def calculate_angular_Acceleration(self):
        angular_acceleration = -2 * self.__mu * 9.81 * self.__radius
        return angular_acceleration
    
    def update_motion(self, dt, icesheet, forces):
        # calculating new velocities
        accelerations = self.calculate_Acceleration(icesheet, forces)
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
    
    def calculate_contact_area(self, angle, pebbles, n):
        width = 360/n
        area = 0
        # angular width of each section is width
        # add width over 2 to each angle to get angular section boundaries
        # calculate x and y values
        pebblesrel = icesheet.get_relevant_pebbles(self.__xpos, self.__ypos)

        for i in range (len(pebblesrel)): # assign angle to each pebble
            xval = pebblesrel[i].get_xpos() - self.__xpos
            yval = pebblesrel[i].get_ypos() - self.__ypos
            theta = np.degrees(np.arctan2(yval, xval))
            if(yval>0):
                theta=-theta+360
            else:
                theta=-theta
            
            # check if the pebble is inside circle of stone
            dx = pebblesrel[i].get_xpos() - self.__xpos
            dy = pebblesrel[i].get_ypos() - self.__ypos
            distance = np.sqrt(dx*dx + dy*dy)
            diff = ((theta - angle + 180) % 360) - 180
            if((self.__radiusrun-0.005) <=  distance <= (self.__radiusrun)):
                if abs(diff) <= width/2:
                    # add areas
                    area += np.pi*(pebblesrel[i].get_radius())**2
        return area

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
    def get_rotation(self):
        return self.__rotation
    def get_mass(self):
        return self.__mass
    def get_radius_band(self):
        return self.__radiusrun


# ---------------------------------ice-----sheet-----class-----------------------------------------------------
class Ice_Sheet:

    # constructor
    def __init__(self, width, length, temperature, pebbles, scratches, cellsize):
        self.__width = width
        self.__length = length
        self.__temperature = temperature
        self.__pebbles = pebbles
        self.__scratches = scratches
        nx = int(length / cellsize)
        ny = int(width / cellsize)
        self.__grid = [[[] for j in range(ny)] for i in range(nx)]
        self.__cellsize = cellsize
    
    # methods
    def find_mu(self, angle, xvelocity, yvelocity):
        mu = 0
        rotation = -np.degrees(np.arctan2(yvelocity, xvelocity))
        angle2 = angle-rotation
        if(angle2>360):
            #wrap around
            angle2=angle2-360
        elif(angle2<0):
            angle2=angle2+360
        if(angle2>300 or angle2<60):
            mu = 0.0120
        elif((60<=angle2<=90) or (270<=angle2<=300)):
            mu = 0.0121
        elif((90<=angle2<=120 ) or (240<=angle2<=270)):
            mu = 0.0122
        elif(120<angle2<240):
            mu = 0.0123
        return mu
    
    def calculate_mu_pebble(self, angle, stone, n):
        velocity = (stone.get_xvel()**2 + stone.get_yvel()**2)**(0.5)
        area = stone.calculate_contact_area(angle, icesheet.__pebbles, n)
        mu = (velocity/2)*((stone.get_mass()/84)*0.007886 + 0.019*area - 6000*area**2)
        return mu

    def create_pebble_field(self):
        # create pebbbles
        pebbles = []
        n = round((self.__length*self.__width)/(0.0075)) # no. pebbles to generate
        #n=1000
        for i in range (n):
            xposition = random.uniform(0, self.__length)
            yposition = random.uniform(-self.__width/2, self.__width/2)
            pebbles.append(Pebble(xposition,yposition,(0.0013/2),0.000013))
        self.__pebbles = pebbles
        return self.__pebbles

    def create_pebble_grid(self):
        # assign each pebble to a grid cell
        n = len(self.__pebbles)
        for z in range (n):
            # round both values down
            i = int(np.floor(self.__pebbles[z].get_xpos()/ self.__cellsize))
            j = int(np.floor((self.__pebbles[z].get_ypos() + self.__width/2) / self.__cellsize))
            self.__grid[i][j].append(self.__pebbles[z])
        return self.__grid
    
    def get_relevant_pebbles(self, x, y):
        # first, find which grid cell currently in from the x and y co-ordinates
        i = int(np.floor(x/self.__cellsize))
        j = int(np.floor((y+ self.__width/2)/self.__cellsize))
        # add all elements from grid cell and adjacent ones to a list of pebbles
        plist = []

        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                m = i + di
                n = j + dj

                if (0 <= m < (len(self.__grid))) and (0 <= n < (len(self.__grid[0]))):
                    plist.extend(self.__grid[m][n])

        return plist
        

            

# -----------------------------------Ice-------Features------------------------------------------------------------
class Pebble:

    # constructor
    def __init__(self, xpos, ypos, radius, height):
        self.__xpos = xpos
        self.__ypos = ypos
        self.__radius = radius
        self.__height = height
    
    # methods
    def get_radius(self):
        return self.__radius
    def get_xpos(self):
        return self.__xpos
    def get_ypos(self):
        return self.__ypos

class Scratch:

    # constructor
    def __init__(self, xpos, ypos, depth, direction):
        self.__xpos = xpos
        self.__ypos = ypos
        self.__depth = depth
        self.__direction = direction
    
    # methods

# -----------------------------------------Model----Classes----------------------------------------------------------
class Model:

    # constructor
    def __init__(self):
        pass

    # methods

# next three are inheritance examples, subclasses of the model class
class Friction_Imbalance_Thermal_Model(Model):

    # constructor
    def __init__(self):
        super().__init__()
    
    # methods
    def Calculate_Thermal_Force(self, stone, icesheet):
        angles = stone.divide_circumference(30)
        n: int=len(angles)
        diff_mu=0
        icesheet = icesheet
        xforce = 0
        yforce = 0
        x = int(np.ceil(n/2))
        for i in range (x):
            # difference in coefficients of friction
            mu1 = icesheet.find_mu(angles[i], stone.get_xvel(), stone.get_yvel())
            mu2 = icesheet.find_mu(angles[x+i], stone.stone.get_xvel(), stone.get_yvel())
            diff_mu = mu1 - mu2
            force = diff_mu * stone.get_mass() * 9.81
            xforce += force*np.sin(angles[i]*(np.pi/180))
            yforce += force*np.cos(angles[i]*(np.pi/180))
        yforce = yforce/(x)
        xforce = xforce/(x)
        return xforce, yforce

class Friction_Imbalance_Pebble_Model(Model):
    # constructor
    def __init__(self):
        super().__init__()

    def Calculate_Pebble_Force(self, pebbles, stone, icesheet):
        angles = stone.divide_circumference(30)
        n: int=len(angles)
        diff_mu=0
        icesheet = icesheet
        xforce = 0
        yforce = 0
        x = int(np.ceil(n/2))
        for i in range (x):
            # difference in coefficients of friction
            mu1 = icesheet.calculate_mu_pebble(angles[i], stone, n)
            mu2 = icesheet.calculate_mu_pebble(angles[x+i], stone, n)
            diff_mu = mu1 - mu2
            force = diff_mu * stone.get_mass() * 9.81
            xforce += force*np.sin(angles[i]*(np.pi/180))
            yforce += force*np.cos(angles[i]*(np.pi/180))
        yforce = yforce/(x)
        xforce = xforce/(x)
        return xforce, yforce
    




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





#-------------------------main bit---------------------------------------------------------------------------------
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
icesheet = Ice_Sheet(20, 45, -5, 0, 0, 0.25)
# instantiate the stone
stone = Stone(v, 0, angularv, mass, radius, 0.0635, x, y, angle, coef, -1)
# instantiate pebbles
pebblefield = icesheet.create_pebble_field()
pebblegrid = icesheet.create_pebble_grid()

# instantiate model
#thermal_model = Friction_Imbalance_Thermal_Model()
# calculate x and y forces
#forces = thermal_model.Calculate_Thermal_Force(stone, icesheet)

# instantiate Model
pebble_model = Friction_Imbalance_Pebble_Model()
# calculate x and y forces
forces = pebble_model.Calculate_Pebble_Force(pebblefield, stone, icesheet)
    


# graph stuff, labels, animation, limits
fig, axis = plt.subplots() 
xpositions = []
ypositions =[]
animated_path, = axis.plot([], [], '-.', color="red") # path of stone
animated_stone, = axis.plot([], [], 'o', markersize=15, color="blue") # creation of curling stone
animated_pointer, = axis.plot([], [], '-', color="yellow", linewidth=1) # pointer for rotation purposes
axis.set_xlim([0,38])
axis.set_ylim([-4,4])
axis.set_title("Stone Motion with a Clockwise Rotation")
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
    forces = pebble_model.Calculate_Pebble_Force(pebblefield, stone, icesheet)
    stone.update_motion(dt, icesheet, forces)
    # previous positions of the stone
    x=stone.get_xpos()
    y=stone.get_ypos()
    xpositions.append(x)
    ypositions.append(y)

    # update graphics
    x2 = x+(0.90*np.cos(stone.get_angle()))
    y2 = y-(stone.get_rotation())*(0.90*np.sin(stone.get_angle()))
    animated_stone.set_data([x], [y])
    animated_path.set_data(xpositions, ypositions)
    animated_pointer.set_data([x, x2], [y, y2])

    # update legend, shows velocity and other variables as they are updated
    legend_text.set_text(
        f"X Velocity: {stone.get_xvel():.3f}\n"
        f"Y Velocity: {stone.get_yvel():.3f}\n"
        f"Angular vel: {stone.get_angularv():.3f}\n"
        #f"X disp: {stone.get_xpos():.3f}\n"
        f"Y disp: {stone.get_ypos():.3f}")

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
#animation.save("Thermal_Clockwise.gif", writer=PillowWriter(fps=10))

# plots everything
plt.show()


