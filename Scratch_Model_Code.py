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
    def __init__(self, xvel, yvel, angvel, mass, radius, radiusrun, xpos, ypos, angle, mu, rotation, pivoting, pivot_t, pivot_p, icesheet_r):
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
        self.__pivoting = pivoting
        self.__pivot_t = pivot_t
        self.__pivot_p = pivot_p
        self.__grid_cell_i = int(np.floor(xpos/icesheet_r.get_cellsize()))
        self.__grid_cell_j = int(np.floor((ypos+ icesheet_r.get_width()/2)/icesheet_r.get_cellsize()))
        self.__rel_peb_list = []
        self.__p_a_v = 0
        self.__p_radius = 0
        self.__exclusion_list = []
        self.__pivot_pebble = 0
        self.__velocity_angle = 0
        self.__counter=0
        self.__coefficient2 = 0
    
    # methods
    def calculate_Acceleration(self, icesheet, forces):
        xvelocity = self.__xvel
        yvelocity = self.__yvel
        velocity = np.sqrt(xvelocity*xvelocity+yvelocity*yvelocity)
        coefficient=0
        if velocity>0:
            coefficient = 0.025*velocity**(-0.5)
        if coefficient > 0.3:
            coefficient=0.3
        self.__coefficient2 = coefficient
        xvelocity = self.__xvel
        yvelocity = self.__yvel
        anglev = np.arctan2(yvelocity, xvelocity)

        fx = coefficient*9.81*np.cos(anglev)
        fy = coefficient*9.81*np.sin(anglev)
        xthermal = forces[0]
        ythermal = (self.__rotation)*forces[1]

        accelerationx = -xthermal - fx
        accelerationy = -ythermal - fy
        return accelerationx, accelerationy
    
    def calculate_angular_Acceleration(self):
        angular_acceleration = -2 * self.__mu * 9.81 * self.__radius
        return angular_acceleration
    
    def update_motion(self, dt_s, dt_p, icesheet, forces, model):
        pebbleslist = self.calculate_pebbles_touching_running_band(self.__rel_peb_list)[0]

        # determine whether the stone should be pivoting or sliding
        if(self.__pivoting == True and self.__pivot_t >= dt_p):
            self.pivot(dt_p, icesheet)
        elif(self.__pivoting == True and self.__pivot_t < dt_p):
            self.slide(dt_s, forces, icesheet)
            self.__pivoting = False
            self.check_excluded_pebbles()
            self.__exclusion_list.append(self.__pivot_pebble)
        else:
            if(model == "Pi"):
                pivot_point = self.find_pivot_point(pebbleslist)
                if(pivot_point[0]>=0):
                    self.__p_radius = np.sqrt((pivot_point[0]-self.__xpos)**2+(pivot_point[1]-self.__ypos)**2)
                    vel = np.sqrt(self.__xvel**2 + self.__yvel**2)
                    self.__p_a_v = vel/self.__p_radius

                    self.__pivoting = True
                    xvelocity = self.__xvel
                    yvelocity = self.__yvel
                    velocity = np.sqrt(xvelocity*xvelocity+yvelocity*yvelocity)
                    self.__pivot_t = (0.000000045/velocity)
                    self.__pivot_p = pivot_point
                    self.__counter +=1
                    self.pivot(dt_p, icesheet)
                else:
                    self.slide(dt_s, forces, icesheet)
            else:
                self.slide(dt_s, forces, icesheet)

    def slide(self, dt, forces, icesheet):
        # calculating new velocities
        accelerations = self.calculate_Acceleration(icesheet, forces)
        p_xvel = self.__xvel + accelerations[0] * dt
        p_yvel = self.__yvel + accelerations[1] * dt

        # X velocity
        if self.__xvel > 0 and p_xvel < 0:
            self.__xvel = 0
        else:
            self.__xvel = p_xvel

        # Y velocity
        if self.__yvel != 0 and np.sign(self.__yvel) != np.sign(p_yvel): # shouldnt go from positive to negative to positive
            self.__yvel = 0
        else:
            self.__yvel = p_yvel

        self.__angvel = self.__angvel+(self.calculate_angular_Acceleration())*dt
        if(self.__angvel<=0):
            self.__angvel=0
        new_x = self.__xpos+self.__xvel*dt
        new_y = self.__ypos+self.__yvel*dt

        # check if changed grid cell
        self.check_grid_cell(new_x, new_y, icesheet)

        # calculate new position and angle
        self.__xpos = new_x
        self.__ypos = new_y
        self.__angle = self.__angle+self.__angvel*dt
        
    def pivot(self, dt, icesheet):
        acc = self.calculate_angular_Acceleration()
        X1 = self.__pivot_p[0] # X1 and Y1 are for pivot point
        Y1 = self.__pivot_p[1]
        X2 = self.__xpos       # X2 and Y2 are for stone centre
        Y2 = self.__ypos
        angv = self.__p_a_v
        radius = self.__p_radius
        pi=np.pi
        # change in distances
        dx = X2-X1
        dy = Y2-Y1
        angle = np.arctan2(dy,dx)
        # wrap around corrections
        if(angle<0):
            angle = angle+2*pi
        elif(angle>=(2*pi)):
            angle = angle-2*pi
        angle_change = angv*dt*self.__rotation
        self.__velocity_angle += angle_change
        new_angle = angle + angle_change # add on the change in angle
        X2 = np.cos(new_angle)*radius + X1
        Y2 = np.sin(new_angle)*radius + Y1
        self.__xvel = radius * angv * np.cos(self.__velocity_angle)
        self.__yvel = -radius * angv * np.sin(self.__velocity_angle)
        # check if changed grid cell
        self.check_grid_cell(X2, Y2, icesheet)
        self.__pivot_t = self.__pivot_t - dt
        self.__xpos = X2
        self.__ypos = Y2
        self.__angle += self.__angvel*dt
        self.__p_a_v = angv + acc*dt
        self.__angvel += acc*dt

    def check_excluded_pebbles(self):
        e_list = self.__exclusion_list
        pebbles = self.calculate_pebbles_touching_running_band(e_list)[0]
        self.__exclusion_list = pebbles
            

    def divide_circumference(self, n):
        # divides the circumference into n points and returns the angle for each point
        angles = []
        theta = 0
        n=n
        pi=np.pi
        while(theta<=((2*pi)-((2*pi)/n))):
            angles.append(theta)
            theta+=((2*pi)/n)
        return angles
    
    def calculate_contact_area(self, angle, n):
        width = 360/n
        area = 0
        # angular width of each section is width
        # add width over 2 to each angle to get angular section boundaries
        # calculate x and y values
        pebblesrel = self.calculate_pebbles_touching_running_band()[0]
        dxs = self.calculate_pebbles_touching_running_band()[2]
        dys = self.calculate_pebbles_touching_running_band()[3]

        for i in range (len(pebblesrel)): # assign angle to each pebble
            dx = dxs[i]
            dy = dys[i]
            theta = np.arctan2(dx, dy)
            if(dy>0):
                theta=-theta+2*np.pi
            else:
                theta=-theta
            
            # check if the pebble is inside circle of stone
            diff = ((theta - angle + np.pi) % 2*np.pi) - np.pi
            if abs(diff) <= width/2:
                # add areas
                area += np.pi*(pebblesrel[i].get_radius())**2
        return area
    
    def calculate_pebbles_touching_running_band(self, pebbles):
        pebblesrel = pebbles
        pebblesnew = []
        #distances = []
        dxs = []
        dys = []
        for i in range(len(pebblesrel)):
            dx = pebblesrel[i].get_xpos() - self.__xpos
            dy = pebblesrel[i].get_ypos() - self.__ypos
            distance2 = dx*dx + dy*dy
            radius2 = self.__radiusrun * self.__radiusrun
            radius_inner2 = (self.__radiusrun-0.005)*(self.__radiusrun-0.005)
            if(radius_inner2 <=  distance2 <= radius2):
                #distances.append(distance2)
                dxs.append(dx)
                dys.append(dy)
                pebblesnew.append(pebblesrel[i])
        return pebblesnew, dxs, dys

    def find_pivot_point(self, pebbles_list):
        pivot_point = [0,0]
        pebbles_list = list(set(pebbles_list)-set(self.__exclusion_list))
        length = len(pebbles_list)
        if(length == 0):
            pivot_point = [-2,0]
        elif(length == 1):
            x=pebbles_list[0].get_xpos()
            y=pebbles_list[0].get_ypos()
            pivot_point = [x, y]
            self.__pivot_pebble = pebbles_list[0]
        else:
            z = random.randint(0, length-1)
            pivot_point = [pebbles_list[z].get_xpos(), pebbles_list[z].get_ypos()]
            self.__pivot_pebble = pebbles_list[z]
        return pivot_point

    def check_grid_cell(self, x, y, icesheet):
        i = int(np.floor(x/icesheet.get_cellsize()))
        j = int(np.floor((y+ icesheet.get_width()/2)/icesheet.get_cellsize()))
        # compare to previous values
        i2 = self.__grid_cell_i
        j2 = self.__grid_cell_j
        answer = False
        if(i==i2 and j==j2):
            answer=False
        else:
            answer=True
            pebbles = icesheet.get_relevant_pebbles(i, j)
            self.__grid_cell_i=i
            self.__grid_cell_j=j
            self.set_rel_pebbles(pebbles)

    def Front_Or_Back(self, pebbles_t, dxs, dys):
            x_velocity = self.__xvel
            y_velocity = self.__yvel
            front = []
            back = []
            angles_f = []
            angles_b = []

            # directional angle of the stone
            theta = np.arctan2(y_velocity, x_velocity)
            if(y_velocity>0):
                theta=-theta+2*np.pi
            else:
                theta=-theta
            self.__velocity_angle = theta
            angle = theta
            # determine whether each pebble is in the front or back of the stone
            for i in range(len(pebbles_t)):
                #calculate angle for each pebble compared to the stone centre
                theta = np.arctan2(dys[i], dxs[i])
                if(dys[i]>0):
                    theta=-theta+2*np.pi
                else:
                    theta=-theta
                if((angle-np.pi/2) <= theta <= (angle+np.pi/2)):
                    # front of stone
                    front.append(pebbles_t[i])
                    angles_f.append(theta)
                else:
                    # back of stone
                    back.append(pebbles_t[i])
                    angles_b.append(theta)
            return front, back, angles_f, angles_b
            
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
    def set_rel_pebbles(self, rel_pebbles):
        self.__rel_peb_list = rel_pebbles
    def get_rel_pebbles(self):
        return self.__rel_peb_list
    def get_counter(self):
        return self.__counter
    def get_coefficient2(self):
        return self.__coefficient2
    def get_Velocity_Angle(self):
        return self.__velocity_angle

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
        pi=np.pi
        rotation = -np.arctan2(yvelocity, xvelocity)
        angle2 = angle-rotation
        if(angle2>(2*pi)):
            #wrap around
            angle2=angle2-(2*pi)
        elif(angle2<0):
            angle2=angle2+(2*pi)
        if(angle2>(5*pi/3) or angle2<(pi/3)):
            mu = 0.0120
        elif(((pi/3)<=angle2<=(pi/4)) or ((3*pi/2)<=angle2<=(5*pi/3))):
            mu = 0.0121
        elif(((pi/2)<=angle2<=(2*pi/3) ) or ((4*pi/3)<=angle2<=(3*pi/2))):
            mu = 0.0122
        elif((2*pi/3)<angle2<(4*pi/3)):
            mu = 0.0123
        return mu
    
    def calculate_mu_pebble(self, angle, stone, n):
        velocity = (stone.get_xvel()**2 + stone.get_yvel()**2)**(0.5)
        area = stone.calculate_contact_area(angle, n)
        mu = (velocity/2)*((stone.get_mass()/84)*0.007886 + 0.019*area - 6000*area**2)
        return mu

    def create_pebble_field(self):
        # create pebbbles
        pebbles = []
        n = round((self.__length*self.__width)/(0.0075)) # no. pebbles to generate
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
    
    def get_relevant_pebbles(self, i, j):
        # first, find which grid cell currently in from the x and y co-ordinates
        i = i
        j = j
        # add all elements from grid cell and adjacent ones to a list of pebbles
        plist = []

        for e in [-1, 0, 1]:
            for f in [-1, 0, 1]:
                m = i + e
                n = j + f
                if (0 <= m < (len(self.__grid))) and (0 <= n < (len(self.__grid[0]))):
                    plist.extend(self.__grid[m][n])
        return plist

    # getters and setters
    def get_cellsize(self):
        return self.__cellsize
    def get_width(self):
        return self.__width
        

            

# -----------------------------------Ice-------Features------------------------------------------------------------
class Pebble:

    # constructor
    def __init__(self, xpos, ypos, radius, height):
        self.__xpos = xpos
        self.__ypos = ypos
        self.__radius = radius
        self.__height = height
        self.__scratch = None
    
    # methods
    def Check_For_Scratch(self):
        ans = False
        if(self.__scratch == None):
            pass
        else:
            ans = True
        return ans

    def Add_Scratches(self, pebbles, angles):
        for i in range(len(pebbles)):
            ans = pebbles[i].Check_For_Scratch()
            new_angle = (angles[i] + np.pi/2) % (2*np.pi)
            if (ans):
                # replace old scratch with new
                scratch = Scratch(pebbles[i].get_xpos(), pebbles[i].get_ypos(), 0, new_angle)
                pebbles[i].Set_Scratch(scratch)
            else:
                # instantiate new scratch
                scratch = Scratch(pebbles[i].get_xpos(), pebbles[i].get_ypos(), 0, new_angle)
                pebbles[i].Set_Scratch(scratch)


    # getters
    def get_radius(self):
        return self.__radius
    def get_xpos(self):
        return self.__xpos
    def get_ypos(self):
        return self.__ypos
    def get_Scratch_Angle(self):
        return self.__scratch.get_Angle()
    def Set_Scratch(self, scratch):
        self.__scratch = scratch


class Scratch:

    # constructor
    def __init__(self, xpos, ypos, depth, direction):
        self.__xpos = xpos
        self.__ypos = ypos
        self.__depth = depth
        self.__direction = direction
    
    # methods
    def Rewrite_Scratch(self, depth, direction):
        self.__depth = depth
        self.__direction = direction

    def get_Angle(self):
        return self.__direction

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

    def Calculate_Pebble_Force(self, stone, icesheet):
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
    def Calculate_Scratch_Force(self, stone_, D):
        p = stone_.get_rel_pebbles()
        pebbles, dxs, dys = stone_.calculate_pebbles_touching_running_band(p)
        front, back, angles_f, angles_b = stone_.Front_Or_Back(pebbles, dxs, dys)
        force_y = 0 
        force_x = 0

        # add the scratches
        if(len(front) > 0):
            front[0].Add_Scratches(front, angles_f)

        # back interacting with scratches
        for x in range(len(back)):
            ans = back[x].Check_For_Scratch()
            if(ans):
                # apply the force 
                angle = back[x].get_Scratch_Angle()
                kappa = D * np.abs(np.cos(angle))
                force = kappa * (stone_.get_coefficient2()*stone_.get_mass()*9.81)/len(back)
                force_x += -force*np.sin(stone_.get_Velocity_Angle())
                force_y += force*np.cos(stone_.get_Velocity_Angle())
        return force_x, force_y
    

        




#-------------------------main bit---------------------------------------------------------------------------------
def main():
    # timestep
    dt_p=0.000000005
    dt_s=0.000167/2
    # stone info
    x=0
    y=0
    t=0
    mass=19
    v=3
    angularv=1.0
    g=9.81
    angle=0
    radius = 0.145
    coef = 0.01

    # instantiate the ice sheet
    icesheet = Ice_Sheet(20, 45, -5, 0, 0, 0.25)
    # instantiate the stone
    stone = Stone(v, 0, angularv, mass, radius, 0.0635, x, y, angle, coef, 1, False, 0, [-2,0], icesheet)
    # instantiate pebbles
    pebblefield = icesheet.create_pebble_field()
    pebblegrid = icesheet.create_pebble_grid()


    answer = input("Choose A Model To Run: T, Pe, Pi, S (Standing for: Thermal, Pebble, Pivot, Scratch)")

    if(answer == "T"):
        # Instantiate Thermal Model
        model = Friction_Imbalance_Thermal_Model()
    elif(answer == "Pe"):
        # Instantiate Pebble Model
        model = Friction_Imbalance_Pebble_Model()
    elif(answer == "Pi"):
        # Instantiate Pivot Slide Model
        model = Pivot_Slide_Model()
    elif(answer == "S"):
        # Instantiate Scratch Model
        model = Scratch_Model()


    # graph stuff, labels, animation, limits
    fig, axis = plt.subplots() 
    xpositions = []
    ypositions =[]
    animated_path, = axis.plot([], [], '-.', color="red") # path of stone
    animated_stone, = axis.plot([], [], 'o', markersize=15, color="blue") # creation of curling stone
    animated_pointer, = axis.plot([], [], '-', color="yellow", linewidth=1) # pointer for rotation purposes
    axis.set_xlim([0,38])
    axis.set_ylim([-3,3])
    axis.set_title("Stone Motion with an Anti-Clockwise Rotation")
    axis.set_xlabel("X Displacement")
    axis.set_ylabel("Y Displacement")
    axis.set_aspect('equal', adjustable='box')
    # add legend that will later show all numerical no.s
    legend_text = axis.text(
        0.02, 0.95, "", transform=axis.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)
    )
    # steps per frame
    steps = 10000

    # update function for the animation
    def animate(frame):
        for i in range (steps):
            if(answer == "T"):
                forces = model.Calculate_Thermal_Force(stone, icesheet)
            elif(answer == "Pe"):
                forces = model.Calculate_Pebble_Force(stone, icesheet)
            elif(answer == "Pi"):
                forces = [0,0]
            elif(answer == "S"):
                forces = model.Calculate_Scratch_Force(stone)
            stone.update_motion(dt_s, dt_p, icesheet, forces, model)
            #angles = stone.divide_circumference(30)

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
        print("y: ", stone.get_ypos(), "P_c: ", stone.get_counter())
        return animated_stone, animated_path, animated_pointer

    # animation
    animation = FuncAnimation ( 
                fig=fig, 
                func=animate,
                frames=10000,
                interval=10, 
                repeat=False
    )

    # saves animation as a gif
    #animation.save("Pebble_Imbalance_Model.gif", writer=PillowWriter(fps=10))

    # plots everything
    plt.show()
    

#main()
def run_scratch_model(D):

    dt_p = 0.000000005
    dt_s = 0.000167 / 2

    for i in range(6):
        icesheet = Ice_Sheet(20, 45, -5, 0, 0, 0.25)
        stone = Stone(
                3, 0, 1.0, 19, 0.145, 0.0635,
                0, 0, 0, 0.01, 1, False, 0, [-2, 0], icesheet)
        print("Run ", i+1)
        icesheet.create_pebble_field()
        icesheet.create_pebble_grid()

        model = Scratch_Model()

        while stone.get_angularv() > 0:
            forces = model.Calculate_Scratch_Force(stone, D)
            stone.update_motion(dt_s, dt_p, icesheet, forces, model)
        print(stone.get_ypos())

run_scratch_model(0.15)


