# my hybrid model :)

# import libraries
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
import random
import pandas as pd
from numba import njit, config
config.BOUNDSCHECK = False

# stone class
class Stone:

    # constructor
    def __init__(self, xvel, yvel, angvel, mass, radius, radiusrun, xpos, ypos, angle, mu, rotation, pivoting, pivot_t, pivot_p, icesheet_r, width):
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
        self.__rel_peb_list = np.empty(0, dtype=np.int64)
        self.__p_a_v = 0
        self.__p_radius = 0
        self.__p_radius2 = 0
        self.__exclusion_list = []
        self.__pivot_pebble = 0
        self.__velocity_angle = 0
        self.__counter=0
        self.__coefficient2 = 0
        self.__width = width
        self.__velocity = np.sqrt(self.__xvel*self.__xvel+self.__yvel*self.__yvel)
    # methods
    def divide_circumference(self, n):
        # for friction imbalance model, divides circumference into n equal parts
        return list(np.linspace(0, 2*np.pi, n, endpoint=False))
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
    def get_width(self):
            return self.__width           
    def get_velocity(self):
            return self.__velocity

# ice sheet class
class Ice_Sheet:
    # constructor
    def __init__(self, width, length, temperature, pebbles, scratches, cellsize):
        self.__width = width
        self.__length = length
        self.__temperature = temperature

        # no. grid cells in x and y
        nx = int(np.ceil(length / cellsize))
        ny = int(np.ceil(width / cellsize))
        # size of grid cell
        self.__cellsize = cellsize
        # grid, to list indices of pebbles that are in each grid cell, 2D
        self.__grid = [[[] for j in range(ny)] for i in range(nx)]
    # methods
    def create_pebble_grid(self, pebbles_x, pebbles_y):
        # assign each pebble to a grid cell
        n = len(pebbles_x)
        for z in range (n):
            # round both values down
            i = int(np.floor(pebbles_x[z]/ self.__cellsize))
            j = int(np.floor((pebbles_y[z] + self.__width/2) / self.__cellsize))
            self.__grid[i][j].append(z) 

        # flatten this grid to make it more usable for numba
        nx = len(self.__grid)
        ny = len(self.__grid[0])
        cell_start = np.zeros((nx, ny), dtype=np.int64) # start of grid cell in flattened array
        cell_end = np.zeros((nx, ny), dtype=np.int64) # end of grid cell in flattened array

        pebble_indices = []
        # converting here
        for i in range(nx):
            for j in range(ny):
                cell_start[i, j] = len(pebble_indices) 
                pebble_indices.extend(self.__grid[i][j])
                cell_end[i, j] = len(pebble_indices)
        pebble_indices = np.asarray(pebble_indices, dtype=np.int64)
        return self.__grid, cell_start, cell_end, pebble_indices
    def get_cellsize(self):
        return self.__cellsize
    def get_width(self):
        return self.__width

# scratch model class - no longer class but used to be
@njit
def Calculate_Scratch_Force(D, pebbles_t, dxs, dys, velocity_angle, mass, mu, xvel, yvel, scratches):
        front, back, angles_f, angles_b, velocity_angle = Front_Or_Back(pebbles_t, dxs, dys, xvel, yvel, velocity_angle)
        force_y = 0 
        force_x = 0
        force = 0

        # add the scratches
        if(len(front) > 0):
            Add_Scratches(angles_f, scratches, front)

        # back interacting with scratches
        s = np.sin(velocity_angle)
        c = np.cos(velocity_angle)
        coefficient = 0
        velocity = np.sqrt(xvel*xvel+yvel*yvel)
        if velocity > 0:
            coefficient = 0.025 * velocity**(-0.5)
        if coefficient > 0.3:
            coefficient = 0.3

        n_scratched = 0
        friction = coefficient * mass * 9.81
        for x in range(len(back)):
            index = back[x]

            if not np.isnan(scratches[index]):
                # apply the force 
                angle = scratches[index]
                kappa = D * np.abs(np.cos(angle))
                force += (kappa * friction)/len(back)
                #n_scratched += 1
        #if (n_scratched > 0):
            #force = force/n_scratched
        force_x = -force*s
        force_y = force*c
        return force_x, force_y


@njit
def Front_Or_Back(pebbles_t, dxs, dys, xvel, yvel, velocity_angle):
        front = np.empty(len(pebbles_t), dtype=np.int64)
        back = np.empty(len(pebbles_t), dtype=np.int64)
        angles_f = np.empty(len(pebbles_t))
        angles_b = np.empty(len(pebbles_t))

        N_f = 0
        N_b = 0
    
        # directional angle of the stone
        theta = np.arctan2(yvel, xvel)
        if(yvel>0):
            theta=-theta+2*np.pi
        else:
            theta=-theta
        velocity_angle = theta
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
                front[N_f] = pebbles_t[i]
                angles_f[N_f] = theta
                N_f += 1
            else:
                # back of stone
                back[N_b] = pebbles_t[i]
                angles_b[N_b] = theta
                N_b += 1
        return (front[:N_f], back[:N_b],angles_f[:N_f], angles_b[:N_b],velocity_angle)

# method to add scratches to the scratch array, each scratch corresponding to a pebble, scratches only front pebbles, back interact with pebbles
@njit
def Add_Scratches(angles, scratches, pebbles_f):
        for i in range(len(pebbles_f)):
            new_angle = (angles[i] + np.pi/2) % (2*np.pi) # tangent to running band (90 degrees on top)
            index = pebbles_f[i] # index of pebble in the pebble arrays
            scratches[index] = new_angle # assigned the angle

# friction imbalance class - no longer class but used to be
@njit
def Calculate_Friction_Imbalance_Force(dt, pebbles_x, pebbles_y, pebbles_temp,pebbles_t, pebbles_r,dxs, dys, touching_stamp, heated_mask, V, w, angles, mass, pebbles_h, n_heated, step, sectors, prad,ice_temp):
        # setting variables 
        N_p = len(pebbles_t) # no.touching pebbles
        t_p = 0
        if(V > 0.09):
            t_p = (((prad))*2)/V
            t_b = w/V
        else:
            t_b = w/0.01

        # increase temperatures of pebbles in contact.
        if (N_p>0):
                n_heated = Pebble_Add_Temperature(pebbles_temp, pebbles_t, V, dt, N_p, touching_stamp, heated_mask, pebbles_h, n_heated, step)
        
        # pebble cooling
        n_heated = Pebble_Cooling(pebbles_temp, pebbles_r, t_b, dt, touching_stamp, heated_mask, pebbles_h, n_heated, step, prad, ice_temp)

        # calculate friction imbalance force
        n: int=len(angles)
        xforce, yforce = calculate_Force_Friction_Imbalance(angles, mass, t_p, t_b, V, pebbles_temp, pebbles_t, dxs, dys, n, sectors)
        return (xforce,yforce), n_heated

@njit
def calculate_Force_Friction_Imbalance(angles, mass, t_p, t_b, V, pebbles_temp, pebbles_t, dxs, dys, n, sectors):
    diff_mu=0
    xforce = 0
    yforce = 0
    x = int(np.ceil(n/2))
     
    if(V > 0.09):
        # constants
        k_i = 2.3
        kappa_i = 1.23e-6
        T = -5.0
        p = (14.7 - 0.6*T) * 1000000
        A = 2/(p*V)
        denominator = np.sqrt(np.pi*kappa_i*t_b)
        x = n // 2

        T = Calculate_Sector_Temperature(n, sectors, pebbles_temp, pebbles_t)

        for i in range (x):
            # difference in coefficients of friction
            T1 = T[i]
            T2 = T[i+x]
            diff_mu = ((A * k_i * (T2 - T1)) / denominator) # difference in mu dependent on temperature difference
            force = diff_mu * mass * 9.81
            xforce += force*np.sin(angles[i])
            yforce += force*np.cos(angles[i])
        yforce = yforce/(x)
        xforce = xforce/(x)
    return xforce, yforce

@njit
def calculate_pebbles_touching_running_band(xpos, ypos, radiusrun, width, pebbles_x, pebbles_y, pebbles_r, touching, dxs, dys, sectors, n, N_r):
        # working with distance/radius squared to avoid square root for efficiency
        radius2 = radiusrun * radiusrun
        radius_inner2 = (radiusrun-width)*(radiusrun-width)
        sector_width = 2*np.pi/n
        count = 0

        for i in range(N_r):
            index = pebbles_r[i]

            dx = pebbles_x[index] - xpos
            dy = pebbles_y[index] - ypos
            distance2 = dx*dx + dy*dy

            if(radius_inner2 <=  distance2 <= radius2): # inside of running band
                touching[count] = index
                dxs[count] = dx
                dys[count] = dy

                # calculate sector for each pebble
                theta = np.arctan2(dx, dy)
                if dy > 0:
                    theta = -theta + 2*np.pi
                else:
                    theta = -theta

                theta = theta % (2*np.pi) # angle

                # sector
                sector = int(np.floor((theta + sector_width/2) / sector_width))
                if sector >= n:
                    sector = n - 1
                sectors[count] = sector
                count += 1 # increase count
        return count 
@njit
def Calculate_Sector_Temperature(n, sectors, pebbles_temp, pebbles_t):
    # calculates average temp of each sector, with sectors of each pebble passed in 
    sector_temp = np.zeros(n)
    sector_count = np.zeros(n)

    for i in range(len(pebbles_t)):
        index = pebbles_t[i]
        sector = sectors[i]

        sector_temp[sector] += pebbles_temp[index]
        sector_count[sector] += 1

    for i in range(n):
        if sector_count[i] > 0:
            sector_temp[i] = sector_temp[i] / sector_count[i]
        else:
            sector_temp[i] = -5
    return sector_temp

@njit
def Pebble_Add_Temperature(pebble_temp, pebbles_t, V, dt, N_p, touching_stamp, heated_mask, heated_pebbles, n_heated, step):
    mu_0 = 0.01
    g = 9.81
    c = 2100
    if N_p == 0:
         return n_heated
    dT = (mu_0*g*V*dt)/(N_p*c)
    for i in range(N_p):
        index = pebbles_t[i]
        pebble_temp[index] += dT
        touching_stamp[index] = step  # Mark this pebble as touching

        if not heated_mask[index]:
            heated_mask[index] = True
            heated_pebbles[n_heated] = index
            n_heated += 1

    return n_heated
@njit
def Pebble_Cooling(pebbles_temp2, pebbles_r, t_b, dt, touching_stamp, heated_mask, pebbles_h, n_heated, step, prad, ice_temp):
    k_i = 2.3
    kappa_i = 1.23*0.000001
    A = (prad)**2*np.pi
    m = 1*0.000001
    c = 2100
    denominator = np.sqrt(np.pi*kappa_i*t_b)

    i = 0

    while i < n_heated:

        index = pebbles_h[i]

        if(touching_stamp[index] != step and heated_mask[index]): # must not be touching and band and must be heated
                # pebble cooling
                delta_T_i = - pebbles_temp2[index]
                dT = -(2*k_i*delta_T_i)/denominator *(A*dt)/(m*c)
                pebbles_temp2[index] += dT

                # Don't allow temperature below ice temperature
                if pebbles_temp2[index] < ice_temp:
                    pebbles_temp2[index] = ice_temp
                    heated_mask[index] = False # pebble no longer heated

                    pebbles_h[i] = pebbles_h[n_heated - 1]
                    n_heated -= 1
                    continue
        i += 1
    return n_heated

@njit
def get_relevant_pebbles(cell_i, cell_j, cell_start, cell_end,pebble_indices, nx, ny, pebbles_r):
    count = 0 # counter for no. of relevant pebbles
    for i in range(cell_i - 1, cell_i + 2): # +2 as does not include final number
        for j in range(cell_j - 1, cell_j + 2): # all adjacent grid cells accessed
            if 0 <= i < nx and 0 <= j < ny: # boundary check
                start = cell_start[i, j] # start and end indices of pebbles in this grid cell
                end = cell_end[i, j]
                for k in range(start, end):
                    index = pebble_indices[k]
                    pebbles_r[count] = index
                    count += 1
    return count

@njit
def update_motion(forces, dt, xvel, yvel, xpos, ypos, angvel, angle, rotation, mu, radius, velocity):
    # does the physics, sets new positions, velocities and angles

    # calculate acceleration
    # deceleration on stone due to friction and other forces
    coefficient=0
    if velocity>0:
        coefficient = 0.025*velocity**(-0.5)
    if coefficient > 0.3:
        coefficient=0.3
    
    anglev = np.arctan2(yvel, xvel)
    
    fx = coefficient*9.81*np.cos(anglev)
    fy = coefficient*9.81*np.sin(anglev)
    fx1 = forces[0]
    fy1 = rotation*forces[1]
    
    accelerationx = -fx1 - fx
    accelerationy = -fy1 - fy

    # slide
    # calculating new velocities
    p_xvel = xvel + accelerationx * dt
    p_yvel = yvel + accelerationy * dt
        
    # X velocity
    if xvel > 0 and p_xvel <= 0:
        xvel = 0
    elif xvel == 0 and p_xvel < 0:
        xvel = 0
    else:
        xvel = p_xvel
    
    # Y velocity
    if yvel != 0 and np.sign(yvel) != np.sign(p_yvel): # shouldnt go from positive to negative to positive
        yvel = 0
    else:
        yvel = p_yvel
    
    # apply angular deceleration
    angular_acceleration = -2 * mu * 9.81 * radius
    angvel = angvel+(angular_acceleration)*dt
    if(angvel<=0):
        angvel=0
    new_x = xpos+xvel*dt
    new_y = ypos+yvel*dt
        
    # calculate (and set) new position and angle
    xpos = new_x
    ypos = new_y
    angle = angle+angvel*dt
    velocity = np.sqrt(xvel*xvel+yvel*yvel)

    return xvel, yvel, xpos, ypos, angvel, angle, velocity

@njit
def Simulation(D, dt, steps, radiusrun, width, pebbles_x, pebbles_y, pebbles_temp, touching, dxs, dys, sectors, touching_stamp, heated_mask, angles, mass, pebbles_h, xpos, ypos, xvel, yvel, velocity, angvel, angle, n_heated, pebbles_r, cell_start, cell_end, pebble_indices, nx, ny, rotation, mu, scratches, prad, ice_temp, radius):
    # simulation loop
    V = velocity

    # get relevant pebbles
    cell_i = int(np.floor(xpos / 0.07))
    cell_j = int(np.floor((ypos + 20/2) / 0.07))
    N_r = get_relevant_pebbles(cell_i, cell_j, cell_start, cell_end, pebble_indices, nx, ny, pebbles_r)
         
    for step in range(steps):
            # find pebbles touching running band
            N_p = calculate_pebbles_touching_running_band(xpos,ypos,radiusrun,width,pebbles_x,pebbles_y, pebbles_r,touching, dxs, dys, sectors, 30, N_r)

            # calculate friction imbalance force
            forces1, n_heated = Calculate_Friction_Imbalance_Force(dt, pebbles_x, pebbles_y, pebbles_temp, touching[:N_p], pebbles_r, dxs, dys, touching_stamp, heated_mask, velocity, width, angles, mass, pebbles_h, n_heated, step, sectors, prad, ice_temp)
            forces2 = Calculate_Scratch_Force(D, touching[:N_p], dxs, dys,0.0, mass, mu, xvel, yvel, scratches)
            forces = [forces1[0] + forces2[0], forces1[1] + forces2[1]]
            
            # update stone motion
            xvel, yvel, xpos, ypos, angvel, angle, velocity = update_motion(forces, dt,xvel, yvel, xpos, ypos, angvel, angle, rotation, mu, radius, velocity)
            if abs(xvel) < 1e-6 and abs(yvel) < 1e-6:
                break
            #if step % 10000 == 0:
                #print("step =", step)

            new_cell_i = int(np.floor(xpos / 0.07))
            new_cell_j = int(np.floor((ypos + 20/2) / 0.07))

            if new_cell_i != cell_i or new_cell_j != cell_j: # check if have changed grid cell, if so, change relevant pebbles
                cell_i = new_cell_i
                cell_j = new_cell_j

                N_r = get_relevant_pebbles(cell_i, cell_j,cell_start, cell_end, pebble_indices,nx, ny,pebbles_r)

    return xpos, ypos

def main(D, mass, Vel, w, radiusb, density, prad, ice_temp, angularv):
    if angularv == 0:
        D = 0
    x=0 # initial x position stone
    y=0 # initial y position stone
    t=0 # initial time
    length = 45 # length of ice sheet in m
    width = 20 # width of icesheet in m
    n_pebbles = int(round((length*width)/(0.0075)*density)) # no. pebbles to generate

    # create ice sheet + grid
    pebbles_x = np.random.uniform(0, length, n_pebbles) # array of pebble x positions
    pebbles_y = np.random.uniform(-width/2, width/2, n_pebbles) # array of pebble y positions
    icesheet = Ice_Sheet(width, length, -5, [], [], 0.07)
    grid, cell_start, cell_end, pebble_indices = icesheet.create_pebble_grid(pebbles_x, pebbles_y) # grid, start and end of cells, pebble indices, flattened array
    nx = cell_start.shape[0] # number of cells in x
    ny = cell_start.shape[1] # number of cells in y
    # create stone
    stone = Stone(Vel, 0, angularv, mass, 0.145, 0.0635,0, 0, 0, 0.01, -1, False, 0, [-2, 0], icesheet, w)

    # OOP - get parameters
    width = w
    radiusrun = radiusb
    radius = 0.145
    xpos = stone.get_xpos()
    ypos = stone.get_ypos()
    xvel = Vel
    yvel = stone.get_yvel()
    velocity = np.sqrt(xvel*xvel+yvel*yvel)
    angvel = angularv
    angle = stone.get_angle()
    v = np.sqrt(xvel*xvel+yvel*yvel)
    rotation = stone.get_rotation()
    #mu = stone.get_coefficient()
    mu = 0.01
    angles = stone.divide_circumference(30)

    # arrays for replacing OOP with efficiency
    pebbles_temp = np.full(n_pebbles, ice_temp, dtype = np.float64) # array of pebble temperatures - initially -5 degrees C and heated through contact with stone
    touching_stamp = np.zeros(n_pebbles, dtype=np.int32) # mask to show which pebbles are touching, true if pebble touching
    heated_mask = np.zeros(n_pebbles, dtype=np.bool_) # mask to show which pebbles are heated, true if pebble heated
    pebbles_h = np.empty(n_pebbles, dtype=np.int64) # array of heated pebbles
    n_heated = 0 # number of pebbles that are heated
    touching = np.empty(n_pebbles, dtype=np.int64) # touching pebbles
    dxs = np.empty(n_pebbles, dtype = np.float64) # distance from stone centre to pebble in x direction
    dys = np.empty(n_pebbles, dtype = np.float64) # distance from stone centre to pebble in y direction
    sectors = np.empty(n_pebbles, dtype=np.int64) # storing sector pebble is in
    pebbles_r=np.empty(n_pebbles,dtype=np.int64) # pebbles in relevant grid cells
    scratches = np.full(n_pebbles, np.nan) # scratch array

    # timings so using correct no. of steps
    dt = 0.000167/2
    total_time = 200
    steps = int(total_time / dt)

    # simulate - calls force methods, calculates acceleration, updates position, direction, velocity... then returns final position of stone
    xpos, ypos = Simulation(D, dt, steps, radiusrun, width,
    pebbles_x, pebbles_y, pebbles_temp,touching, dxs, dys, sectors,
    touching_stamp, heated_mask,angles, mass, pebbles_h,
    xpos, ypos, xvel, yvel, velocity,angvel, angle, n_heated,
    pebbles_r, cell_start, cell_end,pebble_indices, nx, ny,rotation, mu, scratches,prad, ice_temp, radius)

    return xpos, ypos


import time
import csv

# parameter studies
def parameter_studies():

    # constant A
    times = []
    A_values = np.arange(0.00, 0.251, 0.01)

    # CSV file
    with open("2_constant_A.csv", "w", newline="") as file:

        writer = csv.writer(file)

        # column headings
        writer.writerow(["Constant A", "Run", "Final X", "Final Y", "Runtime (s)"])

        # Numba warm-up
        print("Numba warm-up...")
        main(D=0.21, mass = 17.24, Vel=3, w=0.005, radiusb=0.0635, density=1, prad=0.00065, ice_temp=-5, angularv=1)

        for A in A_values:
            print("A =", A)

            for run in range(50):
                start = time.perf_counter()
                xpos, ypos = main(D=A, mass=19, Vel=3, w=0.005, radiusb=0.0635, density=1, prad=0.00065, ice_temp=-5, angularv = 1)
                runtime = time.perf_counter() - start
                times.append(runtime)
                # save result immediately
                writer.writerow([A,run + 1,xpos,ypos,runtime])

    # Overall average runtime
    average_time = np.mean(times)

    print("Parameter study on A complete!")
    print("Total simulations:", len(times))
    print("Average runtime:", average_time, "s")
    print("Total runtime:", np.sum(times), "s")


    # parameter study mass
    times = []
    m_values = np.arange(17.24, 19.965, 0.247)
    
    # CSV file
    with open("2_mass.csv", "w", newline="") as file:
    
            writer = csv.writer(file)
    
            # column headings
            writer.writerow(["Mass", "Run", "Final X", "Final Y", "Runtime (s)"])
    
            # Numba warm-up
            print("Numba warm-up...")
            main(D=0.21, mass = 17.24, Vel=3, w=0.005, radiusb=0.0635, density=1, prad=0.00065, ice_temp=-5, angularv=1)
    
            for m in m_values:
                print("Mass =", m)
    
                for run in range(50):
                    start = time.perf_counter()
                    xpos, ypos = main(D=0.15, mass=m, Vel=3, w=0.005, radiusb=0.0635, density=1, prad=0.00065, ice_temp=-5, angularv = 1)
                    runtime = time.perf_counter() - start
                    times.append(runtime)
                    # save result immediately
                    writer.writerow([m,run + 1,xpos,ypos,runtime])
    
    # Overall average runtime
    average_time = np.mean(times)
    
    print("Parameter study on mass complete!")
    print("Total simulations:", len(times))
    print("Average runtime:", average_time, "s")
    print("Total runtime:", np.sum(times), "s")


    # parameter study velocity
    times = []
    v_values = np.arange(2, 6, 0.2)
    
    # CSV file
    with open("2_velocity.csv", "w", newline="") as file:
    
            writer = csv.writer(file)
    
            # column headings
            writer.writerow(["Velocity", "Run", "Final X", "Final Y", "Runtime (s)"])
    
            # Numba warm-up
            print("Numba warm-up...")
            main(D=0.21, mass = 17.24, Vel=3, w=0.005, radiusb=0.0635, density=1, prad=0.00065, ice_temp=-5, angularv=1)
    
            for v in v_values:
                print("Velocity =", v)
    
                for run in range(50):
                    start = time.perf_counter()
                    xpos, ypos = main(D=0.15, mass=19, Vel=v, w=0.005, radiusb=0.0635, density=1, prad=0.00065, ice_temp=-5, angularv = 1)
                    runtime = time.perf_counter() - start
                    times.append(runtime)
                    # save result immediately
                    writer.writerow([v,run + 1,xpos,ypos,runtime])
    
    # Overall average runtime
    average_time = np.mean(times)
    
    print("Parameter study on velocity complete!")
    print("Total simulations:", len(times))
    print("Average runtime:", average_time, "s")
    print("Total runtime:", np.sum(times), "s")

    # parameter study running band width
    times = []
    w_values = np.arange(0.001, 0.011, 0.001)
        
    # CSV file
    with open("2_rband_width.csv", "w", newline="") as file:
        
                writer = csv.writer(file)
        
                # column headings
                writer.writerow(["Running Band Width", "Run", "Final X", "Final Y", "Runtime (s)"])
        
                # Numba warm-up
                print("Numba warm-up...")
                main(D=0.21, mass = 17.24, Vel=3, w=0.005, radiusb=0.0635, density=1, prad=0.00065, ice_temp=-5, angularv=1)
        
                for w in w_values:
                    print("Width =", w)
        
                    for run in range(50):
                        start = time.perf_counter()
                        xpos, ypos = main(D=0.15, mass=19, Vel=3, w=w, radiusb=0.0635, density=1, prad=0.00065, ice_temp=-5, angularv = 1)
                        runtime = time.perf_counter() - start
                        times.append(runtime)
                        # save result immediately
                        writer.writerow([w,run + 1,xpos,ypos,runtime])
        
    # Overall average runtime
    average_time = np.mean(times)
    print("Parameter study on Band Width complete!")
    print("Total simulations:", len(times))
    print("Average runtime:", average_time, "s")
    print("Total runtime:", np.sum(times), "s")

    # parameter study radius running band
    times = []
    r_values = np.arange(0.05, 0.08, 0.0025)
    
    # CSV file
    with open("2_radius_band.csv", "w", newline="") as file:
    
            writer = csv.writer(file)
    
            # column headings
            writer.writerow(["Radius Running Band", "Run", "Final X", "Final Y", "Runtime (s)"])
    
            # Numba warm-up
            print("Numba warm-up...")
            main(D=0.21, mass = 17.24, Vel=3, w=0.005, radiusb=0.0635, density=1, prad=0.00065, ice_temp=-5, angularv=1)
    
            for r in r_values:
                print("Radius =", r)
    
                for run in range(50):
                    start = time.perf_counter()
                    xpos, ypos = main(D=0.15, mass=19, Vel=3, w=0.005, radiusb=r, density=1, prad=0.00065, ice_temp=-5, angularv = 1)
                    runtime = time.perf_counter() - start
                    times.append(runtime)
                    # save result immediately
                    writer.writerow([r,run + 1,xpos,ypos,runtime])
    
    # Overall average runtime
    average_time = np.mean(times)
    print("Parameter study on radius band complete!")
    print("Total simulations:", len(times))
    print("Average runtime:", average_time, "s")
    print("Total runtime:", np.sum(times), "s")



    # parameter study pebble density factor
    times = []
    f_values = np.arange(1, 8.1, 0.5)
    
    # CSV file
    with open("2_density_factor.csv", "w", newline="") as file:
    
            writer = csv.writer(file)
    
            # column headings
            writer.writerow(["Pebble Density Factor", "Run", "Final X", "Final Y", "Runtime (s)"])
    
            # Numba warm-up
            print("Numba warm-up...")
            main(D=0.21, mass = 17.24, Vel=3, w=0.005, radiusb=0.0635, density=1, prad=0.00065, ice_temp=-5, angularv=1)
    
            for f in f_values:
                print("Pebble Density Factor =", f)
    
                for run in range(50):
                    start = time.perf_counter()
                    xpos, ypos = main(D=0.15, mass=19, Vel=3, w=0.005, radiusb=0.0635, density=f, prad=0.00065, ice_temp=-5, angularv = 1)
                    runtime = time.perf_counter() - start
                    times.append(runtime)
                    # save result immediately
                    writer.writerow([f,run + 1,xpos,ypos,runtime])
    
    # Overall average runtime
    average_time = np.mean(times)
    print("Parameter study on mass complete!")
    print("Total simulations:", len(times))
    print("Average runtime:", average_time, "s")
    print("Total runtime:", np.sum(times), "s")



    # parameter study average pebble radius
    times = []
    p_values = np.arange(0.000325, 0.0039, 0.000325)
    
    # CSV file
    with open("2_pebble_radius.csv", "w", newline="") as file:
    
            writer = csv.writer(file)
            # column headings
            writer.writerow(["Average Pebble Radius", "Run", "Final X", "Final Y", "Runtime (s)"])
    
            # Numba warm-up
            print("Numba warm-up...")
            main(D=0.21, mass = 17.24, Vel=3, w=0.005, radiusb=0.0635, density=1, prad=0.00065, ice_temp=-5, angularv=1)
    
            for p in p_values:
                print("Pebble Radius =", p)
    
                for run in range(50):
                    start = time.perf_counter()
                    xpos, ypos = main(D=0.15, mass=19, Vel=3, w=0.005, radiusb=0.0635, density=1, prad=p, ice_temp=-5, angularv = 1)
                    runtime = time.perf_counter() - start
                    times.append(runtime)
                    # save result immediately
                    writer.writerow([p,run + 1,xpos,ypos,runtime])
    
    # Overall average runtime
    average_time = np.mean(times)
    print("Parameter study on pebble radius complete!")
    print("Total simulations:", len(times))
    print("Average runtime:", average_time, "s")
    print("Total runtime:", np.sum(times), "s")



    # parameter study ice temperature
    times = []
    t_values = np.arange(-7, -1.49, 0.5)
    
    # CSV file
    with open("2_ice_temp.csv", "w", newline="") as file:
    
            writer = csv.writer(file)
    
            # column headings
            writer.writerow(["Ice Temperature", "Run", "Final X", "Final Y", "Runtime (s)"])
    
            # Numba warm-up
            print("Numba warm-up...")
            main(D=0.21, mass = 17.24, Vel=3, w=0.005, radiusb=0.0635, density=1, prad=0.00065, ice_temp=-5, angularv=1)
    
            for t in t_values:
                print("Temperature Ice =", t)
    
                for run in range(50):
                    start = time.perf_counter()
                    xpos, ypos = main(D=0.15, mass=19, Vel=3, w=0.005, radiusb=0.0635, density=1, prad=0.00065, ice_temp=t, angularv = 1)
                    runtime = time.perf_counter() - start
                    times.append(runtime)
                    # save result immediately
                    writer.writerow([t,run + 1,xpos,ypos,runtime])
    
    # Overall average runtime
    average_time = np.mean(times)
    print("Parameter study on mass complete!")
    print("Total simulations:", len(times))
    print("Average runtime:", average_time, "s")
    print("Total runtime:", np.sum(times), "s")



    # parameter study angular velocity
    times = []
    av_values = np.arange(0.0, 7.1, 0.5)
    
    # CSV file
    with open("2_angular_velocity.csv", "w", newline="") as file:
    
            writer = csv.writer(file)
    
            # column headings
            writer.writerow(["Angular Velocity", "Run", "Final X", "Final Y", "Runtime (s)"])
    
            # Numba warm-up
            print("Numba warm-up...")
            main(D=0.21, mass = 17.24, Vel=3, w=0.005, radiusb=0.0635, density=1, prad=0.00065, ice_temp=-5, angularv=1)
    
            for av in av_values:
                print("Angular Velocity =", av)
    
                for run in range(50):
                    start = time.perf_counter()
                    xpos, ypos = main(D=0.15, mass=19, Vel=3, w=0.005, radiusb=0.0635, density=1, prad=0.00065, ice_temp=-5, angularv = av)
                    runtime = time.perf_counter() - start
                    times.append(runtime)
                    # save result immediately
                    writer.writerow([av,run + 1,xpos,ypos,runtime])
    
    # Overall average runtime
    average_time = np.mean(times)
    print("Parameter study on angular velocity complete!")
    print("Total simulations:", len(times))
    print("Average runtime:", average_time, "s")
    print("Total runtime:", np.sum(times), "s")

parameter_studies()