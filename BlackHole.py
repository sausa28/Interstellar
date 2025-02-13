'''
All functions for creating an image of a black hole.
It uses the following as variables in sets of arrays:
vars1d = [r_cam, r_sphere, num, M, L, k, phi_max]
vars2d = [map1d, resolution, F]  map1d is created using vars1d, resolution is a 2 element array, F is the FOV in radians.
varsImage = [map2d, CelestialSphere, resolution]  map2d is created using vars2d, CelestialSphere is a keyword used to identify the image file.  Resolution is the same as before.
varsRotate = [map2d, a, b, c] where a, b, c are Euler angles to rotate the celestial sphere by.
varsGif = [a_vals, b_vals, c_vals] where a_vals, b_vals, c_vals are length num arrays of the angles to run through.

1D Map files contain 2 arrays:  'map1d' and 'vars1d'
2D Map files contain 3 arrays:  'map2d', 'vars1d' and 'vars2d'
'''

import pickle
import os
import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
import matplotlib as mpl

# Core Functions #


def f(t, y, M, L):
    '''This is the right hand side of the set of ODEs, y is a shape (3,) array containing r,s,phi.'''
    drdt = y[1]
    dsdt = - L**2 * (3 * M - y[0]) / y[0]**4
    dphidt = L / y[0]**2

    return [drdt, dsdt, dphidt]


def ray_trace(vars1d, phi_cam):
    '''Takes the angle of the camera (along with other values) and outputs the angle on the celestial sphere.'''
    r_cam, r_sphere, num, M, L, k, phi_max = vars1d  # unpack all the variables

    # Initial Values
    R_0 = r_cam
    S_0 = L / (R_0 * np.tan(phi_cam))
    Phi_0 = np.pi

    if phi_cam > 0:  # determines whether to go backwards or forwards
        k = -k

    y0, t0 = [R_0, S_0, Phi_0], 0

    def evaluator(t, y):
        '''Looks at a set of points on the path, and decides whether to stop the integration.'''
        if y[0] < 0.001:  # stop when close enough to the black hole
            return -1
        else:
            return 0

    r = ode(f).set_integrator('dopri5')
    r.set_solout(evaluator)
    r.set_initial_value(y0, t0).set_f_params(M, L)

    endPoint = r.integrate(k * 1000.)

    # angle = np.arctan2(L, endPoint[0]*endPoint[1])
    if endPoint[0] < 0.001:
        angle = np.nan
    else:  # find the output angle.  See find_angle.
        justBeforeEndPoint = r.integrate(k * 950.)

        p1_polar = (justBeforeEndPoint[0], justBeforeEndPoint[2])
        p2_polar = (endPoint[0], endPoint[2])

        p1_cart = (p1_polar[0] * np.cos(p1_polar[1]), p1_polar[0] * np.sin(p1_polar[1]))
        p2_cart = (p2_polar[0] * np.cos(p2_polar[1]), p2_polar[0] * np.sin(p2_polar[1]))

        y_diff = p2_cart[1] - p1_cart[1]
        x_diff = p2_cart[0] - p1_cart[0]

        angle = np.arctan2(y_diff, x_diff)

    # print phi_cam, ' --> ', angle
    return angle


def create_1dmap(vars1d):
    '''
    Creates a 2-column array with the input angle phi_cam and the output angle on the celestial sphere and saves it.
    Returns (map, vars1d)
    '''
    num, phi_max = vars1d[2], vars1d[6]
    print(f"num = {num}\nphi_max = {phi_max}")
    input_phis = np.linspace(-phi_max, phi_max, num, endpoint=False)

    map = np.zeros((len(input_phis), 2))
    map[:, 0] = input_phis

    for i in range(len(input_phis)):
        print("Creating 1D Map: %.2f" % ((i + 1) / float(num) * 100), end="")  # progress report
        print("\r", end="")
        phi_cam = input_phis[i]
        map[i, 1] = ray_trace(vars1d, phi_cam)

        # name = "n=%i r1=%.1f r2=%.1f M=%.2f L=%.1f k=%.3f" % (num, vars1d[0], vars1d[1], vars1d[3], vars1d[4], vars1d[5])
        # np.savez_compressed("1D Maps/1dmap " + name, vars1d=vars1d, map1d=map)

    print("")  # Move cursor down to next line
    print(map)
    return map, vars1d


def make2dmap(vars2d, vars1d):
    '''
    Creates an array, the same size as the final image, containing angles on the celestial sphere for each pixel and saves it.
    Returns (map2d, vars2d)
    '''
    map1d, resolution, F = vars2d
    FOV = int(F * 180 / np.pi)
    r_max = np.amax(resolution) // 2

    print("Creating 2d Map")

    # Map x,y coordinates to theta, phi on the screen.

    print("Calculating Screen Angles...")

    xv = -np.arange(resolution[1]) + resolution[1] // 2  # recentre origin to use as coodinates
    yv = -np.arange(resolution[0]) + resolution[0] // 2

    x, y = np.meshgrid(xv, yv)

    # sign_y = np.sign(y)
    # sign_y[resolution[0]/2] = np.sign(x)[resolution[0]/2] #sets the row at y=0 to be equal to sign(x) along that row, since we don't want it to be equal to 0.

    # theta_cam = np.mod(np.arctan2(y, x), np.pi)
    # phi_cam = np.arctan((sign_y*np.sqrt(x**2 + y**2)) * np.tan(F/2) / r_max)

    epsilon = np.sign(y)
    epsilon[resolution[0] // 2] = np.sign(x)[resolution[0] // 2]  # sets the row at y=0 to be equal to sign(x) along that row, since we don't want it to be equal to 0.

    theta_cam = np.mod(np.arctan2(y, x), np.pi)
    phi_cam = epsilon * np.arctan((np.sqrt(x**2 + y**2)) * np.tan(F / 2) / r_max)

    # Map screen angles to polar coodinates on the celestial sphere.

    print("Calculating points on celestial sphere...")

    theta = theta_cam  # find angles in our coordinate system
    phi = np.interp(phi_cam, map1d[:, 0], map1d[:, 1])

    sphere_x, sphere_y, sphere_z = np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta) * np.sin(phi)  # takes the angles in our coordinate system to cartesian
    sphere_theta, sphere_phi = np.expand_dims(np.arccos(sphere_z), axis=2), np.expand_dims(np.arctan2(sphere_y, sphere_x), axis=2)  # transforms into normal spherical polars

    map2d = np.concatenate((sphere_theta, sphere_phi), axis=2)

    name = "2D Maps/2dmap res=" + str(resolution) + " FOV=%.i M=%.2f r=%.1f.pickle" % (FOV, vars1d[3], vars1d[0])
    data = {"vars1d": vars1d, "vars2d": vars2d, "map2d": map2d}
    with open(name, 'wb') as f:
        pickle.dump(data, f)

    return map2d, vars2d


def make_image(varsImage, vars2d, vars1d, gif=False, image_no=1):
    '''Creates and saves the final image as an array with colour values based on the celestial sphere chosen.'''
    map2d, CelestialSphere, resolution = varsImage

    print("Creating Image " + str(image_no) + "...")

    map2d = np.nan_to_num(map2d)

    # normalise the coordinate arrays to the size of the celestial sphere.
    yCoords = (map2d[:, :, 0] / np.pi * (np.shape(CelestialSphere)[0] - 1)).astype(int)
    xCoords = (map2d[:, :, 1] / (2 * np.pi) * (np.shape(CelestialSphere)[1] - 1)).astype(int)

    isBlackHole = np.logical_and(np.equal(xCoords, 0), np.equal(yCoords, 0))  # Boolean array specifying where the black hole is.
    black = np.zeros((resolution[0], resolution[1], 3))
    if np.shape(CelestialSphere)[2] == 4:  # Take into account whether the celestial sphere is RGB or RGBA
        black = np.concatenate((black, np.ones((resolution[0], resolution[1]))[..., np.newaxis]), axis=2)

    final_image = np.where(isBlackHole[..., np.newaxis], black, CelestialSphere[yCoords, xCoords])

    FOV = int(vars2d[2] * 180 / np.pi)
    name = "%.ix%.i FOV=%.i M=%.2f r=%.1f " % (vars2d[1][1], vars2d[1][0], FOV, vars1d[3], vars1d[0]) + CelestialSphereName
    if not gif:
        plt.imsave("Images/" + name + ".png", final_image)  # saves image in "Images" folder
    elif gif:
        plt.imsave("Images/gif/" + name + str(image_no) + ".png", final_image)

    return final_image

# Extra Functions #


def rotation_matrix(a, b, c):
    '''Creates a 3x3 rotation matrix for the angles a, b, c in that order.'''

    A = np.array([[1, 0, 0],
                  [0, np.cos(a), -np.sin(a)],
                  [0, np.sin(a), np.cos(a)]])

    B = np.array([[np.cos(b), 0, np.sin(b)],
                  [0, 1, 0],
                  [-np.sin(b), 0, np.cos(b)]])

    C = np.array([[np.cos(c), -np.sin(c), 0],
                  [np.sin(c), np.cos(c), 0],
                  [0, 0, 1]])

    R = np.dot(C, np.dot(B, A))

    return R


def rotate_map(varsRotate, image_no=1):
    '''Takes a 2d map and rotates all the coodinates by the given angles, returns the map with angles used.'''

    # print "Rotating Image " + str(image_no) + "..."

    map2d, a, b, c = varsRotate

    R = rotation_matrix(a, b, c)

    thetas = np.expand_dims(map2d[:, :, 0], axis=2)
    phis = np.expand_dims(map2d[:, :, 1], axis=2)

    carts = np.concatenate((np.sin(thetas) * np.cos(phis), np.sin(thetas) * np.sin(phis), np.cos(thetas)), axis=2)
    new_carts = np.einsum('kl,ijl', R, carts)  # apply rotation to all sets of cartesians, using einstein summation.

    new_thetas = np.expand_dims(np.arccos(new_carts[:, :, 2]), axis=2)
    new_phis = np.expand_dims(np.arctan2(new_carts[:, :, 1], new_carts[:, :, 0]), axis=2)

    map2dRotated = np.concatenate((new_thetas, new_phis), axis=2)

    return map2dRotated, varsRotate


def checkMapsExist(vars1d_desired, vars2d_desired):
    '''
    Compares the desired variables with maps that have already been created.
    Returns: (map1DExists, map2DExists, file1D, file2D)
    where map1DExists and map2DExists are bools, file1D and file2D are file names if they exist.
    file1D or file2D = "None" if they don't exist.
    '''

    map1DExists, map2DExists, file1D, file2D = False, False, 'None', 'None'

    for file in os.listdir("1D Maps"):  # compare all existing 1D maps to desired map
        vars1d_test = np.load("1D Maps/" + file)["vars1d"]
        if np.array_equal(vars1d_desired, vars1d_test):
            map1DExists = True
            file1D = "1D Maps/" + file  # get file path
            break
        else:
            map1DExists = False
            file1D = 'None'

        for file in os.listdir("2D Maps"):  # compare existing 2D maps with desired.
            with open("2D Maps/" + file, 'r') as f:
                mapfiletest = pickle.load(f)

            vars1d_test, vars2d_test = mapfiletest["vars1d"], mapfiletest["vars2d"]

            if np.array_equal(vars1d_desired, vars1d_test) and np.array_equal(vars2d_desired[0], vars2d_test[1]) and np.isclose(vars2d_desired[1], vars2d_test[2]):
                # we want to just compare the resolution and fov.
                map2DExists = True
                file2D = "2D Maps/" + file  # get file path
                break
            else:
                map2DExists = False
                file2D = 'None'

    return map1DExists, map2DExists, file1D, file2D


def loadCreateMaps(vars1d_desired, vars2d_desired):
    '''
    Loads map if they exist, otherwise creates them.
    Note that the inputted vars2d_desired should not contain the 1d map we want, this will be found.
    '''

    print("Checking for existing maps...")

    existance = checkMapsExist(vars1d_desired, vars2d_desired)

    if existance[0] is True and existance[1] is True:  # both already exist
        with open(existance[3], "rb") as f2:
            file1D, file2D = np.load(existance[2]), pickle.load(f2)

        map1d, vars1d = file1D["map1d"], file1D["vars1d"]
        map2d, vars2d = file2D["map2d"], file2D["vars2d"]

    elif existance[0] is True and existance[1] is False:  # only 1d map exists, need to create 2d map
        file1D = np.load(existance[2])
        map1d, vars1d = file1D["map1d"], file1D["vars1d"]

        vars2d_desired_full = [map1d, vars2d_desired[0], vars2d_desired[1]]
        map2d, vars2d = make2dmap(vars2d_desired_full, vars1d)

    elif existance[0] is False:  # 1d map does not exist, need to create both
        map1d, vars1d = create_1dmap(vars1d_desired)

        vars2d_desired_full = [map1d, vars2d_desired[0], vars2d_desired[1]]
        map2d, vars2d = make2dmap(vars2d_desired_full, vars1d)

    return map1d, map2d, vars1d, vars2d


def make_many_images(varsImage, vars2d, vars1d, varsGif):
    a_vals, b_vals, c_vals = varsGif
    map2d = varsImage[0]
    varsImage_temp = np.copy(varsImage)

    for i in range(len(a_vals)):
        varsRotate = [map2d, a_vals[i], b_vals[i], c_vals[i]]
        varsImage_temp[0] = rotate_map(varsRotate, image_no=i + 1)[0]

        make_image(varsImage_temp, vars2d, vars1d, gif=True, image_no=i + 1)


def plot_multiple_paths(vars1d):

    phi_max, num = vars1d[6], vars1d[2]

    input_phis = np.linspace(-phi_max, phi_max, num, endpoint=True)

    norm = mpl.colors.Normalize(0, 2 * np.pi)  # or norm = mpl.colors.Normalize(-phi_max, phi_max)
    color_map = plt.get_cmap('hsv')
    scalar_map = mpl.cm.ScalarMappable(norm, color_map)

    zorders = np.zeros_like(input_phis)
    for i in range(len(input_phis) / 2 + 1):  # code to choose the order in which all the light paths are plotted
        zorders[i], zorders[-i - 1] = len(input_phis) / 2 - i, len(input_phis) / 2 - i

    ax1 = plt.subplot(121, polar=True)
    ax2 = plt.subplot(122)

    vis_colors = np.zeros((len(input_phis), 4))  # initialize colors array

    for i in range(len(input_phis)):
        phi_cam = input_phis[i]  # vars[2] = phi_cam
        path, HitsBlackHole = ray_trace(vars1d, phi_cam)

        if HitsBlackHole is True:  # if a light ray ends in the black hole, color it black, and set it to be on top
            vis_colors[i] = [0.0, 0.0, 0.0, 1.0]
            zorders[i] = len(input_phis)
        else:
            angle = find_angle(path)
            vis_colors[i] = scalar_map.to_rgba(angle)  # color a light ray based on where it 'came from'

        ax1.plot(path[:, 2], path[:, 0], color=vis_colors[i], zorder=zorders[i], linewidth=0.5)
        print(i + 1) / float(num) * 100  # progress report

    ax1.set_ylim(0, r_sphere * 1.1)
    ax1.set_yticks([])
    print('Now plotting vertical plot...')
    input_phis = np.reshape(input_phis, (len(input_phis), 1))  # reshape input_phis to work in plt.eventplot
    ax2.eventplot(input_phis, orientation='vertical', colors=vis_colors, lineoffsets=0)
    plt.show()


def find_angle(path):
    '''Takes a light ray path and calculates the angle at infinity.'''
    if path[-1, 0] < 0:
        angle = np.nan  # identify black hole with NaN
    else:
        p1_index = -(len(path) / 50 + 2)

        p1_polar = (path[p1_index, 0], path[p1_index, 2])
        p2_polar = (path[-1, 0], path[-1, 2])

        p1_cart = (p1_polar[0] * np.cos(p1_polar[1]), p1_polar[0] * np.sin(p1_polar[1]))
        p2_cart = (p2_polar[0] * np.cos(p2_polar[1]), p2_polar[0] * np.sin(p2_polar[1]))

        y_diff = p2_cart[1] - p1_cart[1]
        x_diff = p2_cart[0] - p1_cart[0]

        angle = np.arctan2(y_diff, x_diff)

    return np.mod(angle, 2 * np.pi)


def screen_angles(x_, y_, vars2d):
    '''
    Maps x,y on the screen to theta, phi on the screen
    Returns theta_cam, phi_cam
    '''

    resolution, F = vars2d[1:3]
    r_max = np.amax(resolution) / 2

    x, y = x_ - resolution[1] / 2, y_ - resolution[0] / 2  # recentre origin to use as coordinates

    if y == 0:  # don't want np.sign to give 0
        sign_y = np.sign(x)
    else:
        sign_y = np.sign(y)

        theta_cam = np.mod(np.arctan2(y, x), np.pi)
        phi_cam = np.arctan((sign_y * np.sqrt(x**2 + y**2)) / r_max * np.tan(F / 2))

        return theta_cam, phi_cam


def map_to_sphere(theta_cam, phi_cam, vars2d):
    '''Maps screen angles to polar coodinates on the sphere'''
    plane_map = vars2d[0]

    theta = theta_cam  # find angles in our coordinate system
    phi = np.interp(phi_cam, plane_map[:, 0], plane_map[:, 1])

    if np.isnan(phi):  # if it hits the black hole, identify with NaN, NaN
        return np.nan, np.nan
    else:
        carts = np.array([np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta) * np.sin(phi)])  # takes the angles in our coordinate system to cartesian
        new_polars = (np.arccos(carts[2]), np.arctan2(carts[1], carts[0]))  # transforms into normal spherical polars

        return new_polars[0], new_polars[1]


def get_cel_sphere_color(theta_proper, phi_proper, varsImage, CelestialSphere):
    '''Takes an angle on the celestial sphere and returns the colour'''
    # CelestialSphereName = varsImage[1]
    # CelestialSphere = plt.imread("Celestial Spheres/"+CelestialSphereName+".png")

    if np.isnan(theta_proper) and np.isnan(phi_proper):  # if it is a black hole point, return black
        if np.shape(CelestialSphere)[2] == 3:  # if in RGB
            return [0, 0, 0]
        else:  # if in RGBA
            return [0, 0, 0, 1]

    else:
        a = int(theta_proper / np.pi * np.shape(CelestialSphere)[0]) % np.shape(CelestialSphere)[0]
        b = int(phi_proper / (2 * np.pi) * np.shape(CelestialSphere)[1]) % np.shape(CelestialSphere)[1]

        return CelestialSphere[a, b]


if __name__ == "__main__":
    # Desired Variables #

    # vars1d [r_cam, r_sphere, num, M, L, k, phi_max]
    r_cam = 5.0
    r_sphere = 10.0
    num = 200
    M = 0.5
    L = 1
    k = 1
    phi_max = np.pi

    # vars2d [map1d, resolution, F]
    resolution = np.array([200, 200])
    F = 100 * np.pi / 180

    # varsImage [map2d, CelestialSphere, resolution]
    CelestialSphereName = "colour"
    CelestialSphere = plt.imread("Celestial Spheres/" + CelestialSphereName + ".png")

    # varsRotate [map2d, a, b, c]
    a = 0
    b = 0
    c = 0

    angleNum = 500

    a_vals = np.zeros(angleNum)
    # a_vals.fill(np.pi/4)
    b_vals = np.linspace(0, 2 * np.pi, angleNum, endpoint=False)
    c_vals = np.zeros(angleNum)

    vars1d_desired = [r_cam, r_sphere, num, M, L, k, phi_max]
    vars2d_desired = [resolution, F]

    # vars1d = np.array([r_cam, r_sphere, num, M, L, k, phi_max])
    # map1d = create_1dmap(vars1d)[0]
    # vars2d = np.array([map1d, resolution, F])

    map2d, vars1d, vars2d = loadCreateMaps(vars1d_desired, vars2d_desired)[1:]

    varsRotate = [map2d, a, b, c]

    # varsImage = np.array([map2d, CelestialSphere, resolution])
    varsImage = [rotate_map(varsRotate)[0], CelestialSphere, resolution]
    # varsGif = np.array([a_vals, b_vals, c_vals])

    # make_many_images(varsImage, vars2d, vars1d, varsGif)

    image = make_image(varsImage, vars2d, vars1d, image_no=1)
    # image = imresize(image, 25)
    plt.imshow(image)
    plt.show()
