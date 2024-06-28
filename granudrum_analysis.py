import numpy as np
import os
import fnmatch
from random import randint
import cv2
import imutils
import warnings
import plotly.graph_objects as go
import seaborn as sns
from PIL import Image
from collections import namedtuple
import math
# import pandas as pd


def find_nearest(array, value):
    """Returns the index of the array element that is closest to the specified value"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_first(item, vec):
    """return the index of the first occurrence of item in vec"""
    for i in range(len(vec)):
        if item == vec[i]:
            return i
    return np.nan


def find_distance(points, centre):
    """
    Returns array of distances between each points in array input 'points' and one point 'centre'
    """
    subtracted_x = np.square(np.subtract(centre[0], points[:, 0]))
    subtracted_y = np.square(np.subtract(centre[1], points[:, 1]))
    distance = np.sqrt(np.add(subtracted_x, subtracted_y))

    return distance


def binarise_image(greyscale_image, threshold_val):
    """
    Binarises grey scale image and returns black and white image
    """
    # Convert image to binary black and white image
    thresh, binary_image = cv2.threshold(
        greyscale_image,
        threshold_val,
        255,
        cv2.THRESH_BINARY
    )
    binary_image = binary_image[:, :, 0]

    return binary_image


def canny_edge_detection(image, blur=False, canny_threshold=50):
    """
    Takes a single image and applies a canny edge detection algorithm to it.

    Returns an image with the edges highlighted.

    :param canny_threshold: Threshold for canny edge detection.
    :param blur: If True, the image is blurred before edge detection.
    :param image: Image to be processed.
    :return: Image with edges highlighted.
    """
    # Blur the image to even out noise if required
    if blur is True:
        new_image = cv2.blur(image, (5, 5), 0)
    else:
        new_image = image

    # Canny Edge Detection
    canny_edge_image = cv2.Canny(image=new_image, threshold1=400, threshold2=canny_threshold)  # Canny Edge Detection

    return canny_edge_image

def extract_free_surface_ff(binary_image):
    """
    Returns the locations of the power-air boundary using the find first method
    """
    # Find all coordinates that have values of 255 in each column
    hh, ww = binary_image.shape[:2]
    window = binary_image.transpose()
    xy = []
    for i in range(len(window)):
        col = window[i]
        j = find_first(255, col)
        xy.extend((i, j))
    # Reshape into [[x1, y1],...]
    edge_coordinates = np.array(xy).reshape((-1, 2))
    xdata = edge_coordinates[:, 0]
    ydata = np.subtract(hh, edge_coordinates[:, 1])
    edge_coordinates = np.concatenate((np.vstack(xdata), np.vstack(ydata)), axis=1)

    return edge_coordinates


def crop_image(image, percentage=0):
    """
    Crops input image to a circle with a percentage of the diameter removed from the edges.
    """

    # Extract size
    hh, ww = image.shape[:2]
    hh2 = hh // 2
    ww2 = ww // 2

    # define circles
    radius = int((hh - (hh * (percentage / 100))) / 2)
    xc = hh // 2
    yc = ww // 2

    # Crop image to circle
    mask = np.zeros_like(image)
    mask = cv2.circle(mask, (xc, yc), radius, (255, 255, 255), -1)

    cropped_image = cv2.bitwise_and(image, mask)

    return cropped_image

def crop_points(points, image_resolution, crop_percentage):
    """Returns a set of points that are within a percentage circle of the centre of the original image"""
    # Find threshold radius to crop out points outside of
    threshold_radius = (image_resolution / 2) * ((100 - crop_percentage) / 100)

    # Find distance from centre to all points
    centre = [image_resolution / 2, image_resolution / 2]
    distance = find_distance(points, centre)

    # Threshold out points outside of crop circle
    threshold_index = np.where(distance > threshold_radius)
    y_data = points[:, 1]
    y_data[threshold_index] = np.nan
    y_data = np.array(y_data)
    x_data = points[:, 0]
    cropped_points = np.hstack((np.vstack(x_data), np.vstack(y_data)))

    return cropped_points


def get_edge_coordinates(edge_image):
    """
    Gets the x,y coordinates from an image with just the powder-air interface/free surface highlighted.
    :return: Returns the x,y coordinates of the interface.
    """
    # Get image size
    hh, ww = edge_image.shape[:2]

    # Extract coordinates of powder-air interface from canny edge detection image.
    window = edge_image.transpose()
    edge_coordinates = np.transpose((window == 255).nonzero())  # Find coordinates of pixels with a value of 255
    xdata = edge_coordinates[:, 0]
    ydata = edge_coordinates[:, 1]
    ydata_mirror = np.subtract(hh, ydata)  # Mirror ydata to flip surface to correct orientation.

    # Average y values at each x value
    try:
        minx = np.min(xdata)
        maxx = np.max(xdata)
    except ValueError:
        raise ValueError('ValueError: min() arg is an empty sequence. This is likely due to the image being inverted.')

    avg_y = np.ones(hh)*(np.nan)

    for x in range(minx, maxx):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # Ignore RuntimeWarnings
            avg_y[x] = np.nanmean(ydata_mirror[xdata == x])

    for i in range(minx, maxx):
        if np.isnan(avg_y[i]):
            avg_y[i] = (avg_y[i-1] + avg_y[i+1]) / 2

    # Make new x data
    xdata_new = np.arange(0, hh)

    edge_data = np.array(np.concatenate((np.vstack(xdata_new), np.vstack(avg_y)), axis=1))  # Combine x and y data

    return edge_data


def rotate_and_integerise(origin, points, angle, processing_diameter):
    """
    Rotate a set of points counterclockwise by a given angle around a given origin and integerises the x value.
    The angle should be given in radians.
    """
    ox, oy = origin
    px = points[:, 0]
    py = points[:, 1]

    qx = ox + math.cos(-angle) * (px - ox) - math.sin(-angle) * (py - oy)
    qy = oy + math.sin(-angle) * (px - ox) + math.cos(-angle) * (py - oy)
    qx = np.array(qx)
    qx = np.rint(qx)

    # Make vertical arrays and stack horizontally
    rotated_points = np.hstack((np.array([qx]).T, np.array([qy]).T))
    sorted_rotated_points = rotated_points[rotated_points[:, 0].argsort()]

    output_y = []
    output_x = []
    min_x = np.nanmin(sorted_rotated_points[:, 0])
    max_x = np.nanmax(sorted_rotated_points[:, 0])
    for i in range(processing_diameter-1):
        output_x.append(i)
        index = np.where(sorted_rotated_points[:, 0] == i)[0]

        if i < min_x:
            output_y.append(np.nan)
        elif i > max_x:
            output_y.append(np.nan)
        elif len(index) == 0:
            output_y.append(np.nan)
        elif len(index) == 1:
            value = sorted_rotated_points[:, 1][index]
            output_y.append(value[0])
        elif len(index) >= 2:
            values = sorted_rotated_points[:, 1][index]
            mean_value = np.mean(values)
            output_y.append(mean_value)

    interpolate_output_y = []
    for i in range(processing_diameter - 1):
        if i < min_x:
            interpolate_output_y.append(np.nan)
        elif i > max_x:
            interpolate_output_y.append(np.nan)
        elif math.isfinite(output_y[i]) is False:
            before_values = interpolate_output_y[i-1]
            after_values_fintite = False
            j = 1
            while after_values_fintite is False:
                after_values = output_y[i+j]
                after_values_fintite = math.isfinite(after_values)
                j += 1

            interpolated_value = (before_values + after_values) / 2
            interpolate_output_y.append(interpolated_value)
        elif math.isfinite(output_y[i]) is True:
            interpolate_output_y.append(output_y[i])

    final_rotated_points = np.hstack((np.array([output_x]).T, np.array([interpolate_output_y]).T))

    return final_rotated_points


def shear(angle, x, y):
    """
    |1  -tan(𝜃/2) |  |1        0|  |1  -tan(𝜃/2) |
    |0      1     |  |sin(𝜃)   1|  |0      1     |
    """
    # shear 1
    tangent = math.tan(angle / 2)
    new_x = round(x - y * tangent)
    new_y = y

    # shear 2
    new_y = round(new_x * math.sin(angle) + new_y)  # since there is no change in new_x according to the shear matrix

    # shear 3
    new_x = round(new_x - new_y * tangent)  # since there is no change in new_y according to the shear matrix

    return new_y, new_x


def rotate_3_shear(interface, angle, processing_diameter):
    """
    Rotates a 2D array by angle specified using the 3 shears method.
    :param interface: 2D array of the powder-air interface.
    :param angle: Angle to rotate the interface by in radians.
    :return: Returns the rotated interface.
    """
    new_y_array = []
    new_x_array = []
    for j, point in interface:
        y = processing_diameter-1-point-(processing_diameter/2)
        x = processing_diameter-1-j-(processing_diameter/2)
        if np.isnan(y):
            new_y, new_x = np.nan, np.nan
        else:
            new_y, new_x = shear(angle, x, y)
            new_y = processing_diameter/2 - new_y
            new_x = processing_diameter/2 - new_x

        new_y_array.append(new_y)
        new_x_array.append(new_x)

    rotated_array = np.hstack((np.array([new_x_array]).T, np.array([new_y_array]).T))
    sorted_rotated_points = rotated_array[rotated_array[:, 0].argsort()]

    output_y = []
    output_x = []
    min_x = np.nanmin(sorted_rotated_points[:, 0])
    max_x = np.nanmax(sorted_rotated_points[:, 0])
    for i in range(processing_diameter - 1):
        output_x.append(i)
        index = np.where(sorted_rotated_points[:, 0] == i)[0]

        if i < min_x:
            output_y.append(np.nan)
        elif i > max_x:
            output_y.append(np.nan)
        elif len(index) == 0:
            output_y.append(np.nan)
        elif len(index) == 1:
            value = sorted_rotated_points[:, 1][index]
            output_y.append(value[0])
        elif len(index) >= 2:
            values = sorted_rotated_points[:, 1][index]
            mean_value = np.mean(values)
            output_y.append(mean_value)

    interpolate_output_y = []
    for i in range(processing_diameter - 1):
        if i < min_x:
            interpolate_output_y.append(np.nan)
        elif i > max_x:
            interpolate_output_y.append(np.nan)
        elif math.isfinite(output_y[i]) is False:
            before_values = interpolate_output_y[i - 1]
            after_values_fintite = False
            j = 1
            while after_values_fintite is False:
                after_values = output_y[i + j]
                after_values_fintite = math.isfinite(after_values)
                j += 1

            interpolated_value = (before_values + after_values) / 2
            interpolate_output_y.append(interpolated_value)
        elif math.isfinite(output_y[i]) is True:
            interpolate_output_y.append(output_y[i])

    final_rotated_points = np.hstack((np.array([output_x]).T, np.array([interpolate_output_y]).T))

    return final_rotated_points


def find_center(image):
    white_pixels = np.argwhere(image == 255)
    center = np.mean(white_pixels, axis=0)
    return tuple(center.astype(int))


def avg_diff(numbers):
    numbers = sorted(numbers) # sort the numbers
    diffs = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)] # calculate the differences
    return sum(diffs)/len(diffs) # calculate the average


class AnalyseGranuDrum:

    def __init__(self, crop_percentage, common_filename, images_path, number_of_images, processing_diameter=400, use_3_shear_rotation=False, percentage_of_images=1, add_randomness=True):
        self.original_hh = None
        self.original_ww = None
        self.crop_percentage = crop_percentage
        self.common_filename = common_filename
        self.images_path = images_path
        self.number_of_images = number_of_images
        self.new_image_list = None
        self.interface_data = None
        self.processing_diameter = processing_diameter
        self.plotting_image = None
        self.use_3_shear_rotation = use_3_shear_rotation
        self.percentage_of_images = percentage_of_images
        self.add_randomness = add_randomness

    def import_images(self):
        """
        Takes a list of GranuDrum digital twin images and selects images at intervals based on the number of images available and the number of images being used.

        :return: Returns a list of images.
        """
        # Get list of all images with the common filename in images_path.
        image_list = []
        for file in os.listdir(self.images_path):
            if fnmatch.fnmatch(file, f'*{self.common_filename}*'):
                image_list.append(file)
        image_list.sort()
        if len(image_list) < 1:
            raise Exception(f'There are no images in "{self.images_path}" which contain "{self.common_filename}" in their filename.')

        # Reduce list of images to a percentage of the total number of images.
        if self.percentage_of_images < 1:
            image_list_length = len(image_list)
            new_image_list_length = int(image_list_length * self.percentage_of_images)
            image_list = image_list[:new_image_list_length]

        # Select all images or a set of images from the list with some randomness.
        if self.number_of_images == -1:
            new_image_list = image_list
        elif self.number_of_images < 1:
            raise Exception(f'The number of images must be >=1 or the number of images must be == -1 to use all images. The number of images to be used was found to be equal to {self.number_of_images}')
        elif self.number_of_images >= 1 and self.add_randomness is True:
            new_image_list = []
            image_index = np.delete(np.linspace(0, len(image_list), self.number_of_images + 1), 0)
            gap_size = image_index[1] - image_index[0]

            for j, image_number in enumerate(image_index):
                rand_val = randint(1, int(gap_size*0.05))  # Add some randomness to images chosen for processing
                new_image_list.append(image_list[int(image_number)-1])
        elif self.number_of_images >= 1 and self.add_randomness is False:
            new_image_list = []
            image_index = np.delete(np.linspace(0, len(image_list), self.number_of_images + 1), 0)
            for j, image_number in enumerate(image_index):
                new_image_list.append(image_list[int(image_number)-1])

        self.new_image_list = new_image_list

        return self.new_image_list

    def extract_interface(self, rotation=0, binary_threshold=10, edge_detection_threshold=10, canny_bool=False):
        """
        Takes a set of images from the GranuDrum digital twin and extracts x,y points of the free surface/air-powder interface.

        :param edge_detection_threshold: The threshold for the canny edge detection algorithm.
        :param binary_threshold: The threshold for making a binary version of greyscale image.
        :param rotation: The angle to rotate the images in degrees.
        :param canny_bool: If True, the canny edge detection algorithm is used to extract the interface. If False, the find first edge detection algorithm is used.
        :return: Stack of x,y coordinates for all images imported.
        """
        # Get image list.
        image_list = self.import_images()
        self.plotting_image = image_list[0]

        # Get images size
        cv_img_size_extract = cv2.imread(f'{self.images_path}/{image_list[0]}')
        self.original_hh, self.original_ww = cv_img_size_extract.shape[:2]

        # Take images listed in self.new_image_list and extract the interface.
        all_free_surface_points = []  # Array of all free surface points for all images
        centre_points = []  # Array of all centre points for all images

        for i, image in enumerate(image_list):
            cv_img = cv2.imread(f'{self.images_path}/{image}')

            # Rotate image if required
            if rotation != 0:
                cv_img = imutils.rotate(cv_img, angle=rotation)

            # Resize image if required
            cv_img = cv2.resize(cv_img, (0, 0), fx=(self.processing_diameter / self.original_hh),
                                fy=(self.processing_diameter / self.original_ww))  # Resize image

            # Extract interface
            cropped_image = crop_image(cv_img, percentage=0)  # Crop image to remove stray particles.
            binary_image = binarise_image(cropped_image, binary_threshold)  # Binarise image
            center = find_center(binary_image)
            if canny_bool is True:
                edge_image = canny_edge_detection(binary_image, canny_threshold=edge_detection_threshold)  # Run Canny algorithm to extract interface.
                cropped_image2 = crop_image(edge_image, percentage=self.crop_percentage)  # Crop image to get interface
                free_surface_points = get_edge_coordinates(cropped_image2)  # Extract x,y coordinates of interface.
            else:
                pre_cropped_image = crop_image(binary_image, percentage=1)  # Crop image to get interface
                find_first_interface = extract_free_surface_ff(pre_cropped_image)
                free_surface_points = crop_points(find_first_interface, image_resolution=self.processing_diameter, crop_percentage=self.crop_percentage)

            all_free_surface_points.append(np.array(free_surface_points))
            centre_points.append(center)

        # Combine all free surface points into one array.
        all_free_surface_points = np.stack(all_free_surface_points, axis=0)
        self.interface_data = all_free_surface_points

        # Find the horizontal shift between the images.
        x_centre_points = np.array(centre_points)[:, 0]
        horizontal_shift = avg_diff(x_centre_points)

        return all_free_surface_points, horizontal_shift

    def average_interface(self, interface_data):
        """
        Finds the average interface from a set of GranuDrum digital twin interfaces.

        :param interface_data: An array of x,y interface data from the GranuDrum digital twin.
        :return: average interface
        """

        # Find the angle to rotate the images by using best fit line.
        all_angles = []
        for i, data in enumerate(interface_data):
            data_no_nans = data[~np.isnan(data).any(axis=1)]
            best_fit_line = np.poly1d(np.polyfit(data_no_nans[:, 0], data_no_nans[:, 1], 1))
            x0, x1 = 0, 10
            y0, y1 = best_fit_line(x0), best_fit_line(x1)
            rotation_angle_radians_one_interface = np.arctan2((y1 - y0), (x1 - x0))
            all_angles.append(rotation_angle_radians_one_interface)

        rotation_angle_radians = np.mean(all_angles)
        rotation_angle_radians_3shears = rotation_angle_radians+0.5*np.pi

        # Rotate interfaces by the average angle.
        rotated_interfaces = []
        for i, interface_column in enumerate(interface_data):
            # Rotate interface
            if self.use_3_shear_rotation is True:
                rot_interface = rotate_3_shear(interface_column,rotation_angle_radians_3shears, self.processing_diameter)
                rotated_interfaces.append(rot_interface)
            else:
                rotated_interfaces.append(rotate_and_integerise([self.processing_diameter/2, self.processing_diameter/2],
                                                                interface_column,
                                                                rotation_angle_radians,
                                                                self.processing_diameter))
        rotated_interfaces = np.array(rotated_interfaces)

        # Find average interface of rotated interfaces
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # Ignore RuntimeWarnings
            y_mean_rotated = np.nanmean(rotated_interfaces[:, :, 1], axis=0)
        x = rotated_interfaces[0, :, 0]

        rotated_averaged_interface = np.concatenate((
            np.vstack(x),
            np.vstack(y_mean_rotated)),
            axis=1)

        if self.use_3_shear_rotation is True:
            derotated_averaged_interface = rotate_3_shear(rotated_averaged_interface,
                                                          -rotation_angle_radians_3shears,
                                                          self.processing_diameter)
        else:

            derotated_averaged_interface = rotate_and_integerise(
            [self.processing_diameter/2, self.processing_diameter/2],
            rotated_averaged_interface,
            -rotation_angle_radians,
            self.processing_diameter)



        # Return values
        all_interfaces = namedtuple("all_interfaces", ["rotated_average_interface",
                                                       "rotated_interfaces",
                                                       "derotated_averaged_interface",
                                                       "interfaces"])

        return all_interfaces(
            rotated_averaged_interface,
            rotated_interfaces,
            derotated_averaged_interface,
            interface_data
        )

    def cohesive_index(self, average_interface, interface_data):
        """
        Calculates the cohesive index from the average interface and a set of interface x,y positions.
        :param average_interface: An array of x,y interface data from the GranuDrum digital twin.
        :param interface_data: An array of x,y interface data from the GranuDrum digital twin.
        :return: cohesive index
        """
        # Find the standard deviation of all the interfaces at each x coordinate
        all_y_std = np.zeros(len(average_interface[:, 0]))
        all_y_data = []
        all_avg_minus_all_squared = []
        for x, value in enumerate(average_interface):
            all_y_data.append(interface_data[:, x, 1])
            number_y_points = np.sum(np.isfinite(interface_data[:, x, 1]))
            y_avg_minus_y = np.subtract(value[1], interface_data[:, x, 1])
            avg_minus_all_squared = np.power(y_avg_minus_y, 2)
            all_avg_minus_all_squared.append(avg_minus_all_squared)
            y_sum = np.nansum(avg_minus_all_squared)
            y_sum_divided_by_number_y_points = np.divide(y_sum, number_y_points)
            y_std = np.sqrt(y_sum_divided_by_number_y_points)
            all_y_std[x] = y_std

        all_y_std_vstack = np.vstack(np.array(all_y_std))

        # df1 = pd.DataFrame(all_avg_minus_all_squared)
        # df1.to_csv('all_avg_minus_all_squared.csv')
        # df = pd.DataFrame(all_y_data)
        # df.to_csv('y_data.csv')

        # Sum standard deviations and divide by the crop diameter
        y_std_sum = np.nansum(all_y_std)
        cohesive_index = y_std_sum / (self.processing_diameter-(self.processing_diameter*(self.crop_percentage/100)))
        cohesive_index = cohesive_index * (800 / self.processing_diameter)  # Correction factor

        return cohesive_index

    def plot_interface(self, interface_data, average_interface=None, dynamic_angle_points=None, polynomial=None, save_html=False, save_path=None, use_image=True, use_binary_background=False):
        """
        Plots the average interface and a set of interface x,y positions.
        :param interface_data: An array of x,y interface data from the GranuDrum digital twin.
        :param average_interface: An array of x,y average interface data from the GranuDrum digital twin.
        :param dynamic_angle_points: The top and bottom points used to calculate the dynamic angle of repose.
        :param polynomial: The polynomial function that has been fit to the average interface.
        :param save_html: Boolean to save the plot as a html file.
        :param save_path: The path to save the html file to.
        :return: None
        """

        fig = go.Figure()
        colours = sns.color_palette("colorblind")
        colours = colours.as_hex()


        for p, points in enumerate(interface_data):
            fig.add_trace(go.Scatter(x=interface_data[p][:, 0], y=interface_data[p][:, 1], mode='markers',
                                     name=f'Interface_{p}', marker=dict(color=colours[0], size=3)
                                     )
                          )

        if average_interface is not None:
            fig.add_trace(go.Scatter(x=average_interface[:, 0], y=average_interface[:, 1], mode='markers',
                                     name='Averaged Interface', line=dict(color=colours[1]))
                          )

        if dynamic_angle_points is not None:
            fig.add_trace(go.Scatter(x=data.points[:, 0], y=data.points[:, 1], mode='lines',
                                     name='Dynamic Angle of Repose', line=dict(color=colours[3], width=8))
                          )
        if polynomial is not None:
            poly_x = np.linspace(0, self.processing_diameter)
            poly_coeff_reversed = polynomial[::-1]
            poly_plot = np.poly1d(poly_coeff_reversed)
            y_data = poly_plot(poly_x)
            fig.add_trace(go.Scatter(x=poly_x, y=y_data, mode='lines',
                                     name='3rd Order Polynomial', line=dict(color=colours[4], width=4))
                          )
        if use_image is True:
            background_image = Image.open(f"{self.images_path}/{self.plotting_image}")
            if use_binary_background is True:
                background_image = background_image.convert('L')
                background_image = background_image.point( lambda p: 255 if p < 2 else 0 )
                background_image = background_image.convert('1')
                background_image = background_image.resize((self.processing_diameter, self.processing_diameter))
            else:
                background_image = Image.open(f"{self.images_path}/{self.plotting_image}")
                background_image = background_image.resize((self.processing_diameter, self.processing_diameter))
            w, h = background_image.size

        fig.update_layout(
            title_text="Air-Powder Interface",
            title_font_size=30,
            legend=dict(font=dict(size=20))
        )
        fig.update_xaxes(
            title_text="Pixels",
            title_font={"size": 20},
            tickfont_size=20)
        fig.update_yaxes(
            title_text="Pixels",
            title_font={"size": 20},
            tickfont_size=20,
            scaleanchor="x",
            scaleratio=1)
        if use_image is True:
            fig.add_layout_image(
                dict(
                    source=background_image,
                    xref="x",
                    yref="y",
                    x=0,
                    y=h,
                    sizex=w,
                    sizey=h,
                    sizing="stretch",
                    opacity=1,
                    layer="below"
                )
            )
        fig.update_layout(
            xaxis_range=[0, self.processing_diameter],
            yaxis_range=[0, self.processing_diameter],
        )
        fig.add_shape(type="circle",
                      xref="x", yref="y",
                      x0=0, y0=0, x1=self.processing_diameter, y1=self.processing_diameter,
                      line_color="red",
                      )
        fig.add_shape(type="circle",
                      xref="x", yref="y",
                      x0=(self.processing_diameter*(self.crop_percentage/100))/2,
                      y0=(self.processing_diameter*(self.crop_percentage/100))/2,
                      x1=self.processing_diameter-(self.processing_diameter*(self.crop_percentage/100))/2,
                      y1=self.processing_diameter-(self.processing_diameter*(self.crop_percentage/100))/2,
                      line=dict(
                          color="red",
                          dash='dot'
                      ),
                      )
        if save_html:
            fig.write_html("GD-Interface.html")
        else:
            fig.show()

    def dynamic_angle_of_repose(self, average_interface):
        """
        Calculates the dynamic angle of repose from the average interface and a set of interface x,y positions.
        :param average_interface: An array of x,y interface data from the GranuDrum digital twin.
        :return: Returns dynamic_angle_degrees: dynamic angle of repose and
        points: top and bottom points used to calculate the dynamic angle of repose
        """
        d_5 = self.processing_diameter / 5

        # Find central point to calculate dynamic angle of repose at
        centre_index = int(self.processing_diameter / 2)
        centre = average_interface[centre_index]

        # Find the points to use for dynamic angle of repose calculation
        ordinates_left = average_interface[0:centre_index]
        ordinates_right = average_interface[centre_index:len(average_interface)]

        distance_left = centre[0] - ordinates_left[:, 0]
        distance_right = ordinates_right[:, 0] - centre[0]

        top_left_index = find_nearest(distance_left, d_5 / 2)
        bottom_right_index = find_nearest(distance_right, d_5 / 2)

        top_left = ordinates_left[top_left_index]
        bottom_right = ordinates_right[bottom_right_index]
        points = np.vstack((top_left, bottom_right))

        # Calculate angle of repose
        dynamic_angle_degrees = 180 - math.degrees(
            math.atan2((top_left[1] - bottom_right[1]),
                       (top_left[0] - bottom_right[0])))

        # Return values
        dynamic_angle = namedtuple("dynamic_angle", ["dynamic_angle_degrees", "points"])

        return dynamic_angle(
            dynamic_angle_degrees,
            points
        )

    def polynomial_fit(self, average_interface, order=3):
        """Fits a polynomial to the average free surface calculated from a series of images"""
        # Extract x and y data from array
        average_interface = average_interface[~np.isnan(average_interface).any(axis=1)]
        xdata = average_interface[:, 0]
        ydata = average_interface[:, 1]

        poly_eqn = np.polynomial.polynomial.polyfit(xdata, ydata, order)

        return poly_eqn
