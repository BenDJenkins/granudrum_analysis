# GranuDrum Simulation Analysis Python Package (GSAP)
Python package for use with analysing data from the GranuDrum powder characterisation tool digital twin.

## Example Usage
```
from granudrum_analysis import AnalyseGranuDrum

path = ''  # Path to image data
filename = 'gd_rtd_'  # Common file name of images
n = 50  # Number of images to use in anlaysis

analysis = AnalyseGranuDrum(crop_percentage=10,
                            images_path=path,
                            common_filename=filename,
                            number_of_images=n,
                            processing_diameter=400)

interfaces, horizontal_shift = analysis.extract_interface(binary_threshold=1)  # Get x,y coordinates of all interface

data = analysis.average_interface(interfaces)

cohesive_index = analysis.cohesive_index(data.derotated_averaged_interface, data.interfaces)

angle_data = analysis.dynamic_angle_of_repose(data.derotated_averaged_interface)

poly_3 = analysis.polynomial_fit(data.derotated_averaged_interface, 3)

cohesiveindex_array.append(cohesive_index)
dynamicangle_array.append(angle_data.dynamic_angle_degrees)
```