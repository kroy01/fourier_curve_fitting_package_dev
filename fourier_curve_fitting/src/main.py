import sys
import os

# Add the parent directory of utilities to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utilities.FourierCurve as fc
import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resources', 'square_wave_data_new_frequency.csv'))
df = pd.read_csv(file_path, header=None)

xdata = df[0].to_numpy()
ydata = df[1].to_numpy()

#plt.plot(xdata, ydata, 'bo')
#plt.show()

xdata_processed, ydata_processed = fc.remove_outliers_using_spline(xdata, ydata, smoothing_factor=400, confidence_level=0.9)

print("final removed sezie : ", xdata_processed.size)

plt.plot(xdata_processed, ydata_processed, 'r*')
plt.show()

#fc.frequency_plot(xdata, ydata, step_size=0.01, smoothing_factor=0.5, test_size_ratio=0.2, semilog = False, lower_limit_index=0, upper_limit_index=0 )



#Fit the Fourier curve
# fourier_function, fourier_function_params, linear_model = fc.fourier_curve_fit(
#     xdata, ydata, 
#     step_size=0.01, smoothing_factor=0.5, 
#     test_size_ratio=0.2, 
#     n=4, rounding=3, 
#     freq_size=20, added_freq=0.0,#0.001, 
#     lower_limit_index=0, upper_limit_index=0
# )


# Plot the fitted curve
#fc.fourier_curve_plot(xdata, ydata, fourier_function, fourier_function_params, linear_model)