import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import norm

def remove_outliers_using_spline(xdata, ydata, smoothing_factor=0.5, confidence_level=0.95):
    """
    Remove outliers from the data using spline fitting and a confidence interval.

    Parameters:
    xdata (np.array): The x-axis data.
    ydata (np.array): The y-axis data.
    smoothing_factor (float, optional): The smoothing factor for the spline. Default is 0.5.
    confidence_level (float, optional): The confidence level for identifying outliers (e.g., 0.95 for 95% confidence).

    Returns:
    tuple: Processed xdata and ydata with outliers removed.
    """
    # Compute the z-score for the given confidence level
    z_score = norm.ppf((1 + confidence_level) / 2)
    print(z_score)

    xdata = np.array(xdata)
    ydata = np.array(ydata)
    count = 0
    
    while True:
        print(xdata.size)

        if(count >= 30) : break
        # Ensure there are enough data points for spline fitting
        if len(xdata) < 4:
            raise ValueError("Not enough data points to fit a spline.")

        # Fit a spline to the data
        try:
            spl = UnivariateSpline(xdata, ydata, s=smoothing_factor)
        except Exception as e:
            raise RuntimeError(f"Spline fitting failed: {e}")

        y_hat = spl(xdata)
        plt.plot(xdata, ydata, 'b<')
        plt.plot(xdata, y_hat, 'r-')
        plt.show()
        
        # Calculate the standard deviation
        residuals = np.abs(ydata - y_hat)
        std_dev = np.std(residuals)
        
        # Identify outliers
        outliers = residuals > (std_dev * z_score)
        
        # Check if there are any outliers
        if not np.any(outliers):
            break
        
        # Remove outliers
        xdata = xdata[~outliers]
        ydata = ydata[~outliers]
        count = count + 1

    return xdata, ydata


def data_processing_for_fft(xdata, ydata, step_size=0.01, smoothing_factor=0.5):
    """
    Process data for FFT by smoothing with a spline.

    Parameters:
    xdata (np.array): The x-axis data.
    ydata (np.array): The y-axis data.
    step_size (float, optional): The step size for processing xdata. Default is 0.01.
    smoothing_factor (float, optional): The smoothing factor for the spline. Default is 0.5.

    Returns:
    tuple: Processed xdata and ydata.
    """
    # Ensure xdata and ydata are numpy arrays
    xdata_processed = np.arange(xdata[0], xdata[-1] + step_size, step_size)
    # Fit a spline to the data
    spl = UnivariateSpline(xdata, ydata, s=smoothing_factor)
    # Generate a fine grid for plotting the spline
    y_smooth = spl(xdata_processed)
    ydata_processed = []
    for x in xdata_processed:
        if x in xdata:
            ydata_processed.append(ydata[np.where(xdata == x)])
        else:
            ydata_processed.append(y_smooth[np.where(xdata_processed == x)])

    ydata_processed = np.array(ydata_processed).reshape(-1)
    return xdata_processed, ydata_processed

def fit_linear_regression(xdata, ydata, test_size_ratio=0.2, lower_limit_index=0, upper_limit_index=0):
    """
    Fit a linear regression model to the provided data.

    Parameters:
    xdata (np.array): The x-axis data.
    ydata (np.array): The y-axis data.
    test_size_ratio (float, optional): The ratio of the dataset to be used for testing. Default is 0.2.
    lower_limit_index (int, optional): The lower limit index for slicing the data. Default is 0.
    upper_limit_index (int, optional): The upper limit index for slicing the data. Default is 0.

    Returns:
    LinearRegression: The fitted linear regression model.
    """
    # Determine the subset of data to be used
    if lower_limit_index == 0 and upper_limit_index == 0:
        xdata_set = xdata
        ydata_set = ydata
    elif lower_limit_index != 0 and upper_limit_index == 0:
        xdata_set = xdata[lower_limit_index:]
        ydata_set = ydata[lower_limit_index:]
    elif lower_limit_index == 0 and upper_limit_index != 0:
        xdata_set = xdata[:upper_limit_index]
        ydata_set = ydata[:upper_limit_index]
    else:
        xdata_set = xdata[lower_limit_index:upper_limit_index]
        ydata_set = ydata[lower_limit_index:upper_limit_index]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(xdata_set.reshape(-1, 1), ydata_set, test_size=test_size_ratio, random_state=42)

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def extract_dominant_frequencies(xdata_processed, ydata_processed, model, n, rounding=3):
    """
    Extract the dominant frequencies from the processed data using FFT.

    Parameters:
    xdata_processed (np.array): The processed x-axis data.
    ydata_processed (np.array): The processed y-axis data.
    model (LinearRegression): The fitted linear regression model.
    n (int): The power threshold for selecting significant frequencies.
    rounding (int, optional): The number of decimal places to round the frequencies. Default is 3.

    Returns:
    np.array: The significant positive frequencies rounded to the specified number of decimal places.
    """
    # Perform FFT
    data_len = len(xdata_processed)
    dt = xdata_processed[1] - xdata_processed[0]
    freq = np.fft.fftfreq(data_len, dt)
    fft_values = np.fft.fft(ydata_processed - (model.coef_ * xdata_processed + model.intercept_))
    # Get the power spectrum
    power_spectrum = np.abs(fft_values) ** 2
    # Find the indices of frequencies with power >= 10^n
    significant_indices = np.where(power_spectrum >= 10**n)[0]
    # Extract the corresponding frequencies and powers
    significant_frequencies = freq[significant_indices]
    significant_powers = power_spectrum[significant_indices]
    # Sort the frequencies based on their corresponding power
    sorted_indices = np.argsort(significant_powers)[::-1]
    sorted_frequencies = significant_frequencies[sorted_indices]
    # Filter to keep only positive frequencies
    positive_frequencies = sorted_frequencies[sorted_frequencies > 0]
    return np.round(positive_frequencies, rounding)

def generate_fourier_series_function(frequencies):
    """
    Generate a Fourier series function based on the provided frequencies.

    Parameters:
    frequencies (np.array): An array of frequencies to be used in the Fourier series.

    Returns:
    function: The generated Fourier series function.
    """
    # Create the function definition string
    func_def = "def fourier_series(x, a0, " + ", ".join(
        f"a{i+1}, b{i+1}" for i in range(len(frequencies))
    ) + "):\n"
    func_def += "\n".join(f"    f{i+1} = {f}  # Frequency {i+1}" for i, f in enumerate(frequencies)) + "\n"
    func_def += "    return (a0 +\n"
    func_def += "\n".join(
        f"            a{i+1} * np.cos(2 * np.pi * f{i+1} * x) + b{i+1} * np.sin(2 * np.pi * f{i+1} * x) +" 
        for i in range(len(frequencies) - 1)
    )
    func_def += f"\n            a{len(frequencies)} * np.cos(2 * np.pi * f{len(frequencies)} * x) + b{len(frequencies)} * np.sin(2 * np.pi * f{len(frequencies)} * x))\n"

    # Compile the function definition into a Python object
    exec(func_def, globals())
    return fourier_series

def frequency_plot(xdata, ydata, step_size=0.01, smoothing_factor=0.5, test_size_ratio=0.2, semilog = False, lower_limit_index=0, upper_limit_index=0 ):
    # Process the data
    xdata_processed, ydata_processed = data_processing_for_fft(xdata, ydata, step_size, smoothing_factor)
    # Fit linear regression
    linear_model = fit_linear_regression(xdata, ydata, test_size_ratio, lower_limit_index, upper_limit_index)
    # Perform FFT
    data_len = len(xdata_processed)
    dt = xdata_processed[1] - xdata_processed[0]
    freq = np.fft.fftfreq(data_len, dt)
    fft_values = np.fft.fft(ydata_processed - linear_model.predict(xdata_processed.reshape(-1,1)))
    # Get the power spectrum
    power_spectrum = np.abs(fft_values) ** 2
    # Plot the power spectrum
    plt.figure(figsize=(10, 6))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    if(semilog):
        plt.semilogy(freq, power_spectrum, label='Power Spectrum')
        plt.title('Power Spectrum of the Data (Semilog Plot)')
    else:
        plt.plot(freq, power_spectrum, label='Power Spectrum')
        plt.title('Power Spectrum of the Data (normal Plot)')
    plt.show()



def fourier_curve_fit(xdata, ydata, step_size=0.01, smoothing_factor=0.5, test_size_ratio=0.2, n=4, rounding=3, freq_size=40, added_freq=0.0, lower_limit_index=0, upper_limit_index=0):
    """
    Fit a Fourier series to the data.

    Parameters:
    xdata (np.array): The x-axis data.
    ydata (np.array): The y-axis data.
    step_size (float, optional): The step size for processing xdata. Default is 0.01.
    smoothing_factor (float, optional): The smoothing factor for the spline. Default is 0.5.
    test_size_ratio (float, optional): The ratio of the dataset to be used for testing. Default is 0.2.
    n (int, optional): The power threshold for selecting significant frequencies. Default is 4.
    rounding (int, optional): The number of decimal places to round the frequencies. Default is 3.
    freq_size (int, optional): The number of frequencies to be used in the Fourier series. Default is 40.
    added_freq (float, optional): An additional frequency to be added to the frequency list. Default is 0.0.
    lower_limit_index (int, optional): The lower limit index for slicing the data. Default is 0.
    upper_limit_index (int, optional): The upper limit index for slicing the data. Default is 0.

    Returns:
    tuple: The Fourier series function, its parameters, and the linear regression model.
    """
    
    # Process the data
    xdata_processed, ydata_processed = data_processing_for_fft(xdata, ydata, step_size, smoothing_factor)
    # Fit linear regression
    linear_model = fit_linear_regression(xdata, ydata, test_size_ratio, lower_limit_index, upper_limit_index)
    # Extract dominant frequencies
    frequency_list = extract_dominant_frequencies(xdata_processed, ydata_processed, linear_model, n, rounding)
    # Generate the Fourier series function
    if added_freq == 0:
        fourier_function = generate_fourier_series_function(frequency_list[:freq_size] if freq_size < frequency_list.size else frequency_list)
    else:
        fourier_function = generate_fourier_series_function(np.append(frequency_list[:freq_size] if freq_size < frequency_list.size else frequency_list, added_freq))
    # Fit the Fourier series to the data
    xdata_set = xdata[lower_limit_index:upper_limit_index] if upper_limit_index != 0 else xdata[lower_limit_index:]
    ydata_set = ydata[lower_limit_index:upper_limit_index] if upper_limit_index != 0 else ydata[lower_limit_index:]
    params, params_covariance = curve_fit(fourier_function, xdata_set, ydata_set - linear_model.predict(xdata_set.reshape(-1,1)))
    return fourier_function, params, linear_model

def fourier_curve_plot(xdata, ydata, fourier_function, fourier_function_params, linear_model):
    """
    Plot the original data and the fitted Fourier series curve.

    Parameters:
    xdata (np.array): The x-axis data.
    ydata (np.array): The y-axis data.
    fourier_function (function): The fitted Fourier series function.
    fourier_function_params (np.array): The parameters of the fitted Fourier series function.
    linear_model (LinearRegression): The fitted linear regression model.

    Returns:
    None
    """
    x_fit = xdata
    y_fit = fourier_function(x_fit, *fourier_function_params) + linear_model.predict(x_fit.reshape(-1,1))
    plt.figure(figsize=(10, 6))
    plt.plot(xdata, ydata, 'o', label='Original Data', markersize=2)
    plt.plot(x_fit, y_fit , label='Fitted Fourier curve', color='red')
    plt.legend()
    plt.xlabel('X Data')
    plt.ylabel('Y Data')
    plt.title('Fourier Series curve Fit')
    plt.show()
