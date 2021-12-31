import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.interpolate import interp1d

#####################
# Begin helper code #
#####################

def calculate_std(upper, mean):
    """
	Calculate standard deviation based on the upper 95th percentile

	Args:
		upper: a 1-d numpy array with length N, representing the 95th percentile
            values from N data points
		mean: a 1-d numpy array with length N, representing the mean values from
            the corresponding N data points

	Returns:
		a 1-d numpy array of length N, with the standard deviation corresponding
        to each value in upper and mean
	"""
    return (upper-mean)/st.norm.ppf(.95)

def interp(target_year, input_years, years_data):
    """
	Interpolates data for a given year, based on the data for the years around it

	Args:
		target_year: an integer representing the year which you want the predicted
            sea level rise for
		input_years: a 1-d numpy array that contains the years for which there is data
		    (can be thought of as the "x-coordinates" of data points)
        years_data: a 1-d numpy array representing the current data values
            for the points which you want to interpolate, eg. the SLR mean per year data points
            (can be thought of as the "y-coordinates" of data points)

	Returns:
		the interpolated predicted value for the target year
	"""
    return np.interp(target_year, input_years, years_data, right=-99)

def load_slc_data():
    """
	Loads data from sea_level_change.csv and puts it into a numpy array

	Returns:
		a length 3 tuple of 1-d numpy arrays:
		    1. an array of years as ints
		    2. an array of 2.5th percentile sea level rises (as floats) for the years from the first array
		    3. an array of 97.5th percentile of sea level rises (as floats) for the years from the first array
        eg.
            (
                [2020, 2030, ..., 2100],
                [3.9, 4.1, ..., 5.4],
                [4.4, 4.8, ..., 10]
            )
            can be interpreted as:
                for the year 2020, the 2.5th percentile SLR is 3.9ft, and the 97.5th percentile would be 4.4ft.
	"""
    df = pd.read_csv('sea_level_change.csv')
    df.columns = ['Year','Lower','Upper']
    return (df.Year.to_numpy(),df.Lower.to_numpy(),df.Upper.to_numpy())

###################
# End helper code #
###################


##########
# Part 1 #
##########

def predicted_sea_level_rise(show_plot=False):
    """
	Creates a numpy array from the data in sea_level_change.csv where each row
    contains a year, the mean sea level rise for that year, the 2.5th percentile
    sea level rise for that year, the 97.5th percentile sea level rise for that
    year, and the standard deviation of the sea level rise for that year. If
    the year is between 2020 and 2100 and not included in the data, the values
    for that year should be interpolated. If show_plot, displays a plot with
    mean and the 95%, assuming sea level rise follows a linear trend.

	Args:
		show_plot: displays desired plot if true

	Returns:
		a 2-d numpy array with each row containing a year in order from 2020-2100
        inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
        deviation of the sea level rise for the given year
	"""
    years, lower, upper = load_slc_data()
    data_array = []
    counter = 0

    for year in range(2020, 2101):
        if year in years:
            low, high = lower[counter], upper[counter]
            data_row = [year, 0, low, high, 0]
            counter += 1
        else:
            low, high = interp(year, years, lower), interp(year, years, upper)
            data_row = [year, 0, low, high, 0]

        mean = (low + high)/2
        std = calculate_std(high, mean)
        data_row[1] = mean
        data_row[4] = std
        data_array.append(data_row)


    data = np.array(data_array)

    if show_plot:
        plt.plot(data[:, 0], data[:, 3], linestyle="dashed", label="Upper")
        plt.plot(data[:, 0], data[:, 2], linestyle="dashed", label= "Lower")
        plt.plot(data[:, 0], data[:, 1], label="Mean")
        plt.title("(expected results)")
        plt.xlabel("Year")
        plt.ylabel("Projected annual mean water level (ft)")
        plt.legend()
        plt.show()

    return data


def simulate_year(data, year, num):
    """
	Simulates the sea level rise for a particular year based on that year's
    mean and standard deviation, assuming a normal distribution.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year
		year: the year to simulate sea level rise for
        num: the number of samples you want from this year

	Returns:
		a 1-d numpy array of length num, that contains num simulated values for
        sea level rise during the year specified
	"""
    index = np.where(data[:, 0] == year)
    output = np.random.normal(data[index[0], 1], data[index[0], 4], num)
    return output


def plot_mc_simulation(data):
    """
	Runs and plots a Monte Carlo simulation, based on the values in data and
    assuming a normal distribution. Five hundred samples should be generated
    for each year.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year
	"""
    for year in data[:, 0]:
        samples_list = simulate_year(data, year, 500)
        y = samples_list
        x = np.full(samples_list.size, year)  
        plt.scatter(x, y, s=0.5, c="gray")

    plt.plot(data[:, 0], data[:, 3], linestyle="dashed", label="Upper bound")
    plt.plot(data[:, 0], data[:, 2], linestyle="dashed", label="Lower bound")
    plt.plot(data[:, 0], data[:, 1], label="Mean")
    plt.title("(expected results)")
    plt.xlabel("Year")
    plt.ylabel("Relative Water Level Change (ft)")
    plt.legend()
    plt.show()

##########
# Part 2 #
##########

def water_level_est(data):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year

	Returns:
		a list of simulated water levels for each year, in the order in which
        they would occur temporally
	"""
    water_estimate = [simulate_year(data, year, 1) for year in data[:, 0]]
    return water_estimate


def repair_only(water_level_list, water_level_loss_no_prevention, house_value=400000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a repair only strategy, where you would only pay
    to repair damage that already happened.

    The specific damage cost can be calculated using the numpy array
    water_level_loss_no_prevention, where each water level corresponds to the
    percent of property that is damaged.

    The repair only strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft, the cost is the
           house_value times the percentage of property damage for that water
           level. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_no_prevention: a 2-d numpy array where the first column is
            the SLR levels and the second column is the corresponding property damage expected
            from that water level with no flood prevention (as an integer percentage)
        house_value: the value of the property we are estimating cost for

	Returns:
		an list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
    damage_costs = []

    for water_level in water_level_list:
        if water_level <= 5:
            damage_costs.append(0)
        elif water_level in water_level_loss_no_prevention[:, 0]:
            row_of_water_level = np.where(water_level_loss_no_prevention[:, 0] == water_level)
            damage = water_level_loss_no_prevention[row_of_water_level, 1]*house_value/100/1000
            damage_costs.append(damage)
        elif water_level >= 10:
            damage_costs.append(house_value)
        else:
            loss_percentage = interp1d(water_level_loss_no_prevention[:, 0], water_level_loss_no_prevention[:, 1], fill_value= "extrapolate")
            damage = loss_percentage(water_level)*house_value/100/1000
            damage_costs.append(damage)

    return damage_costs


def wait_a_bit(water_level_list, water_level_loss_no_prevention, water_level_loss_with_prevention, house_value=400000, cost_threshold=100000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a wait a bit to repair strategy, where you start
    flood prevention measures after having a year with an excessive amount of
    damage cost.

    The specific damage cost can be calculated using the numpy array
    water_level_loss_no_prevention and water_level_loss_with_prevention, where
    each water level corresponds to the percent of property that is damaged.
    You should be using water_level_loss_no_prevention when no flood prevention
    measures are in place, and water_level_loss_with_prevention when there are
    flood prevention measures in place.

    Flood prevention measures are put into place if you have any year with a
    damage cost above the cost_threshold.

    The wait a bit to repair only strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft, the cost is the
           house_value times the percentage of property damage for that water
           level, which is affected by the implementation of flood prevention
           measures. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_no_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with no flood prevention
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for
        cost_threshold: the amount of cost incurred before flood prevention
            measures are put into place

	Returns:
		an list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
    prevention = False
    damage_costs = []
    

    for water_level in water_level_list:

        if prevention: 
            water_level_loss = water_level_loss_with_prevention
        else:
            water_level_loss = water_level_loss_no_prevention

        if water_level <= 5:
            damage_costs.append(0)
        elif water_level in water_level_loss[:, 0]:
            row_of_water_level = np.where(water_level_loss[:, 0] == water_level)
            damage = water_level_loss[row_of_water_level, 1]*house_value/100/1000
            damage_costs.append(damage)
        elif water_level >= 10:
            damage_costs.append(house_value)
        else:
            loss_percentage = interp1d(water_level_loss[:, 0], water_level_loss[:, 1], fill_value= "extrapolate")
            damage = loss_percentage(water_level)*house_value/100/1000
            damage_costs.append(damage)

        if damage_costs[-1] >= cost_threshold/1000:
            prevention = True

    return damage_costs


def prepare_immediately(water_level_list, water_level_loss_with_prevention, house_value=400000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a prepare immediately strategy, where you start
    flood prevention measures immediately.

    The specific damage cost can be calculated using the numpy array
    water_level_loss_with_prevention, where each water level corresponds to the
    percent of property that is damaged.

    The prepare immediately strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft, the cost is the
           house_value times the percentage of property damage for that water
           level, which is affected by the implementation of flood prevention
           measures. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for

	Returns:
		an list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
    damage_costs = []

    for water_level in water_level_list:
        if water_level <= 5:
            damage_costs.append(0)
        elif water_level in water_level_loss_with_prevention[:, 0]:
            row_of_water_level = np.where(water_level_loss_with_prevention[:, 0] == water_level)
            damage = water_level_loss_with_prevention[row_of_water_level, 1]*house_value/100/1000
            damage_costs.append(damage)
        elif water_level >= 10:
            damage_costs.append(house_value)
        else:
            loss_percentage = interp1d(water_level_loss_with_prevention[:, 0], water_level_loss_with_prevention[:, 1], fill_value= "extrapolate")
            damage = loss_percentage(water_level)*house_value/100/1000
            damage_costs.append(damage)

    return damage_costs




def plot_strategies(data, water_level_loss_no_prevention, water_level_loss_with_prevention, house_value=400000, cost_threshold=100000):
    """
	Runs and plots a Monte Carlo simulation of all of the different preparation
    strategies, based on the values in data and assuming a normal distribution.
    Five hundred samples should be generated for each year.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, the 5th percentile, 95th percentile, mean, and standard
            deviation of the sea level rise for the given year
        water_level_loss_no_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with no flood prevention
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for
        cost_threshold: the amount of cost incurred before flood prevention
            measures are put into place
	"""
    repair_only_means = []
    wait_a_bit_means = []
    prepare_immediately_means = []

    for year in data[:, 0]:
        samples_list = simulate_year(data, year, 500)

        repair_only_values = np.array(repair_only(samples_list, water_level_loss_no_prevention, house_value))
        x_1 = np.full(repair_only_values.shape, year)
        plt.scatter(x_1, repair_only_values, s=0.5, c="green")

        wait_a_bit_values = np.array(wait_a_bit(samples_list, water_level_loss_no_prevention, water_level_loss_no_prevention, house_value, cost_threshold))
        x_2 = np.full(wait_a_bit_values.shape, year)
        plt.scatter(x_2, wait_a_bit_values, s=0.5, c="blue")

        prepare_immediately_values = np.array(prepare_immediately(samples_list, water_level_loss_with_prevention, house_value))
        x_3 = np.full(prepare_immediately_values.shape, year)
        plt.scatter(x_3, prepare_immediately_values, s=0.5, c="red") 

        repair_only_means.append(np.mean(repair_only_values))
        wait_a_bit_means.append(np.mean(wait_a_bit_values))
        prepare_immediately_means.append(np.mean(prepare_immediately_values))

    repair_only_means_array = np.array(repair_only_means)
    wait_a_bit_means_array = np.array(wait_a_bit_means)
    prepare_immediately_means_array = np.array(prepare_immediately_means)

    plt.plot(data[:, 0], wait_a_bit_means_array, label="Wait-a-bit scenario", c="blue")
    plt.plot(data[:, 0], repair_only_means_array, label="Repair-only scenario", c="green")
    
    plt.plot(data[:, 0], prepare_immediately_means_array, label="Prepare-immediately scenario", c="red")
    plt.axis([2020, 2100, 0, 400])
    plt.xlabel("Year")
    plt.ylabel("Estimated Damage Cost ($K)")
    plt.legend()
    plt.show()




if __name__ == '__main__':
    data = predicted_sea_level_rise()
    water_level_loss_no_prevention = np.array([[5, 6, 7, 8, 9, 10], [0, 10, 25, 45, 75, 100]]).T
    water_level_loss_with_prevention = np.array([[5, 6, 7, 8, 9, 10], [0, 5, 15, 30, 70, 100]]).T
    #plot_mc_simulation(data)
    plot_strategies(data, water_level_loss_no_prevention, water_level_loss_with_prevention)
