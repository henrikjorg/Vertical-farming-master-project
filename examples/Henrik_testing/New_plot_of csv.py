import pandas as pd
import matplotlib.pyplot as plt

import os
import sys  # Add this import

# Set path for running local script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))  # Adjusted path

def plot_simulation_data(csv_file_path):
    """
    Reads a CSV file and generates plots for simulation data.

    Parameters:
        csv_file_path (str): Path to the CSV file to plot.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Convert 'Date' column to datetime if needed
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    # Example 1: Temperature Dynamics
    plt.figure(figsize=(10, 6))
    if 'T_in' in df.columns and 'T_env' in df.columns and 'T_sup' in df.columns:
        plt.plot(df['t'], df['T_in'], label='Indoor Temperature (T_in)')
        plt.plot(df['t'], df['T_env'], label='Environmental Temperature (T_env)')
        plt.plot(df['t'], df['T_sup'], label='Supply Air Temperature (T_sup)')
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature (°C)')
        plt.title('Temperature Dynamics Over Time')
        plt.legend()
        plt.grid()
        plt.show()

    # Example 2: Crop Growth Metrics
    plt.figure(figsize=(10, 6))
    if 'LAI' in df.columns and 'CAC' in df.columns:
        plt.plot(df['t'], df['LAI'], label='Leaf Area Index (LAI)')
        plt.plot(df['t'], df['CAC'], label='Canopy Air CO2 Concentration (CAC)')
        plt.xlabel('Time (s)')
        plt.ylabel('Crop Metrics')
        plt.title('Crop Growth Dynamics Over Time')
        plt.legend()
        plt.grid()
        plt.show()

    # Example 3: Energy Consumption and CO2 Metrics
    plt.figure(figsize=(10, 6))
    if 'Q_env' in df.columns and 'Q_hvac' in df.columns and 'Phi_c_ass' in df.columns and 'Phi_c_inj' in df.columns:
        plt.plot(df['t'], df['Q_env'], label='Energy to Environment (Q_env)')
        plt.plot(df['t'], df['Q_hvac'], label='HVAC Energy (Q_hvac)')
        plt.plot(df['t'], df['Phi_c_ass'], label='CO2 Assimilation (Phi_c_ass)')
        plt.plot(df['t'], df['Phi_c_inj'], label='CO2 Injection (Phi_c_inj)')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy and CO2 Metrics')
        plt.title('Energy and CO2 Dynamics Over Time')
        plt.legend()
        plt.grid()
        plt.show()

    # Example 4: Actuator Settings
    plt.figure(figsize=(10, 6))
    if 'u_rot' in df.columns and 'u_fan' in df.columns and 'u_cool' in df.columns and 'u_heat' in df.columns:
        plt.plot(df['t'], df['u_rot'], label='HVAC Fan Rotation Rate (u_rot)')
        plt.plot(df['t'], df['u_fan'], label='Ventilation Rate (u_fan)')
        plt.plot(df['t'], df['u_cool'], label='Cooling Control (u_cool)')
        plt.plot(df['t'], df['u_heat'], label='Heating Control (u_heat)')
        plt.xlabel('Time (s)')
        plt.ylabel('Actuator Settings')
        plt.title('Actuator Dynamics Over Time')
        plt.legend()
        plt.grid()
        plt.show()

    # Example 5: Electricity Price and CO2 Concentration
    plt.figure(figsize=(10, 6))
    if 'Electricity price' in df.columns and 'CO2_in' in df.columns and 'CO2_out' in df.columns:
        plt.plot(df['t'], df['Electricity price'], label='Electricity Price')
        plt.plot(df['t'], df['CO2_in'], label='Indoor CO2 Concentration (CO2_in)')
        plt.plot(df['t'], df['CO2_out'], label='Outdoor CO2 Concentration (CO2_out)')
        plt.xlabel('Time (s)')
        plt.ylabel('Price and CO2 Concentration')
        plt.title('Electricity Price and CO2 Dynamics Over Time')
        plt.legend()
        plt.grid()
        plt.show()

# Example usage:
# Call this function with the path to your CSV file
folder_path = r"render\csv"
csv_file_path = os.path.join(folder_path, '171124-1719_simulation.csv')

plot_simulation_data(csv_file_path)
