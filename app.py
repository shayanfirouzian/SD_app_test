import streamlit as st
import pysd
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Inject custom CSS to completely remove top padding and margins
st.markdown("""
    <style>
        .css-18e3th9 {
            padding-top: 0rem;
        }
        .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
        }
        h1 {
            margin-top: 0;
            margin-bottom: 0;
        }
        p {
            margin-top: 0;
            margin-bottom: 0;
        }
    </style>
    """, unsafe_allow_html=True)

# --- 1. Load the model from a local file ---
# The script will now load 'pp.mdl' from the same directory.
model_file_name = "pp.mdl"

# Use Streamlit's cache to avoid reloading the model on every interaction.
# This is crucial for performance.
@st.cache_resource
def load_model(path):
    """
    Loads the PySD model from the specified path.
    """
    try:
        model_instance = pysd.read_vensim(path)
        return model_instance
    except FileNotFoundError:
        st.error(f"Error: The file '{path}' was not found. Please ensure it's in the same directory as this script.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- 2. Load the model globally ---
model = load_model(model_file_name)


def main():
    """
    The main function to build the Streamlit application.
    """
    st.set_page_config(
        page_title="Predator-Prey Simulation",
        layout="wide"
    )

    st.title("Interactive Predator-Prey Simulation")
    # st.write("Use the slider on the sidebar to adjust the initial predator population and observe the population dynamics in real-time.")

    if not model:
        # Stop the application if the model failed to load
        return

    # --- 3. Define parameters and their slider ranges ---
    # This dictionary now only contains the 'Relative initial predators' parameter.
    parameters = {
        "relative_initial_predators": [1.0, 0.5, 5.0, 0.1, "Relative Initial Predators"],
    }

    # --- 4. Create sliders in the sidebar ---
    st.sidebar.header("Model Parameters")
    current_params = {}
    for param_name, [initial, min_val, max_val, res, label] in parameters.items():
        # Use st.slider to create a slider widget in the sidebar
        current_params[param_name] = st.sidebar.slider(
            label,
            min_value=min_val,
            max_value=max_val,
            value=initial,
            step=res
        )

    # --- 5. Run the simulation ---
    # The simulation runs automatically whenever a slider value changes.
    results = model.run(
        params=current_params,
        return_columns=['Prey X', 'Predators Y'],
        final_time=100.0
    )

    # --- 6. Create and display the plot ---
    fig, ax = plt.subplots(figsize=(7, 3.5))
    
    # Explicitly convert the pandas Series to a NumPy array for plotting
    ax.plot(results.index.to_numpy(), results['Prey X'].to_numpy(), label='Prey Population', color='green')
    ax.plot(results.index.to_numpy(), results['Predators Y'].to_numpy(), label='Predator Population', color='red')
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 400)

    ax.set_title("Predator-Prey Population Dynamics")
    ax.set_xlabel("Time (Months)")
    ax.set_ylabel("Population")
    ax.grid(True)
    ax.legend(loc='upper center')
    
    # Use st.pyplot() to display the Matplotlib figure
    st.pyplot(fig)
    
    # Clean up the plot to avoid memory issues
    plt.close(fig)


if __name__ == "__main__":
    main()
