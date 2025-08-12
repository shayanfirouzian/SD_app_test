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

MODEL_PY = r'''
"""
Python model 'pp.py'
Translated using PySD
"""

from pathlib import Path
import numpy as np

from pysd.py_backend.statefuls import Integ
from pysd import Component

__pysd_version__ = "3.14.3"

__data = {"scope": None, "time": lambda: 0}

_root = Path(__file__).parent


component = Component()

#######################################################################
#                          CONTROL VARIABLES                          #
#######################################################################

_control_vars = {
    "initial_time": lambda: 0,
    "final_time": lambda: 100,
    "time_step": lambda: 1,
    "saveper": lambda: time_step(),
}


def _init_outer_references(data):
    for key in data:
        __data[key] = data[key]


@component.add(name="Time")
def time():
    """
    Current time of the model.
    """
    return __data["time"]()


@component.add(
    name="FINAL TIME", units="Month", comp_type="Constant", comp_subtype="Normal"
)
def final_time():
    """
    The final time for the simulation.
    """
    return __data["time"].final_time()


@component.add(
    name="INITIAL TIME", units="Month", comp_type="Constant", comp_subtype="Normal"
)
def initial_time():
    """
    The initial time for the simulation.
    """
    return __data["time"].initial_time()


@component.add(
    name="SAVEPER",
    units="Month",
    limits=(0.0, np.nan),
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"time_step": 1},
)
def saveper():
    """
    The frequency with which output is stored.
    """
    return __data["time"].saveper()


@component.add(
    name="TIME STEP",
    units="Month",
    limits=(0.0, np.nan),
    comp_type="Constant",
    comp_subtype="Normal",
)
def time_step():
    """
    The time step for the simulation.
    """
    return __data["time"].time_step()


#######################################################################
#                           MODEL VARIABLES                           #
#######################################################################


@component.add(
    name="Fractional predation rate",
    units="fraction/Time",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "reference_predation_rate": 1,
        "reference_predators": 1,
        "predators_y": 1,
    },
)
def fractional_predation_rate():
    """
    Fractional rate of decrease in prey from predation; equal to (beta*y) in the wiki article.
    """
    return reference_predation_rate() * (predators_y() / reference_predators())


@component.add(
    name="Predation rate per predator beta",
    units="fraction/Time/pred",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"reference_predation_rate": 1, "reference_predators": 1},
)
def predation_rate_per_predator_beta():
    """
    Prey predation parameter; beta in the wiki article
    """
    return reference_predation_rate() / reference_predators()


@component.add(
    name="Predator decrease rate",
    units="pred/Time",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"predators_y": 1, "predator_fractional_decrease_rate_gamma": 1},
)
def predator_decrease_rate():
    """
    Natural rate of decrease of predators from mortality and emmigration.
    """
    return predators_y() * predator_fractional_decrease_rate_gamma()


@component.add(
    name="Predator fractional decrease rate gamma",
    units="fraction/Time",
    limits=(0.0, 1.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def predator_fractional_decrease_rate_gamma():
    return 0.1


@component.add(
    name="Predator fractional growth rate",
    units="fraction/Time",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"reference_predator_growth_rate": 1, "reference_prey": 1, "prey_x": 1},
)
def predator_fractional_growth_rate():
    """
    Fractional rate of increase of predators; equal to (delta*x) in the wiki article.
    """
    return reference_predator_growth_rate() * (prey_x() / reference_prey())


@component.add(
    name="Predator growth per prey delta",
    units="fraction/Time/Prey",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"reference_predator_growth_rate": 1, "reference_prey": 1},
)
def predator_growth_per_prey_delta():
    """
    Predator growth parameter; delta in the wiki article
    """
    return reference_predator_growth_rate() / reference_prey()


@component.add(
    name="Predator increase rate",
    units="pred/Time",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"predators_y": 1, "predator_fractional_growth_rate": 1},
)
def predator_increase_rate():
    return predators_y() * predator_fractional_growth_rate()


@component.add(
    name="Predators Y",
    units="pred",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_predators_y": 1},
    other_deps={
        "_integ_predators_y": {
            "initial": {"relative_initial_predators": 1, "reference_predators": 1},
            "step": {"predator_increase_rate": 1, "predator_decrease_rate": 1},
        }
    },
)
def predators_y():
    return _integ_predators_y()


_integ_predators_y = Integ(
    lambda: predator_increase_rate() - predator_decrease_rate(),
    lambda: relative_initial_predators() * reference_predators(),
    "_integ_predators_y",
)


@component.add(
    name="Prey decrease rate",
    units="Prey/Time",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"prey_x": 1, "fractional_predation_rate": 1},
)
def prey_decrease_rate():
    """
    Rate of decrease in prey from predation
    """
    return prey_x() * fractional_predation_rate()


@component.add(
    name="Prey fractional growth rate alpha",
    units="fraction/Time",
    limits=(0.0, 1.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def prey_fractional_growth_rate_alpha():
    """
    Fractional growth rate of prey per unit time, absent predation
    """
    return 0.3


@component.add(
    name="Prey increase rate",
    units="Prey/Time",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"prey_fractional_growth_rate_alpha": 1, "prey_x": 1},
)
def prey_increase_rate():
    """
    Rate of increase in prey (e.g., births of elk or rabbits); prey are assumed to have unlimited food supply and therefore to increase exponentially in the absence of predation.
    """
    return prey_fractional_growth_rate_alpha() * prey_x()


@component.add(
    name="Prey X",
    units="Prey",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_prey_x": 1},
    other_deps={
        "_integ_prey_x": {
            "initial": {"relative_initial_prey": 1, "reference_prey": 1},
            "step": {"prey_increase_rate": 1, "prey_decrease_rate": 1},
        }
    },
)
def prey_x():
    return _integ_prey_x()


_integ_prey_x = Integ(
    lambda: prey_increase_rate() - prey_decrease_rate(),
    lambda: relative_initial_prey() * reference_prey(),
    "_integ_prey_x",
)


@component.add(
    name="Reference predation rate",
    units="fraction/Time",
    limits=(0.0, 1.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def reference_predation_rate():
    return 0.1


@component.add(
    name="Reference predator growth rate",
    units="fraction/Time",
    limits=(0.0, 1.0),
    comp_type="Constant",
    comp_subtype="Normal",
)
def reference_predator_growth_rate():
    return 0.2


@component.add(
    name="Reference predators",
    units="pred",
    limits=(0.0, np.nan),
    comp_type="Constant",
    comp_subtype="Normal",
)
def reference_predators():
    return 10


@component.add(
    name="Reference prey",
    units="Prey",
    limits=(0.0, np.nan),
    comp_type="Constant",
    comp_subtype="Normal",
)
def reference_prey():
    return 100


@component.add(
    name="Relative initial predators",
    units="Dmnl",
    limits=(0.0, np.nan),
    comp_type="Constant",
    comp_subtype="Normal",
)
def relative_initial_predators():
    """
    Initial predators, relative to the reference value
    """
    return 1


@component.add(
    name="Relative initial prey",
    units="Dmnl",
    limits=(0.0, np.nan),
    comp_type="Constant",
    comp_subtype="Normal",
)
def relative_initial_prey():
    """
    Initial prey, relative to the reference value
    """
    return 1
'''

# Cache writing+loading so it runs only once per session/process
@st.cache_resource
def make_and_load_model(embedded_source: str):
    """
    Writes the embedded model source to a real .py file and loads it with pysd.load().
    Returns (model, file_path) so you can (optionally) remove the file later.
    """
    # choose a stable path inside the app working dir (or tempfile.NamedTemporaryFile)
    # Using a stable name avoids re-writing on each rerun in Streamlit dev mode.
    out_path = pathlib.Path("pp_embedded.py")

    # Only write if missing or contents differ (helps when editing the app)
    should_write = True
    if out_path.exists():
        try:
            if out_path.read_text(encoding="utf-8") == embedded_source:
                should_write = False
        except Exception:
            should_write = True

    if should_write:
        out_path.write_text(embedded_source, encoding="utf-8")

    # Now load the model using pysd.load (expects a file path)
    model = pysd.load(str(out_path))
    return model, out_path

# load once
model, model_file = make_and_load_model(MODEL_PY)

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

