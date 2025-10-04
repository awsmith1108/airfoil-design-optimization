Airfoil Optimization with XFOIL, Genetic Algorithms, and Reinforcement Learning

This project automates airfoil optimization using Genetic Algorithms (GA) and Reinforcement Learning (RL) with XFOIL. It includes tools for generating Bezier-curve-defined airfoils, creating .dat files for XFOIL, running XFOIL simulations, and analyzing aerodynamic performance.

------------------------------------------------------------
1. Genetic Algorithms (GA)

Purpose: Optimize airfoil shapes by evolving populations of candidate designs.

Records Maintained:
- Optimal airfoil shape
- Convergence history
- Performance metrics for each generation

Objective: Maximize lift-to-drag (L/D) ratio at a given angle of attack and Reynolds number.

Constraints:
- Leading and trailing edge coordinates fixed
- Maximum thickness constraint
- No self-intersecting airfoil shapes

Variables Optimized:
- Control points for Bezier curves defining the airfoil shape
- Reynolds number

------------------------------------------------------------
2. Reinforcement Learning (RL)

Purpose: Train an RL agent to iteratively improve airfoil designs.

Records Maintained:
- Optimal airfoil shape
- Action history
- State history
- Reward history
- Training performance over episodes

Objective: Maximize lift-to-drag (L/D) ratio at a given angle of attack and Reynolds number.

Constraints:
- Leading and trailing edge coordinates fixed
- Maximum thickness constraint
- No self-intersecting airfoil shapes

Variables:
- Control points for Bezier curves defining airfoil shape
- Discount factor for future rewards
- Learning rate for updating policy
- Number of episodes for training
- Reynolds number

------------------------------------------------------------
3. XFOIL Automation

This project includes Python scripts to run XFOIL automatically for a given airfoil shape and operating conditions.

Inputs:
- .dat file defining the airfoil
- Operating conditions: Reynolds number, angles of attack, number of iterations

Outputs:
- Polar data file (polar_file.txt) containing:
  - Lift coefficient (CL)
  - Drag coefficient (CD)
  - Lift-to-drag ratio (L/D)

Features:
- Writes XFOIL input files automatically
- Runs XFOIL using subprocess
- Parses polar data for performance metrics

------------------------------------------------------------
4. Bezier Curve Airfoil Generation

Purpose: Define smooth airfoil shapes using a set of control points.

Functions:
- Bezier curve generation: Converts control points into a continuous airfoil shape
- .dat file creation: Generates XFOIL-compatible files for simulation

Inputs:
- Control points
- Number of points to sample
- Sample point distribution

Outputs:
- Continuous airfoil shape function
- .dat file for XFOIL

------------------------------------------------------------
5. Visualization

Purpose: Analyze and visualize airfoil shapes and aerodynamic performance.

Inputs:
- Airfoil shape data
- Performance metrics (CL, CD, L/D)

Outputs:
- Plots of airfoil geometries
- Performance curves vs angle of attack

------------------------------------------------------------
6. Installation

1. Clone the repository:
   git clone <repository_url>
   cd <repository_folder>

2. Install required Python packages:
   pip install -r requirements.txt

3. Ensure XFOIL is installed and accessible from your system path.

------------------------------------------------------------
7. Usage

1. Generate airfoil shape and .dat file:
   python generate_airfoil.py

2. Run XFOIL simulations:
   python run_xfoil.py

3. Perform GA or RL optimization:
   python main_ga.py
   python main_rl.py

4. Visualize results:
   python visualize_results.py

------------------------------------------------------------
Notes

- All airfoil designs follow geometric constraints to ensure realistic shapes.
- Both GA and RL approaches aim to maximize L/D ratio but use different optimization strategies.
- The pipeline supports automated XFOIL simulation and post-processing for large-scale experiments.

------------------------------------------------------------
References

- XFOIL Airfoil Analysis Tool: http://web.mit.edu/drela/Public/web/xfoil/
- Genetic Algorithms: https://en.wikipedia.org/wiki/Genetic_algorithm
- Reinforcement Learning: https://en.wikipedia.org/wiki/Reinforcement_learning
- https://github.com/JARC99/xfoil-runner
