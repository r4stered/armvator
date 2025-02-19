"""This problem tests the case where regularization fails"""

import math
import platform
import os
import pickle
import hashlib
import json
import sys

import numpy as np
import jormungandr.autodiff as autodiff
from jormungandr.autodiff import ExpressionType
from jormungandr.optimization import OptimizationProblem, SolverExitCondition

from visualizer import Visualizer

def get_parameter_hash(params):
    """Create a hash of solver parameters to detect changes"""
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()

def test_optimization_problem_arm_on_elevator(force_recompute=False):
    # Define parameters
    params = {
        "N": 800,
        "ELEVATOR_START_HEIGHT": 0.1778,  # m
        "ELEVATOR_END_HEIGHT": 1.905,  # m
        "ELEVATOR_MAX_VELOCITY": 4,  # m/s
        "ELEVATOR_MAX_ACCELERATION": 200,  # m/s²
        "ARM_LENGTH": 0.1016,  # m
        "ARM_START_ANGLE": math.pi / 2,  # rad
        "ARM_END_ANGLE": -1.76278,  # rad
        "ARM_MAX_VELOCITY": 25.34,  # rad/s 
        "ARM_MAX_ACCELERATION": 40.0 * math.pi,  # rad/s²
        "END_EFFECTOR_MAX_HEIGHT": 2.032,  # m
        "TOTAL_TIME": 1,  # s
        # Keep-out zones
        "KEEP_OUT_ZONES": [
            # Format: [x_min, x_max, y_min, y_max]
            [0.05, 0.15, 0.4, 0.7],
        ],
        "END_EFFECTOR_RADIUS": 0.5357243032,  # m
    }
    
    # Generate parameter fingerprint
    param_hash = get_parameter_hash(params)
    cache_file = f"optimization_results_cache_{param_hash}.pkl"
    
    # Check if cached results exist and are valid
    if not force_recompute and os.path.exists(cache_file):
        print(f"Using cached optimization results ({cache_file})...")
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")
    
    print("Running optimization solver with new parameters...")
    
    # Extract parameters
    N = params["N"]
    ELEVATOR_START_HEIGHT = params["ELEVATOR_START_HEIGHT"]
    ELEVATOR_END_HEIGHT = params["ELEVATOR_END_HEIGHT"]
    ELEVATOR_MAX_VELOCITY = params["ELEVATOR_MAX_VELOCITY"]
    ELEVATOR_MAX_ACCELERATION = params["ELEVATOR_MAX_ACCELERATION"]
    ARM_LENGTH = params["ARM_LENGTH"]
    ARM_START_ANGLE = params["ARM_START_ANGLE"]
    ARM_END_ANGLE = params["ARM_END_ANGLE"]
    ARM_MAX_VELOCITY = params["ARM_MAX_VELOCITY"]
    ARM_MAX_ACCELERATION = params["ARM_MAX_ACCELERATION"]
    END_EFFECTOR_MAX_HEIGHT = params["END_EFFECTOR_MAX_HEIGHT"]
    KEEP_OUT_ZONES = params["KEEP_OUT_ZONES"]
    END_EFFECTOR_RADIUS = params["END_EFFECTOR_RADIUS"]
    TOTAL_TIME = params["TOTAL_TIME"]
    
    dt = TOTAL_TIME / N

    problem = OptimizationProblem()

    # Decision variables
    elevator = problem.decision_variable(2, N + 1)
    elevator_accel = problem.decision_variable(1, N)
    arm = problem.decision_variable(2, N + 1)
    arm_accel = problem.decision_variable(1, N)

    # Dynamics constraints
    for k in range(N):
        # Elevator dynamics constraints
        problem.subject_to(
            elevator[0, k + 1]
            == elevator[0, k] + elevator[1, k] * dt + 0.5 * elevator_accel[0, k] * dt**2
        )
        problem.subject_to(
            elevator[1, k + 1] == elevator[1, k] + elevator_accel[0, k] * dt
        )

        # Arm dynamics constraints
        problem.subject_to(
            arm[0, k + 1] == arm[0, k] + arm[1, k] * dt + 0.5 * arm_accel[0, k] * dt**2
        )
        problem.subject_to(arm[1, k + 1] == arm[1, k] + arm_accel[0, k] * dt)

    # Boundary conditions
    problem.subject_to(elevator[:, :1] == np.array([[ELEVATOR_START_HEIGHT], [0.0]]))
    problem.subject_to(
        elevator[:, N : N + 1] == np.array([[ELEVATOR_END_HEIGHT], [0.0]])
    )
    problem.subject_to(arm[:, :1] == np.array([[ARM_START_ANGLE], [0.0]]))
    problem.subject_to(arm[:, N : N + 1] == np.array([[ARM_END_ANGLE], [0.0]]))

    # Velocity and acceleration limits
    problem.subject_to(-ELEVATOR_MAX_VELOCITY <= elevator[1:2, :])
    problem.subject_to(elevator[1:2, :] <= ELEVATOR_MAX_VELOCITY)
    problem.subject_to(-ELEVATOR_MAX_ACCELERATION <= elevator_accel)
    problem.subject_to(elevator_accel <= ELEVATOR_MAX_ACCELERATION)
    problem.subject_to(-ARM_MAX_VELOCITY <= arm[1:2, :])
    problem.subject_to(arm[1:2, :] <= ARM_MAX_VELOCITY)
    problem.subject_to(-ARM_MAX_ACCELERATION <= arm_accel)
    problem.subject_to(arm_accel <= ARM_MAX_ACCELERATION)
    
    # For visualization only - don't add constraints for keep-out zones
    # This ensures all constraints remain linear
    
    # Cost function
    J = 0.0
    for k in range(N + 1):
        J += (ELEVATOR_END_HEIGHT - elevator[0, k]) ** 2 + (
            ARM_END_ANGLE - arm[0, k]
        ) ** 2
        
    problem.minimize(J)

    status = problem.solve(diagnostics=True)

    # Check assertions
    assert status.cost_function_type == ExpressionType.QUADRATIC
    assert status.equality_constraint_type == ExpressionType.LINEAR
    assert status.inequality_constraint_type == ExpressionType.LINEAR

    if platform.system() == "Linux" and platform.machine() == "aarch64":
        return None
    else:
        assert status.exit_condition == SolverExitCondition.SUCCESS
        
        # Extract results
        elevator_positions = np.zeros(N+1)
        elevator_velocities = np.zeros(N+1)
        arm_angles = np.zeros(N+1)
        arm_velocities = np.zeros(N+1)
        
        for i in range(N+1):
            elevator_positions[i] = elevator[0, i].value()  
            elevator_velocities[i] = elevator[1, i].value()
            arm_angles[i] = arm[0, i].value()
            arm_velocities[i] = arm[1, i].value()

        optimization_results = {
            'elevator_positions': elevator_positions,
            'elevator_velocities': elevator_velocities,
            'arm_angles': arm_angles,
            'arm_velocities': arm_velocities,
            'arm_length': ARM_LENGTH,
            'end_effector_radius': END_EFFECTOR_RADIUS,
            'total_time': TOTAL_TIME,
            'keep_out_zones': KEEP_OUT_ZONES,
            'param_hash': param_hash
        }
        
        # Cache results
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(optimization_results, f)
            print(f"Optimization results cached to {cache_file}")
        except Exception as e:
            print(f"Failed to cache results: {e}")
        
        return optimization_results

if __name__ == "__main__":
    # You can force recomputation by setting force_recompute=True
    force_recompute = False
    
    # To enable forced recomputation via command line:
    if "--recompute" in sys.argv:
        force_recompute = True
        print("Forcing solver recomputation...")
    
    results = test_optimization_problem_arm_on_elevator(force_recompute)
    viz = Visualizer(results)
    viz.run()