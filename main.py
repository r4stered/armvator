"""This problem tests the case where regularization fails"""

import math
import platform

import numpy as np

import jormungandr.autodiff as autodiff
from jormungandr.autodiff import ExpressionType
from jormungandr.optimization import OptimizationProblem, SolverExitCondition

from dash import Dash, html, dcc
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots

import plotly.graph_objects as go

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Arm on Elevator Optimization"),
    dcc.Tabs(id='visualization-tabs', value='tab-animation', children=[
        dcc.Tab(label='Animation', value='tab-animation'),
        dcc.Tab(label='Position Plots', value='tab-position'),
        dcc.Tab(label='Velocity Plots', value='tab-velocity'),
    ]),
    html.Div(id='tab-content'),
])

optimization_results = None

@app.callback(
    Output('tab-content', 'children'),
    Input('visualization-tabs', 'value')
)
def render_tab_content(tab):
    if optimization_results is None:
        return html.Div([
            html.H3("No optimization results available"),
            html.P("The optimization problem did not solve successfully.")
        ])
    
    if tab == 'tab-animation':
        # Animation tab
        fig = visualize_arm_on_elevator(
            optimization_results['elevator_positions'],
            optimization_results['arm_angles'],
            arm_length=optimization_results['arm_length'],
            actual_time=optimization_results['total_time']
        )
        return html.Div([
            dcc.Graph(figure=fig, style={'height': '80vh'})
        ])
        
    elif tab == 'tab-position':
        # Position plots tab
        pos_fig, _ = plot_position_and_velocity(
            optimization_results['elevator_positions'],
            optimization_results['elevator_velocities'],
            optimization_results['arm_angles'],
            optimization_results['arm_velocities'],
            optimization_results['total_time']
        )
        return html.Div([
            dcc.Graph(figure=pos_fig, style={'height': '80vh'})
        ])
        
    elif tab == 'tab-velocity':
        # Velocity plots tab
        _, vel_fig = plot_position_and_velocity(
            optimization_results['elevator_positions'],
            optimization_results['elevator_velocities'],
            optimization_results['arm_angles'],
            optimization_results['arm_velocities'],
            optimization_results['total_time']
        )
        return html.Div([
            dcc.Graph(figure=vel_fig, style={'height': '80vh'})
        ])
    
    # Default case
    return html.Div([
        html.H3("Select a tab to view different visualizations")
    ])

def plot_position_and_velocity(elevator_pos, elevator_vel, arm_angles, arm_vel, actual_time):
    """
    Create plots of position and velocity over time
    
    Parameters:
    - elevator_pos: Array of elevator positions
    - elevator_vel: Array of elevator velocities
    - arm_angles: Array of arm angles
    - arm_vel: Array of arm angular velocities  
    - actual_time: Duration of the motion
    
    Returns:
    - Position plot figure
    - Velocity plot figure
    """
    # Create time array
    N = len(elevator_pos)
    time = np.linspace(0, actual_time, N)
    
    # Position plot
    pos_fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add elevator position
    pos_fig.add_trace(
        go.Scatter(
            x=time,
            y=elevator_pos,
            mode='lines',
            name='Elevator Position (m)',
            line=dict(color='red', width=2)
        ),
        secondary_y=False
    )
    
    # Add arm angle (convert to degrees for better readability)
    pos_fig.add_trace(
        go.Scatter(
            x=time,
            y=np.degrees(arm_angles),
            mode='lines',
            name='Arm Angle (degrees)',
            line=dict(color='blue', width=2)
        ),
        secondary_y=True
    )
    
    # Update position plot layout
    pos_fig.update_layout(
        title='Position vs Time',
        xaxis_title='Time (s)',
        legend=dict(x=0.01, y=0.99),
        hovermode='x unified'
    )
    
    pos_fig.update_yaxes(
        title_text="Elevator Position (m)",
        secondary_y=False
    )
    pos_fig.update_yaxes(
        title_text="Arm Angle (degrees)",
        secondary_y=True
    )
    
    # Velocity plot
    vel_fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add elevator velocity
    vel_fig.add_trace(
        go.Scatter(
            x=time,
            y=elevator_vel,
            mode='lines',
            name='Elevator Velocity (m/s)',
            line=dict(color='red', width=2)
        ),
        secondary_y=False
    )
    
    # Add arm angular velocity (convert to deg/s for better readability)
    vel_fig.add_trace(
        go.Scatter(
            x=time,
            y=np.degrees(arm_vel),
            mode='lines',
            name='Arm Angular Velocity (deg/s)',
            line=dict(color='blue', width=2)
        ),
        secondary_y=True
    )
    
    # Update velocity plot layout
    vel_fig.update_layout(
        title='Velocity vs Time',
        xaxis_title='Time (s)',
        legend=dict(x=0.01, y=0.99),
        hovermode='x unified'
    )
    
    vel_fig.update_yaxes(
        title_text="Elevator Velocity (m/s)",
        secondary_y=False
    )
    vel_fig.update_yaxes(
        title_text="Arm Angular Velocity (deg/s)",
        secondary_y=True
    )
    
    return pos_fig, vel_fig

def visualize_arm_on_elevator(elevator_pos, arm_angles, arm_length=1.0, actual_time=4.0):
    """
    Create an animated visualization of an arm mounted on an elevator with the elevator path shown.
    Uses equal axis scaling to maintain proper proportions.
    """
    # Calculate time steps
    N = len(elevator_pos)
    time_steps = np.linspace(0, actual_time, N)
    
    # Calculate end effector positions
    end_x = []
    end_y = []
    for i in range(N):
        x = arm_length * math.cos(arm_angles[i])
        y = elevator_pos[i] + arm_length * math.sin(arm_angles[i])
        end_x.append(x)
        end_y.append(y)
    
    # Determine axis limits
    min_y = min(elevator_pos) - 0.1
    max_y = max([max(end_y), max(elevator_pos)]) + 0.1
    y_range = max_y - min_y
    x_range = y_range
    min_x = -x_range/2
    max_x = x_range/2
    
    # Create figure
    fig = go.Figure()
    
    # Add initial traces
    fig.add_trace(go.Scatter(
        x=[0, end_x[0]],
        y=[elevator_pos[0], end_y[0]],
        mode='lines+markers',
        name='Arm',
        line=dict(width=3, color='blue'),
        marker=dict(size=[10, 8])
    ))
    
    fig.add_trace(go.Scatter(
        x=[0] * len(elevator_pos),
        y=elevator_pos,
        mode='lines',
        name='Elevator Path',
        line=dict(width=2, color='red'),
        opacity=0.7
    ))
    
    fig.add_trace(go.Scatter(
        x=end_x,
        y=end_y,
        mode='lines',
        name='End Effector Path',
        line=dict(width=1, color='rgba(100, 100, 100, 0.5)'),
        opacity=0.5
    ))
    
    # Create frames for animation
    frames = []
    for i in range(N):
        x_vals = [0, end_x[i]]
        y_vals = [elevator_pos[i], end_y[i]]
        
        elevator_x = [0, 0]
        elevator_y = [min(elevator_pos), elevator_pos[i]]
        
        frame = go.Frame(
            data=[
                go.Scatter(x=x_vals, y=y_vals, mode='lines+markers', line=dict(width=3, color='blue'), marker=dict(size=[10, 8])),
                go.Scatter(x=elevator_x, y=elevator_y, mode='lines', line=dict(width=4, color='red'), opacity=1.0),
                go.Scatter(x=end_x[:i+1], y=end_y[:i+1], mode='lines', line=dict(width=1, color='rgba(100, 100, 100, 0.5)'), opacity=0.5)
            ],
            name=f'frame{i}'
        )
        frames.append(frame)
    
    fig.frames = frames
    
    # Define slider steps
    slider_steps = []
    num_steps = min(41, N)
    step_indices = np.linspace(0, N-1, num_steps, dtype=int)
    
    for i in step_indices:
        slider_steps.append({
            'method': 'animate',
            'label': f'{time_steps[i]:.1f}s',
            'args': [[f'frame{i}'], {
                'frame': {'duration': 0, 'redraw': True},
                'mode': 'immediate',
                'transition': {'duration': 0}
            }]
        })
    
    # Calculate real-time animation speed
    # Important: Use fewer frames for smoother animation
    animation_frames = 60  # Target 30fps for smooth animation
    if N > animation_frames*actual_time:
        # If we have too many frames, sample them to get close to 30fps
        sampled_indices = np.linspace(0, N-1, int(animation_frames*actual_time), dtype=int)
        animation_frames_list = [f'frame{i}' for i in sampled_indices]
        frame_duration = 1000/animation_frames  # Each frame shows for 1/30th of a second
    else:
        # Otherwise use all frames
        animation_frames_list = [f'frame{i}' for i in range(N)]
        frame_duration = actual_time * 1000 / len(animation_frames_list)
    
    # Play button with accurate timing
    play_button = {
        'type': 'buttons',
        'buttons': [
            {
                'label': 'Play (Real-time)',
                'method': 'animate',
                'args': [
                    animation_frames_list,
                    {
                        'frame': {'duration': frame_duration, 'redraw': False},
                        'fromcurrent': True,
                        'transition': {'duration': 0},
                        'easing': 'linear'
                    }
                ]
            },
            {
                'label': 'Pause',
                'method': 'animate',
                'args': [[None], {
                    'frame': {'duration': 0, 'redraw': True},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }]
            }
        ]
    }
    
    # Grid lines (unchanged)
    grid_step = 0.5
    grid_lines = []
    
    for x in np.arange(math.floor(min_x/grid_step)*grid_step, 
                     math.ceil(max_x/grid_step)*grid_step, 
                     grid_step):
        grid_lines.append(dict(
            type='line', xref='x', yref='y',
            x0=x, y0=min_y, x1=x, y1=max_y,
            line=dict(color='rgb(230, 230, 230)', width=1)
        ))
    
    for y in np.arange(math.floor(min_y/grid_step)*grid_step, 
                     math.ceil(max_y/grid_step)*grid_step, 
                     grid_step):
        grid_lines.append(dict(
            type='line', xref='x', yref='y',
            x0=min_x, y0=y, x1=max_x, y1=y,
            line=dict(color='rgb(230, 230, 230)', width=1)
        ))
    
    # Final layout setup
    final_elevator_pos = elevator_pos[-1]
    final_arm_angle = np.degrees(arm_angles[-1])
    
    fig.update_layout(
        title=f'Arm on Elevator Animation (Duration: {actual_time:.2f}s, Final: {final_elevator_pos:.2f}m, {final_arm_angle:.0f}°)',
        xaxis=dict(
            title='X Position (m)',
            range=[min_x, max_x],
            dtick=grid_step,
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=2
        ),
        yaxis=dict(
            title='Y Position (m)',
            range=[min_y, max_y],
            dtick=grid_step,
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=2,
            scaleanchor='x',
            scaleratio=1
        ),
        updatemenus=[play_button],
        sliders=[{
            'active': 0,
            'steps': slider_steps,
            'currentvalue': {'prefix': 'Time: ', 'visible': True},
            'transition': {'duration': 0}
        }],
        showlegend=True,
        shapes=grid_lines,
        plot_bgcolor='rgb(248, 248, 248)'
    )
        
    return fig

def test_optimization_problem_arm_on_elevator(visualize=True):
    N = 800

    ELEVATOR_START_HEIGHT = 0  # m
    ELEVATOR_END_HEIGHT = 2.032  # m
    ELEVATOR_MAX_VELOCITY = 2.04  # m/s
    ELEVATOR_MAX_ACCELERATION = 200 # m/s²

    ARM_LENGTH = 0.1016  # m
    ARM_START_ANGLE = 0.0  # rad
    ARM_END_ANGLE = math.pi  # rad
    ARM_MAX_VELOCITY = 25.34  # rad/s
    ARM_MAX_ACCELERATION = 40.0 * math.pi  # rad/s²

    END_EFFECTOR_MAX_HEIGHT = 2.032 # m

    TOTAL_TIME = 1.5  # s
    dt = TOTAL_TIME / N

    problem = OptimizationProblem()

    elevator = problem.decision_variable(2, N + 1)
    elevator_accel = problem.decision_variable(1, N)

    arm = problem.decision_variable(2, N + 1)
    arm_accel = problem.decision_variable(1, N)

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

    # Elevator start and end conditions
    problem.subject_to(elevator[:, :1] == np.array([[ELEVATOR_START_HEIGHT], [0.0]]))
    problem.subject_to(
        elevator[:, N : N + 1] == np.array([[ELEVATOR_END_HEIGHT], [0.0]])
    )

    # Arm start and end conditions
    problem.subject_to(arm[:, :1] == np.array([[ARM_START_ANGLE], [0.0]]))
    problem.subject_to(arm[:, N : N + 1] == np.array([[ARM_END_ANGLE], [0.0]]))

    # Elevator velocity limits
    problem.subject_to(-ELEVATOR_MAX_VELOCITY <= elevator[1:2, :])
    problem.subject_to(elevator[1:2, :] <= ELEVATOR_MAX_VELOCITY)

    # Elevator acceleration limits
    problem.subject_to(-ELEVATOR_MAX_ACCELERATION <= elevator_accel)
    problem.subject_to(elevator_accel <= ELEVATOR_MAX_ACCELERATION)

    # Arm velocity limits
    problem.subject_to(-ARM_MAX_VELOCITY <= arm[1:2, :])
    problem.subject_to(arm[1:2, :] <= ARM_MAX_VELOCITY)

    # Arm acceleration limits
    problem.subject_to(-ARM_MAX_ACCELERATION <= arm_accel)
    problem.subject_to(arm_accel <= ARM_MAX_ACCELERATION)

    # Height limit
    if 0:
        heights = elevator[:1, :] + ARM_LENGTH * arm[:1, :].cwise_transform(
            autodiff.sin
        )
        problem.subject_to(heights <= END_EFFECTOR_MAX_HEIGHT)

    # Cost function
    J = 0.0
    for k in range(N + 1):
        J += (ELEVATOR_END_HEIGHT - elevator[0, k]) ** 2 + (
            ARM_END_ANGLE - arm[0, k]
        ) ** 2
    problem.minimize(J)

    status = problem.solve(diagnostics=True)

    assert status.cost_function_type == ExpressionType.QUADRATIC
    assert status.equality_constraint_type == ExpressionType.LINEAR
    assert status.inequality_constraint_type == ExpressionType.LINEAR

    if platform.system() == "Linux" and platform.machine() == "aarch64":
        # FIXME: Fails on Linux aarch64 with "factorization failed"
        assert status.exit_condition == SolverExitCondition.FACTORIZATION_FAILED
    else:
        assert status.exit_condition == SolverExitCondition.SUCCESS

        if status.exit_condition == SolverExitCondition.SUCCESS:
            # Extract the solved values
            elevator_positions = np.zeros(N+1)
            elevator_velocities = np.zeros(N+1)
            arm_angles = np.zeros(N+1)
            arm_velocities = np.zeros(N+1)
            
            for i in range(N+1):
                elevator_positions[i] = elevator[0, i].value()  
                elevator_velocities[i] = elevator[1, i].value()
                arm_angles[i] = arm[0, i].value()
                arm_velocities[i] = arm[1, i].value()

            global optimization_results
            optimization_results = {
                'elevator_positions': elevator_positions,
                'elevator_velocities': elevator_velocities,
                'arm_angles': arm_angles,
                'arm_velocities': arm_velocities,
                'arm_length': ARM_LENGTH,
                'total_time': TOTAL_TIME
            }
            
            # Only show fig directly if visualize is True
            if visualize:
                fig = visualize_arm_on_elevator(
                    elevator_positions,
                    arm_angles,
                    arm_length=ARM_LENGTH,
                    actual_time=TOTAL_TIME
                )
                fig.show()
            
    return status, elevator, arm

if __name__ == "__main__":
    result = test_optimization_problem_arm_on_elevator(visualize=False)
    app.run_server(debug=True)
