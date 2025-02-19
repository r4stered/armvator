import math
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

optiresults = None

app = Dash(__name__)

@app.callback(
Output('tab-content', 'children'),
Input('visualization-tabs', 'value')
)
def render_tab_content(tab):
    if optiresults is None:
        return html.Div([
            html.H3("No optimization results available"),
            html.P("The optimization problem did not solve successfully.")
        ])
    
    if tab == 'tab-animation':
        # Animation tab
        fig = visualize_arm_on_elevator(
            optiresults['elevator_positions'],
            optiresults['arm_angles'],
            arm_length=optiresults['arm_length'],
            actual_time=optiresults['total_time'],
            end_effector_radius=optiresults.get('end_effector_radius', None),
            keep_out_zones=optiresults.get('keep_out_zones', None),
            end_effector_offset_x=0.0980873832,   # 7cm offset in x-direction
            end_effector_offset_y=0.221176723    # 3cm offset in y-direction
        )
        return html.Div([
            dcc.Graph(figure=fig, style={'height': '80vh'})
        ])
        
    elif tab == 'tab-position':
        # Position plots tab
        pos_fig, _ = plot_position_and_velocity(
            optiresults['elevator_positions'],
            optiresults['elevator_velocities'],
            optiresults['arm_angles'],
            optiresults['arm_velocities'],
            optiresults['total_time']
        )
        return html.Div([
            dcc.Graph(figure=pos_fig, style={'height': '80vh'})
        ])
        
    elif tab == 'tab-velocity':
        # Velocity plots tab
        _, vel_fig = plot_position_and_velocity(
            optiresults['elevator_positions'],
            optiresults['elevator_velocities'],
            optiresults['arm_angles'],
            optiresults['arm_velocities'],
            optiresults['total_time']
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

def visualize_arm_on_elevator(elevator_pos, arm_angles, arm_length=1.0, actual_time=4.0, 
                              end_effector_radius=None, keep_out_zones=None,
                              end_effector_offset_x=0.0, end_effector_offset_y=0.0):
    """
    Create an animated visualization of an arm mounted on an elevator with the elevator path shown.
    The end effector circle can be offset from the end point of the arm.
    
    Parameters:
    - elevator_pos: Array of elevator positions
    - arm_angles: Array of arm angles
    - arm_length: Length of the arm
    - actual_time: Duration of the motion
    - end_effector_radius: Radius of the end effector (if None, no circle is drawn)
    - keep_out_zones: List of [x_min, x_max, y_min, y_max] for each zone
    - end_effector_offset_x: X offset of the end effector circle from arm end point
    - end_effector_offset_y: Y offset of the end effector circle from arm end point
    """
    # Calculate time steps
    N = len(elevator_pos)
    time_steps = np.linspace(0, actual_time, N)
    
    # Calculate end effector positions (arm end point)
    arm_end_x = []
    arm_end_y = []
    
    # Calculate offset end effector positions (center of circle)
    offset_ee_x = []
    offset_ee_y = []
    
    for i in range(N):
        # Arm end point
        x = arm_length * math.cos(arm_angles[i])
        y = elevator_pos[i] + arm_length * math.sin(arm_angles[i])
        arm_end_x.append(x)
        arm_end_y.append(y)
        
        # Apply offset in the direction of the arm angle
        # We rotate the offset vector by the arm angle
        offset_angle = arm_angles[i]
        rotated_offset_x = end_effector_offset_x * math.cos(offset_angle) - end_effector_offset_y * math.sin(offset_angle)
        rotated_offset_y = end_effector_offset_x * math.sin(offset_angle) + end_effector_offset_y * math.cos(offset_angle)
        
        # Add offset to arm end point
        offset_ee_x.append(x + rotated_offset_x)
        offset_ee_y.append(y + rotated_offset_y)
    
    # Determine axis limits with padding
    padding = (end_effector_radius or 0.1) + max(abs(end_effector_offset_x), abs(end_effector_offset_y))
    min_y = min(elevator_pos) - padding
    max_ee_y = max(offset_ee_y) if offset_ee_y else max(arm_end_y)
    max_y = max([max_ee_y, max(elevator_pos)]) + padding
    
    y_range = max_y - min_y
    x_range = y_range
    min_x = -x_range/2
    max_x = x_range/2
    
    # Create figure
    fig = go.Figure()
    
    # Add keep-out zones as shapes
    shapes = []
    if keep_out_zones:
        for i, zone in enumerate(keep_out_zones):
            x_min, x_max, y_min, y_max = zone
            shapes.append({
                'type': 'rect',
                'xref': 'x',
                'yref': 'y',
                'x0': x_min,
                'y0': y_min,
                'x1': x_max,
                'y1': y_max,
                'line': {'color': 'orange', 'width': 2},
                'fillcolor': 'rgba(255, 165, 0, 0.3)',
                'layer': 'below'
            })
    
    # Grid lines
    grid_step = 0.5
    for x in np.arange(math.floor(min_x/grid_step)*grid_step, 
                    math.ceil(max_x/grid_step)*grid_step, 
                    grid_step):
        shapes.append({
            'type': 'line',
            'xref': 'x',
            'yref': 'y',
            'x0': x,
            'y0': min_y,
            'x1': x,
            'y1': max_y,
            'line': {'color': 'rgb(230, 230, 230)', 'width': 1},
            'layer': 'below'
        })
    
    for y in np.arange(math.floor(min_y/grid_step)*grid_step, 
                    math.ceil(max_y/grid_step)*grid_step, 
                    grid_step):
        shapes.append({
            'type': 'line',
            'xref': 'x',
            'yref': 'y',
            'x0': min_x,
            'y0': y,
            'x1': max_x,
            'y1': y,
            'line': {'color': 'rgb(230, 230, 230)', 'width': 1},
            'layer': 'below'
        })
    
    # Add initial traces
    # Arm
    fig.add_trace(go.Scatter(
        x=[0, arm_end_x[0]],
        y=[elevator_pos[0], arm_end_y[0]],
        mode='lines+markers',
        name='Arm',
        line=dict(width=3, color='blue'),
        marker=dict(size=[10, 8])
    ))
    
    # Add connector line to offset end effector if needed
    if end_effector_radius and (end_effector_offset_x != 0 or end_effector_offset_y != 0):
        fig.add_trace(go.Scatter(
            x=[arm_end_x[0], offset_ee_x[0]],
            y=[arm_end_y[0], offset_ee_y[0]],
            mode='lines',
            name='End Effector Mount',
            line=dict(width=2, color='green', dash='dash'),
        ))
    
    # Elevator path
    fig.add_trace(go.Scatter(
        x=[0] * len(elevator_pos),
        y=elevator_pos,
        mode='lines',
        name='Elevator Path',
        line=dict(width=2, color='red'),
        opacity=0.7
    ))
    
    # End effector path - use offset position if available
    path_x = offset_ee_x if end_effector_radius else arm_end_x
    path_y = offset_ee_y if end_effector_radius else arm_end_y
    fig.add_trace(go.Scatter(
        x=path_x,
        y=path_y,
        mode='lines',
        name='End Effector Path',
        line=dict(width=1, color='rgba(100, 100, 100, 0.5)'),
        opacity=0.5
    ))
    
    # Create frames for animation
    frames = []
    for i in range(N):
        # Check for collisions with keep-out zones
        collisions = []
        if end_effector_radius and keep_out_zones:
            current_ee_x = offset_ee_x[i]
            current_ee_y = offset_ee_y[i]
            
            for zone_idx, zone in enumerate(keep_out_zones):
                x_min, x_max, y_min, y_max = zone
                # Circle-rectangle collision check
                closest_x = max(x_min, min(current_ee_x, x_max))
                closest_y = max(y_min, min(current_ee_y, y_max))
                dist_x = current_ee_x - closest_x
                dist_y = current_ee_y - closest_y
                dist_squared = dist_x**2 + dist_y**2
                if dist_squared <= end_effector_radius**2:
                    collisions.append(zone_idx)
        
        # Create frame-specific shapes with collision highlighting
        frame_shapes = shapes.copy()
        for idx in collisions:
            zone = keep_out_zones[idx]
            x_min, x_max, y_min, y_max = zone
            # Add a highlighted version of the colliding zone
            frame_shapes.append({
                'type': 'rect',
                'xref': 'x',
                'yref': 'y',
                'x0': x_min,
                'y0': y_min,
                'x1': x_max,
                'y1': y_max,
                'line': {'color': 'red', 'width': 2},
                'fillcolor': 'rgba(255, 0, 0, 0.3)',
                'layer': 'below'
            })
        
        # Prepare frame data
        frame_data = []
        
        # Arm
        frame_data.append(go.Scatter(
            x=[0, arm_end_x[i]], 
            y=[elevator_pos[i], arm_end_y[i]],
            mode='lines+markers' if not end_effector_radius else 'lines',
            line=dict(width=3, color='blue'),
            marker=dict(size=[10, 8]) if not end_effector_radius else None,
            name='Arm'
        ))
        
        # Add connector line to offset end effector if needed
        if end_effector_radius and (end_effector_offset_x != 0 or end_effector_offset_y != 0):
            frame_data.append(go.Scatter(
                x=[arm_end_x[i], offset_ee_x[i]],
                y=[arm_end_y[i], offset_ee_y[i]],
                mode='lines',
                line=dict(width=2, color='green', dash='dash'),
                name='End Effector Mount'
            ))
        
        # Circular end effector if radius is provided
        if end_effector_radius:
            # Draw circle around the offset position
            theta = np.linspace(0, 2*np.pi, 20)
            circle_x = end_effector_radius * np.cos(theta) + offset_ee_x[i]
            circle_y = end_effector_radius * np.sin(theta) + offset_ee_y[i]
            
            # Set color based on collision status
            circle_color = 'red' if collisions else 'blue'
            circle_fill = 'rgba(255, 0, 0, 0.3)' if collisions else 'rgba(0, 0, 255, 0.2)'
            
            frame_data.append(go.Scatter(
                x=circle_x,
                y=circle_y,
                mode='lines',
                line=dict(color=circle_color),
                fill='toself',
                fillcolor=circle_fill,
                name='End Effector'
            ))
        
        # Elevator track
        elevator_x = [0, 0]
        elevator_y = [min(elevator_pos), elevator_pos[i]]
        frame_data.append(go.Scatter(
            x=elevator_x,
            y=elevator_y,
            mode='lines',
            line=dict(width=4, color='red'),
            opacity=1.0,
            name='Elevator Track'
        ))
        
        # End effector trail - use offset position if available
        trail_x = offset_ee_x[:i+1] if end_effector_radius else arm_end_x[:i+1]
        trail_y = offset_ee_y[:i+1] if end_effector_radius else arm_end_y[:i+1]
        frame_data.append(go.Scatter(
            x=trail_x,
            y=trail_y,
            mode='lines',
            line=dict(width=1, color='rgba(100, 100, 100, 0.5)'),
            opacity=0.5,
            name='End Effector Path'
        ))
        
        # Create frame with layout including the shapes
        frame = go.Frame(
            data=frame_data,
            name=f'frame{i}',
            layout=go.Layout(shapes=frame_shapes)
        )
        frames.append(frame)
    
    # Add frames to figure
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
    animation_frames = 60  # Target 60fps for smooth animation
    if N > animation_frames*actual_time:
        sampled_indices = np.linspace(0, N-1, int(animation_frames*actual_time), dtype=int)
        animation_frames_list = [f'frame{i}' for i in sampled_indices]
        frame_duration = 1000/animation_frames
    else:
        animation_frames_list = [f'frame{i}' for i in range(N)]
        frame_duration = actual_time * 1000 / len(animation_frames_list)
    
    # Animation controls
    play_button = {
        'type': 'buttons',
        'buttons': [
            {
                'label': 'Play (Real-time)',
                'method': 'animate',
                'args': [
                    animation_frames_list,
                    {
                        'frame': {'duration': frame_duration, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 0},
                        'mode': 'immediate'
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
    
    # Final layout setup
    final_elevator_pos = elevator_pos[-1]
    final_arm_angle = np.degrees(arm_angles[-1])
    
    # Create title with offset information if applicable
    title = f'Arm on Elevator Animation (Duration: {actual_time:.2f}s, Final: {final_elevator_pos:.2f}m, {final_arm_angle:.0f}°)'
    if end_effector_offset_x != 0 or end_effector_offset_y != 0:
        title += f' - End Effector Offset: ({end_effector_offset_x:.2f}, {end_effector_offset_y:.2f})m'
    
    fig.update_layout(
        title=title,
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
        shapes=shapes,  # Add initial shapes to layout
        showlegend=True,
        plot_bgcolor='rgb(248, 248, 248)'
    )
    
    return fig

class Visualizer:
    def __init__(self, results):
        global optiresults
        optiresults = results
        
        app.layout = html.Div([
            html.H1("Arm on Elevator Optimization"),
            dcc.Tabs(id='visualization-tabs', value='tab-animation', children=[
                dcc.Tab(label='Animation', value='tab-animation'),
                dcc.Tab(label='Position Plots', value='tab-position'),
                dcc.Tab(label='Velocity Plots', value='tab-velocity'),
            ]),
            html.Div(id='tab-content'),
        ])
    
    def run(self):
        app.run_server(debug=True)