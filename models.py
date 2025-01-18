import plotly.express as px
import pandas as pd
import pandas as pd
import networkx as nx
from pyvis.network import Network
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
from pyvis.network import Network
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import numpy as np  

def load_df(path_to_df):
        
    df  = pd.read_csv(path_to_df)
    df['from-to']=df['FROM STORE']+' to '+df['TO STORE']
    df['DATE REQUESTED'] = pd.to_datetime(df['DATE REQUESTED'], format='%m/%d/%Y')

    return df


def filter_df(df,to_store =None,date_from=None,date_to=None):
    
    
    if to_store:
        df = df[df['TO STORE'] ==to_store]
        
        
    if date_from and date_to:
        
        df = df[(df['DATE REQUESTED'] >=date_from) & (df['DATE REQUESTED'] <=date_to)]
        
    elif date_from:
        df = df[df['DATE REQUESTED'] >=date_from]
        
    elif date_to:
        df = df[df['DATE REQUESTED'] <=date_to]    

    return df


def bubble_df(df,to_store =None):
    
    if to_store is None:
            
        store_names = list(set(df['TO STORE']))
        
        small_df = pd.DataFrame(columns=['To', 'Volume', 'Cost'])
        
        
        for i in range(0,len(store_names)):
            
            to = store_names[i]
            volume = df[df['TO STORE']==store_names[i]]['Quantity'].sum()
            cost = df[df['TO STORE']==store_names[i]]['cost'].sum()
            
            
            row = pd.DataFrame({'To':[to], 'Volume':[volume], 'Cost':[cost]})
            
            small_df = pd.concat([small_df, row], ignore_index=True)
        
    
        return small_df
    
    else: 
        
        store_names = list(set(df['from-to']))
        
        small_df = pd.DataFrame(columns=['To', 'Volume', 'Cost'])
        
        
        for i in range(0,len(store_names)):
            
            to = store_names[i]
            volume = df[df['from-to']==store_names[i]]['Quantity'].sum()
            cost = df[df['from-to']==store_names[i]]['cost'].sum()
            
            
            row = pd.DataFrame({'To':[to], 'Volume':[volume], 'Cost':[cost]})
            
            small_df = pd.concat([small_df, row], ignore_index=True)
        

        return small_df


def plot_bubble_volume(small_df):
   small_df = small_df.sort_values(by="Volume", ascending=False)

   fig = px.bar(small_df, x="To", y="Volume", color="To")
   
   fig.update_layout(
        title="Quantity Moved To Store",
        yaxis=dict(title=''),margin=dict(l=40, r=40, t=40, b=40)
        
    )
   
   fig.update_layout(width=560,height=350)
   
   return fig.to_html(full_html=False)

def plot_bubble_cost(small_df):

    small_df = small_df.sort_values(by="Cost", ascending=False)
    
    small_df.reset_index(drop=True, inplace=True)
    sum_rows = small_df.iloc[len(small_df)-5:len(small_df)].sum()
    small_df = small_df[0:5]
    new_row = sum_rows.to_frame().T
    new_row['To'] = 'Others'

    small_df = pd.concat([small_df, new_row], ignore_index=True)

    # Add columns for random movement for animation
    small_df['x_pos'] = np.random.rand(len(small_df))
    small_df['y_pos'] = np.random.rand(len(small_df))

    # Set initial size of the bubbles, and apply a scaling factor
    bubble_sizes = list(small_df["Cost"])  # Get bubble sizes from the 'Cost' column
    scaling_factor = 0.1  # Reduce the size by this factor
    bubble_sizes = [size * scaling_factor for size in bubble_sizes]

    # Create a color scale based on the 'To' column (categorical data)
    color_map = {category: idx for idx, category in enumerate(small_df['To'].unique())}
    small_df['color_val'] = small_df['To'].map(color_map)

    # Create initial scatter trace
    trace = go.Scatter(
        x=small_df['x_pos'],
        y=small_df['y_pos'],
        mode='markers+text',  # Show both markers and text
        marker=dict(
            size=bubble_sizes,
            color=small_df['color_val'],  # Pass the numerical color value
            colorscale='Viridis',  # Use a color scale for better visualization
            showscale=True,  # Display the color scale
            sizemode='diameter',  # Ensure sizes are based on bubble diameter
            sizemin=4,  # Minimum size for smaller bubbles
            sizeref=2  # Reference size to control the maximum size
        ),
        text=small_df['To'],  # Display names inside bubbles
        textposition='middle center',  # Position the text inside the bubbles
        customdata=small_df['Cost'],  # Attach the cost value to each marker for hover
        hovertemplate="<b>%{text}</b><br>Cost: £%{customdata:.2f}"  # Hover shows the cost value
    )

    # Create frames for the animation (make the bubbles move around)
    frames = []
    for _ in range(30):  # 30 frames for smooth animation
        small_df['x_pos'] += np.random.uniform(-0.05, 0.05, len(small_df))
        small_df['y_pos'] += np.random.uniform(-0.05, 0.05, len(small_df))
        small_df['x_pos'] = small_df['x_pos'].clip(0, 1)  # Ensure x stays between 0 and 1
        small_df['y_pos'] = small_df['y_pos'].clip(0, 1)  # Ensure y stays between 0 and 1
        
        frame = go.Frame(
            data=[go.Scatter(
                x=small_df['x_pos'],
                y=small_df['y_pos'],
                mode='markers+text',  # Show both markers and text in each frame
                marker=dict(
                    size=bubble_sizes,
                    color=small_df['color_val'],  # Update color values in each frame
                    colorscale='Viridis',
                    showscale=True
                ),
                text=small_df['To'],
                textposition='middle center',
                customdata=small_df['Cost'],
                hovertemplate="<b>%{text}</b><br>Cost: £%{customdata:.2f}"  # Hover shows the cost value
            )],
            name=str(_)
        )
        frames.append(frame)

    # Create the figure and add initial trace
    fig = go.Figure(
        data=[trace],
        frames=frames
    )

    # Update layout to include animation and remove the play button, grid, and scale
    fig.update_layout(
        title="Value Moved To Store (£)",
        xaxis=dict(showline=False, showgrid=False, showticklabels=False, zeroline=False, title=''),
        yaxis=dict(showline=False, showgrid=False, showticklabels=False, title=''),
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        width=560,
        height=350,
        updatemenus=[dict(
            type='buttons', 
            showactive=False,
            buttons=[dict(
                label='Play', 
                method='animate', 
                args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]
            )]
        )]
    )

    return fig.to_html(full_html=False)

def network_graph(df):
    
    df['count'] = df.groupby('from-to')['from-to'].transform('count')
    df['weighted_sum'] = df.groupby('from-to')['Quantity'].transform('sum')
    df['weight'] = df['count'] *df['weighted_sum']

    df = df.drop_duplicates(subset='from-to', keep='first')
    df =df[['FROM STORE','TO STORE','weight']]
    df.columns = ['source','target','weight']

    scaler = MinMaxScaler()
    df['scaled_weight'] = scaler.fit_transform(df[['weight']])  # Scale 'weight' to the range [0, 1]

    # Create a directed NetworkX graph
    nx_graph = nx.DiGraph()

    # Add edges with scaled weight to the graph
    for _, row in df.iterrows():
        source = row['source']
        target = row['target']
        scaled_weight = row['scaled_weight']  # Use the scaled weight
        nx_graph.add_edge(source, target, weight=scaled_weight)

    nt = Network(height='100%', width='100%')  # Dynamic sizing based on container size
    nt.from_nx(nx_graph)
    # Count the frequency a node is a target (in-degree)
    node_target_frequency = {node: 0 for node in nx_graph.nodes()}
    for edge in nx_graph.edges:
        target_node = edge[1]  # Get the target node of the edge
        node_target_frequency[target_node] += 1

    # Normalize the target frequency to use it for sizing the nodes
    max_frequency = max(node_target_frequency.values())
    min_frequency = min(node_target_frequency.values())
    normalized_frequency = {
        node: (frequency - min_frequency) / (max_frequency - min_frequency)
        for node, frequency in node_target_frequency.items()
    }

    # Define the blue to orange to red heatmap palette for node colors
    color_palette = ['#CCE5FF', '#99CCFF', '#66B3FF', '#3399FF', '#0066FF', '#0033CC', '#000099']

    # Create a Pyvis Network and populate it with the NetworkX graph
    nt = Network('1000px', '1500px')

    # Pass the NetworkX graph to Pyvis
    nt.from_nx(nx_graph)

    # Define minimum and maximum size limits for the nodes
    size_min = 10   # Minimum node size
    size_max = 50   # Maximum node size

    # Color and size the nodes based on their frequency
    for node in nt.nodes:
        # Get the node's normalized target frequency
        frequency = normalized_frequency.get(node['id'], 0)
        
        # Scale the node size based on frequency, and clamp between size_min and size_max
        size = frequency * (size_max - size_min) + size_min  # Scale to range [size_min, size_max]
        node['size'] = size
        
        # Color the nodes dynamically based on frequency (optional)
        color_index = int(frequency * (len(color_palette) - 1))  # Map the frequency to an index in the color palette
        node['color'] = color_palette[color_index]

        # Make the node labels black and bold
        node['font'] = {'size': 30, 'color': 'black', 'face': 'arial', 'weight': 'bold'}

    # Adjust edge widths based on the scaled weight
    for edge in nt.edges:
        scaled_weight = edge['width']  # Directly access the 'weight' attribute
        edge["width"] = scaled_weight * 10  # Adjust the width factor as necessary

        # Add arrows to the directed edges
        edge["arrows"] = "to"

    # Adjust the physics settings to space the nodes out more
    nt.set_options("""
    var options = {
"physics": {
    "enabled": true,
    "barnesHut": {
        "gravitationalConstant": -3000,
        "springLength": 600,
        "springConstant": 0.05,
        "damping": 0.1
    },
    "repulsion": {
        "nodeDistance": 300,
        "centralGravity": 0.05
    },
    "hierarchicalRepulsion": {
        "nodeDistance": 200,
        "springLength": 400
    }

    }
    }
    """)

    
    nt.write_html('E:/WebApp_Projects/IBT_dashboard/templates/network_graph.html')
    



def line_graph(df):

    df = df.groupby(by='DATE REQUESTED')["Auth Code"].nunique()

    df = pd.DataFrame(df)

    df = df.reset_index()


    x_min, x_max = df["DATE REQUESTED"].min(), df["DATE REQUESTED"].max()
    y_min, y_max = 0, df["Auth Code"].max() + 1  # Add buffer for better visualization

    frames = []
    for i in range(1, len(df) + 1):
        frame = go.Frame(
            data=[
                go.Scatter(
                    x=df["DATE REQUESTED"][:i],
                    y=df["Auth Code"][:i],
                    mode="markers+lines",
                    marker=dict(size=10, color="blue"),
                    name="Auth Code"
                )
            ],
            name=f"Frame {i}"
        )
        frames.append(frame)

    # Create initial scatter plot
    fig = go.Figure(
        data=[
            go.Scatter(
                x=[df["DATE REQUESTED"][0]],
                y=[df["Auth Code"][0]],
                mode="markers+lines",
                marker=dict(size=10, color="blue"),
                name="Auth Code"
            )
        ],
        layout=go.Layout(
            title="Auth Code TimeSeries",
            xaxis=dict(
                title="Date Requested",
                type="date",
                range=[x_min, x_max],  # Set static range for x-axis
            ),
            yaxis=dict(
                title="Auth Codes Count",
                range=[y_min, y_max],  # Set static range for y-axis
            ),
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                None, 
                                {
                                    "frame": {"duration": 500, "redraw": True},
                                    "fromcurrent": True,
                                    "mode": "immediate",
                                    "animation_loop": True  # Enable looping
                                }
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                }
            ],
        ),
        frames=frames,
    )
    fig.update_layout(width=560,height=350,margin=dict(l=40, r=40, t=40, b=40))
    return fig.to_html(full_html=False)


def my_map():
        towns_data = {
            "East Midlands": (52.95, -1.15),
            "Dalton Park": (54.8167, -1.3667),
            "Doncaster": (53.5228, -1.1287),
            "Castleford": (53.725, -1.356),
            "Batley Mill": (53.7034, -1.633),
            "Hull": (53.7676, -0.3274),
            "Meadowhall": (53.414, -1.412),
            "Icon": (52.0406, -0.7594),
            "Bridgend": (51.5079, -3.5778),
            "Gretna": (54.998, -3.066),
            "Birmingham": (52.4862, -1.8904),
            "Shiremoor": (55.032, -1.508),
            "Westfield (London)": (51.5423, -0.005),
            "Metro Centre": (54.958, -1.665),
            "Colne": (53.856, -2.175),
            "Swindon": (51.5685, -1.7722),
            "Coventry": (52.4081, -1.5106),
            "Ashford": (51.148, 0.87),
            "Fleetwood": (53.9167, -3.0357),
            "Braintree": (51.8787, 0.5536),
            "BM Grantham": (52.9128, -0.6424),
            "Morleys Bexleyheath": (51.457, 0.139),
            "Aylesbury": (51.8156, -0.8084)
        }

        weights = {
            "East Midlands": 21,
            "Dalton Park": 42,
            "Doncaster": 38,
            "Castleford": 73,
            "Batley Mill": 36,
            "Hull": 3,
            "Meadowhall": 6,
            "Icon": 16,
            "Bridgend": 19,
            "Gretna": 13,
            "Birmingham": 10,
            "Shiremoor": 1,
            "Westfield (London)": 17,
            "Metro Centre": 6,
            "Colne": 2,
            "Swindon": 3,
            "Coventry": 13,
            "Ashford": 1,
            "Fleetwood": 14,
            "Braintree": 4,
            "BM Grantham": 1,
            "Morleys Bexleyheath": 4,
            "Aylesbury": 4
        }
        avg_lat = sum([towns_data[town][0] for town in towns_data]) / len(towns_data)
        avg_lon = sum([towns_data[town][1] for town in towns_data]) / len(towns_data)

        # Create a list of latitudes, longitudes, and weights
        locations_with_weights = [(towns_data[town][0], towns_data[town][1], weight) for town, weight in weights.items()]

        # Create the plotly heatmap map
        fig = px.density_mapbox(
            locations_with_weights,
            lat=[item[0] for item in locations_with_weights],
            lon=[item[1] for item in locations_with_weights],
            z=[item[2] for item in locations_with_weights],
            color_continuous_scale="Viridis",
            mapbox_style="carto-positron",
            title="Heatmap of IBT Activity Across The UK",
            center={"lat": avg_lat, "lon": avg_lon},
            zoom=4
        )

        fig.update_layout(width=560,height=350,margin=dict(l=40, r=40, t=40, b=40))
        return fig.to_html(full_html=False)