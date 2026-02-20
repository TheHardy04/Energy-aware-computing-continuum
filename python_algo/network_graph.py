import networkx as nx
import matplotlib.pyplot as plt
import random

# Create a new graph
G = nx.Graph()

# Add nodes with CPU, RAM, and type attributes
# Let's assume CPU is in cores and RAM is in GB
G.add_node("IoT_Device", cpu=1, ram=2, type='edge')
G.add_node("Gateway", cpu=2, ram=8, type='edge')
G.add_node("Fog_Server_1", cpu=8, ram=32, type='fog')
G.add_node("Fog_Server_2", cpu=4, ram=16, type='fog')
G.add_node("Cloud_Server", cpu=16, ram=64, type='cloud')
G.add_node("Cloud_DataCenter", cpu=128, ram=512, type='cloud')


# Add edges with Bandwidth (in Gbps) and Latency (in ms) attributes
G.add_edge("IoT_Device", "Gateway", bw=0.1, latency=2)
G.add_edge("Gateway", "Fog_Server_1", bw=1, latency=20)
G.add_edge("Gateway", "Fog_Server_2", bw=1, latency=20)
G.add_edge("Fog_Server_1", "Fog_Server_2", bw=5, latency=10)
G.add_edge("Fog_Server_1", "Cloud_Server", bw=10, latency=50)
G.add_edge("Cloud_Server", "Cloud_DataCenter", bw=40, latency=5)


# --- Visualization ---
plt.figure(figsize=(15, 12))

# Define colors for each node type
color_map = {
    'edge': 'lightgreen',
    'fog': 'skyblue',
    'cloud': 'lightcoral'
}
node_colors = [color_map.get(G.nodes[node].get('type', 'default'), 'gray') for node in G.nodes()]

# Use a spring layout for a nice display
pos = nx.spring_layout(G, seed=42, k=0.8)

# Draw the graph
nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=3000, edge_color='gray', width=2, font_size=10)

# Draw node labels (attributes)
node_labels = {node: f"{node}\nCPU: {attrs['cpu']}\nRAM: {attrs['ram']}" for node, attrs in G.nodes(data=True)}
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, verticalalignment='center')


# Draw edge labels (attributes)
edge_labels = {(u, v): f"BW: {attrs['bw']}Gbps\nLat: {attrs['latency']}ms" for u, v, attrs in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

# Add a legend
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'{node_type.capitalize()} Node',
                             markerfacecolor=color, markersize=10) for node_type, color in color_map.items()]
plt.legend(handles=legend_handles, title="Node Types", loc='best')

plt.title("Cloud, Fog, and Edge Network Graph")
plt.show()

print("Graph created and visualization should be displayed.")
print("Nodes:", G.nodes(data=True))
print("Edges:", G.edges(data=True))
