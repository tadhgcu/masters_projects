
# coding: utf-8

# # Section 1 - Question 1
# 
# This exercise must be done with pen and paper (or with pen and Word) as you have
# to indicate the calculations you have done.
# 1. Represent the following directed graph by means of (a) and adjacency matrix and (b)
# and adjacency list.
(a)	Adjacency Matrix:
 	1	2	3	4	5	6	7	8	9	10
1	0	1	0	0	1	0	0	0	0	1
2	0	0	0	0	0	0	0	0	0	0
3	0	0	0	1	0	0	0	0	0	0
4	0	1	0	0	1	0	0	0	0	0
5	1	0	0	0	0	0	0	0	0	0
6	0	0	0	0	0	0	0	0	0	1
7	0	0	0	0	0	1	0	1	1	0
8	0	0	0	0	0	0	0	0	0	0
9	0	0	0	0	0	0	0	1	0	0
10	1	0	0	0	0	1	0	0	0	0
(b)	Adjacency list:
1 - 2 - 5 - 10
2
3 - 4 
4 - 2 - 5
5 - 1
6 - 10
7 - 6 - 8 - 9
8 
9 - 8
10 - 1 - 6
# # Section 1 - Question 2
# <font color='black'>Reply the following questions</font>  
# <br>
#     a. Is it weighted?<br>
#     No, this is not weighted. <br><br>
#     b. Is it connected?<br>
#     Yes, it is connected via an articulation point at node 10<br><br>
#     c. Is it weakly connected?<br>
#     This is a weakly connected graph. Each node cannot be reached from any other node.<br><br>
#     d. Which are its order and size?<br>Order = 10, Size = 12<br><br>
#     e. Has it any articulation point. If so, indicate such point.<br>Has it any articulation point. If so, indicate such point.<br><br>
#     f. Has it loops?<br>No, it has no loops<br><br>
#     g. Has it cycles? If so, indicate which.<br>No, it has no cycles<br><br>
#     h. Does it exists a path between nodes 4 and 7? If so, indicate it and its length.<br>No, there is no path between node 7 and 4<br><br>
#     i. Does it exists a path between nodes 3 y 9? If so, indicate it and its length.<br>Yes, there is a path between node 3 and 9 of length 6. <br> C={ 3 , 4 , 5 , 1 , 10 , 6 , 9 }<br><br>
# 
# 

# # Section 1 - Question 3
# <font color='black'>Consider now the graph as a undirected graph</font>  
# <br>
# a. Has it cycles? If so, indicate which.<br>Yes, it has 4 cycles.<br>
# C1 = {4,2,1,5,4}<br>
# C2 = {6,9,7,6}<br>
# C3 = {6,9,8,7,6}<br>
# C4 = {9,8,7,9}<br><br>
# b. Indicate the highest k for which a k-core exist.<br>
# K=2<br><br>
# c. Compute the clustering index of node 10.<br>
# Cant compute as the neighbouring nodes 1 and 6 share no connections.<br><br>
# d. Compute the characteristic path of node 10.<br>
# Answer = 2.911
# <br><br>
# e. Does it exists a clique with order higher than 2 ? If so, indicate the nodes in the
# clique.<br>Yes<br> C = {9,8,7}<br> C = {6,9,7}<br>

# # Section 2
# 
# <br><font color='blue'>1. Download the CaernoElegans-LC_uw.txt graph from Moodle, the graph contains a protein interaction network corresponding to the Caernobidis Elegans worm.</font> 
# 
# <br><font color='blue'>2. The file that contains the network is in edge list format, therefore, load the graph in a variable G_CE using the function read_edgelist ("CL-LC_uw.txt").</font>
# <br>
# <br>
#  

# In[1]:


import networkx as nx
G_CE = nx.read_edgelist ("CaernoElegans-LC_uw.txt")


# <br><font color='blue'>3. Obtain:</font><br> 
# a) print the order<br> b) print the size of the graph by the output <br>c) find out whether
# the graph is directed or not.<br> d) Is it a dense or sparse graph?

# In[2]:


order = nx.number_of_nodes(G_CE) #a
size = nx.number_of_edges(G_CE) #b
graph_type = nx.is_directed(G_CE) #c
density = nx.density(G_CE) #d

print("The order of the graph is ",order, "and the size of the graph is ", size)
print("")
print("Is the graph directed (Ture=Yes, False=No) = ", graph_type)
print("")
print("The density of the graph is ",density)


# <font color='blue'>4. Create a random graph G_AL that has the same order and size as the graph that you just loaded using the function gnm_random_graph (n, m).</font>  

# In[3]:


n = nx.number_of_nodes(G_CE)
m = nx.number_of_edges(G_CE)
random = nx.gnm_random_graph(n, m)


# <font color='blue'>5. Tell if both graphs are connected..</font>  

# In[4]:


connectedCE = nx.is_connected(G_CE)
connectedR = nx.is_connected(random)

print("Is G_CE graph connected?",connectedCE)
print("Is random graph connected?",connectedR)


# <font color='blue'> 6. Compute the number of connected components of each graph. </font>  

# In[5]:


componentsCE = nx.number_connected_components(G_CE)
componentsR = nx.number_connected_components(random)
print("The number of connected components in G_CE are ",componentsCE)
print("The number of connected components in random are ",componentsR)


# <font color='blue'>7. What is the node with the highest grade in each graph?</font> 

# In[6]:


max_degree = 0
node = ""

for n, d in G_CE.degree:
    if d > max_degree:
        max_degree = d
        node = n
        
print("The node ",node, "has the highest degree in G_CE graph with a value of ",max_degree)

degreeR = nx.degree(random)
max_degreeR = 0
nodeR = ""

for node, degree in degreeR:
    if degree > max_degreeR:
        max_degreeR = degree
        nodeR = node
        
print("The node ",node, "has the highest degree in random graph with a value of ",max_degree)


# <font color='blue'>8. What is the node with the highest betweeness?</font> 

# In[7]:


import operator
betweenCE = nx.load_centrality(G_CE)
betweenR = nx.load_centrality(random)
max_betweenCE = max(betweenCE.items(), key=operator.itemgetter(1))
max_betweenR = max(betweenR.items(), key=operator.itemgetter(1))

print("The node ",max_betweenCE[0], "has the highest betweeness in G_CE graph with a value of ",max_betweenCE[1])
print("The node ",max_betweenR[0], "has the highest betweeness in random graph with a value of ",max_betweenR[1])


# <font color='blue'>9. What is the node with the highest closeness?</font> 

# In[8]:


closeCE = nx.closeness_centrality(G_CE)
closeR = nx.closeness_centrality(random)
max_closeCE = max(closeCE.items(), key=operator.itemgetter(1))
max_closeR = max(closeR.items(), key=operator.itemgetter(1))

print("The node ",max_closeCE[0], "has the highest closeness in G_CE graph with a value of ",max_closeCE[1])
print("The node ",max_closeR[0], "has the highest closeness in random graph with a value of ",max_closeR[1])


# <font color='blue'>10. What is the node with the highest distance between two nodes in the graph (graph
# diameter)?</font> 

# In[9]:


try:
    diameter = nx.diameter(G_CE)
    print("The diameter of G_CE is ",diameter)
except:
  print("As the graph G_CE is not connected, the path length is infinite")

try:
    diameter = nx.diameter(random)
    print("The diameter of the random graph is ",diameter)
except:
  print("As the random graph is not connected, the path length is infinite")


# # Section 3: Node Degree Distribution

# <font color='blue'>1. Plot the node degree distribution of both graphs.</font> 
# <br>

# In[19]:


import matplotlib.pyplot as plt
#Random graph
R=nx.degree_histogram(random)
plt.bar(range(len(R)),R, width=5, color='b')
plt.title("1 - Degree Histogram for Random graph")
plt.ylabel("Count")
plt.xlabel("Degree")
plt.show()

#G_CE graph
L=nx.degree_histogram(G_CE)
plt.bar(range(len(L)),L, width=5, color='b')
plt.title("2 - Degree Histogram for CE graph")
plt.ylabel("Count")
plt.xlabel("Degree")
plt.show()


# <font color='blue'>2. Are the node degree distribution of both graphs the same?, What conclusion do you draw from the above?</font> 
# <br><br>
# The degree distributions of both graphs are different.
# <br><br>
# Graph 1 (the random network) 
# <br>This graph is what we would typically expect to see when we graph a random network of this size. In these simple types of networks, the nodes we find should have similar degrees. This can been clearly seen in graph 1.
# <br><br>
# Graph 2 (G_CE netwrok)
# <br>This graphhas a distribution more similar to a real world netwrok. In a real world network, most nodes have a relatively small degree, but a few nodes will have very large degree. These large nodes will cause the average degree to be larger. Such a degree distribution is said to have a long tail.

# <font color='blue'>3. Now draw the node degree distribution of the protein interaction network using logarithmic scale in both axes,</font> 
# <br>
# Add these two lines of code to change the type of scale in each axis
# <br>plt.xscale("log", nonposx='clip')
# <br>plt.yscale("log", nonposy='clip')

# In[20]:


#random graph
R=nx.degree_histogram(random)
plt.bar(range(len(R)),R, width=5, color='b')
plt.xscale("log", nonposx='clip')
plt.yscale("log", nonposy='clip')
plt.title("1 -  - Log Degree Histogram for Random graph")
plt.ylabel("Count")
plt.xlabel("Degree")
plt.show()


#G_CE graph
L=nx.degree_histogram(G_CE)
plt.bar(range(len(L)),L, width=5, color='b')
plt.xscale("log", nonposx='clip')
plt.yscale("log", nonposy='clip')
plt.title("2 - Log Degree Histogram for CE graph")
plt.ylabel("Count")
plt.xlabel("Degree")
plt.show()


# <font color='blue'>4. What kind of plot do you get? Could you roughly calculate the slope of the data?</font> 
# <br>
# <br>
# This is a log scale graph. They both appear to be more similar here compared to before however, plot 2 had additional points on the graph with gaps between node degree distribution points

# # Section 4: 
# Visualization of the graphs (do not expect wonders in this section, NetworkX is not very powerful nor very easy to use drawing).
#     
# 1. Plot both graphs. Do you observe any differences between them?
# 
# 2. Choose a layout where the difference between the graphs can be clearly observed.
# 
# 3. Export both graph to a Gephy compatible file format and plot the graphs in Gephy
# following these steps:
# 
#     1. Download the Java file from Moodle and install it. Export both graphs in GexF
# format using write_gexf (G, path).
# 
# 2. Load the file with Gephi indicating that it is a directed graph.
# 
# 3. In the lower panel on the left side, where you put distribution (layout), select some
# distribution (in the case of the protein network I recommend you choose
# Fruchterman Reingold) and press the Execute button. Observe what happens.

# In[14]:


nx.draw_spring(random)


# In[16]:


print("G_CE network plot",nx.draw_spring(G_CE))


# In[17]:


#write the newtroks to a file and open in gephy
nx.write_gexf(G_CE,"G_CE.gexf")
nx.write_gexf(random,"random_network.gexf")


# In[18]:


from IPython.display import Image, display

listOfImageNames = ['random_network.PNG',
                   'G_CE.png']

#Printing the graphs
count=0
num=1
for imageName in listOfImageNames:
    print("Image",num,"from graph -",listOfImageNames[count])
    display(Image(filename=imageName))
    count= count+1
    num = num+1


# The first image is the random network. Here you can clearly see the graph is more dense in the center but as you move to the outside of the graph, there are more unconnected nodes. Compare this to the graph of G_CE, you can see there are two highly connected nodes on the edge of the graph. One of these is not connected to the main graph.<br>
# When you compare the output from gephy to Networkx, a difference can be seen between the two graphs. The two highly connected nodes from the G_CE graph cannot be easily distinguished using the visualization tools of Networkx but can be using gephy
# 
