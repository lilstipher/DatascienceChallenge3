# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 16:09:11 2017

@author: cbothore
"""


import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import pylab
import numpy as np
import pickle
from collections import Counter
from scipy.stats.stats import pearsonr
import nltk
from prettytable import PrettyTable
import community
from collections import defaultdict

def naive_method(graph, empty, attr):
    """   Predict the missing attribute with a simple but effective
    relational classifier. 
    
    The assumption is that two connected nodes are 
    likely to share the same attribute value. Here we chose the most frequently
    used attribute by the neighbors
    
    Parameters
    ----------
    graph : graph
       A networkx graph
    empty : list
       The nodes with empty attributes 
    attr : dict 
       A dict of attributes, either location, employer or college attributes. 
       key is a node, value is a list of attribute values.

    Returns
    -------
    predicted_values : dict 
       A dict of attributes, either location, employer or college attributes. 
       key is a node (from empty), value is a list of attribute values. Here 
       only 1 value in the list.
     """
    nbrs_attr_values=[] 
    predicted_values={}
    for n in empty:
        for nbr in graph.neighbors(n):
            if nbr in attr:
                for val in attr[nbr]:
                    nbrs_attr_values.append(val)
        predicted_values[n]=[]
        if nbrs_attr_values: # non empty list
            # count the number of occurrence each value and returns a dict
            cpt=Counter(nbrs_attr_values)
            # take the most represented attribute value among neighbors
            a,nb_occurrence=max(cpt.items(), key=lambda t: t[1])
            predicted_values[n].append(a)
    return predicted_values
    
 
def evaluation_accuracy(groundtruth, pred):
    """    Compute the accuracy of your model.

     The accuracy is the proportion of true results.

    Parameters
    ----------
    groundtruth :  : dict 
       A dict of attributes, either location, employer or college attributes. 
       key is a node, value is a list of attribute values.
    pred : dict 
       A dict of attributes, either location, employer or college attributes. 
       key is a node, value is a list of attribute values. 

    Returns
    -------
    out : float
       Accuracy.
    """
    true_positive_prediction=0   
    for p_key, p_value in pred.items():
        if p_key in groundtruth:
            # if prediction is no attribute values, e.g. [] and so is the groundtruth
            # May happen
            if not p_value and not groundtruth[p_key]:
                true_positive_prediction+=1
            # counts the number of good prediction for node p_key
            # here len(p_value)=1 but we could have tried to predict more values
            true_positive_prediction += len([c for c in p_value if c in groundtruth[p_key]])          
        # no else, should not happen: train and test datasets are consistent
    return true_positive_prediction/len(pred)
   

# load the graph
G = nx.read_gexf("mediumLinkedin.gexf")
print("Nb of users in our graph: %d" % len(G))

# load the profiles. 3 files for each type of attribute
# Some nodes in G have no attributes
# Some nodes may have 1 attribute 'location'
# Some nodes may have 1 or more 'colleges' or 'employers', so we
# use dictionaries to store the attributes
college={}
location={}
employer={}
# The dictionaries are loaded as dictionaries from the disk (see pickle in Python doc)
with open('mediumCollege_60percent_of_empty_profile.pickle', 'rb') as handle:
    college = pickle.load(handle)
with open('mediumLocation_60percent_of_empty_profile.pickle', 'rb') as handle:
    location = pickle.load(handle)
with open('mediumEmployer_60percent_of_empty_profile.pickle', 'rb') as handle:
    employer = pickle.load(handle)

print("--------------------------college stats--------------------------")
print("Nb of users with one or more attribute college: %d" % len(college))
print("Fraction revelated: %f %s" % (((len(college)/len(G))*100),"%"))

print(" \n --------------------------location stats--------------------------")

print("Nb of users with one or more attribute location: %d" % len(location))
print("Fraction revelated: %f %s" % (((len(location)/len(G))*100),"%"))

print("\n--------------------------employer stats--------------------------")

print("Nb of users with one or more attribute employer: %d" % len(employer))
print("Fraction revelated: %f %s" % (((len(employer)/len(G))*100),"%"))

# here are the empty nodes for whom your challenge is to find the profiles
empty_nodes=[]
with open('mediumRemovedNodes_60percent_of_empty_profile.pickle', 'rb') as handle:
    empty_nodes = pickle.load(handle)
print("\n Your mission, find attributes to %d users with empty profile" % len(empty_nodes))


# --------------------- Baseline method -------------------------------------#
# Try a naive method to predict attribute
# This will be a baseline method for you, i.e. you will compare your performance
# with this method
# Let's try with the attribute 'employer'
employer_predictions=naive_method(G, empty_nodes, employer)
groundtruth_employer={}
with open('mediumEmployer.pickle', 'rb') as handle:
    groundtruth_employer = pickle.load(handle)
result=evaluation_accuracy(groundtruth_employer,employer_predictions)
print("%f%% of the predictions are true" % (result*100))
print("Very poor result!!! Try to better!!!!")
i=0
#for n,nbrs in G.adjacency():
 #   i+=1
 #   for nbr in nbrs:
  #      if n in employer and nbr in employer:
   #         print(employer[nbr])
#print(i)
# --------------------- Now your turn -------------------------------------#
# Explore, implement your strategy to fill empty profiles of empty_nodes


# and compare with the ground truth (what you should have predicted)
# user precision and recall measures
count={}
peronSamelocation=0
i=0
urbana_champaign={}
countCollege={}
countEmployer={}
for n in location:#location count
    for p in location[n]:
                count[p] = count.get(p, 0) + 1
#print(count)
for n in college:#college count
    for p in college[n]:
        countCollege[p] = countCollege.get(p, 0) + 1

for n in employer:#employer count
    for p in employer[n]:
        countEmployer[p] = countEmployer.get(p, 0) + 1






#"############################################"""
for n in location:
        if "san francisco bay area" in location[n] and n in college:
            i+=1
            urbana_champaign[n]=college[n]
            #for p in urbana_champaign [n]:
                #count[p] = count.get(p, 0) + 1
#print ("number of person is %d" %i)
#for p in count :
    #print("University : %s, occurrence = %d" %(p,count[p]))

for n in count:
    if count[n] >1:
        peronSamelocation+=count[n]
        print("location : %s, occurrence = %d" %(n,count[n]))
#print(i)
#print(Counter(location.items()))
#print(nx.degree_pearson_correlation_coefficient(G,nodes=employer))
#print(nx.numeric_assortativity_coefficient(G, "location", nodes=location))
#plt.hist(nx.degree_histogram(G))
#plt.show()
#print(G)


collegeTab={}
for n in count :
    if count[n] >2 :
        collegeTab[n]=count[n]

#location tab
x = PrettyTable()
x.field_names = ["Location", "Count"]
for n in collegeTab:
    x.add_row([n,collegeTab[n]])
x.sortby = "Count"
x.reversesort=True
x.get_string(start=1,end=4)
print(x)
#college tab

collegeTab={}
for n in countCollege :
    if countCollege[n] >1 :
        collegeTab[n]=countCollege[n]

x = PrettyTable()
x.field_names = ["college", "Count"]
for n in collegeTab:
    x.add_row([n,collegeTab[n]])
x.sortby = "Count"
x.reversesort=True
x.get_string(start=1,end=4)
print(x)

#emplyer tab
employerTab={}
for n in countEmployer :
    if countEmployer[n] >2 :
        employerTab[n]=countEmployer[n]

x = PrettyTable()
x.field_names = ["employer", "Count"]
for n in employerTab:
    x.add_row([n,employerTab[n]])
x.sortby = "Count"
x.reversesort=True
x.get_string(start=1,end=4)
print(x)


similar_neighbors=0
total_number_neighbors=0 # to verify the number of edges ;-)!!!
for n,nbrs in G.adjacency():
    for nbr in nbrs:
        if n in employer and nbr in employer:
            total_number_neighbors+=1
            if len([val for val in employer[n] if val in employer[nbr]]) > 0:
                similar_neighbors+=1
homophily=similar_neighbors/total_number_neighbors
print(homophily)
 # top 10
#from prettytable import PrettyTable
#f#or label, data in (('college', collegeTab)):
#    pt = PrettyTable(field_names=[label, 'Count'])
 #   c = Counter(data)
 #   [ pt.add_row(kv) for kv in c.most_common() ]
#    pt.align[label], pt.align['Count'] = 'l', 'r' # Set column alignment
 #   print(pt)
 ##################""correlation"""#########################################"""
count={}
urbana_champaign={}
i=0
for n in location:
        if "urbana-champaign illinois area" in location[n] and n in employer:
            i+=1
            urbana_champaign[n]=employer[n]
            for p in urbana_champaign [n]:
                count[p] = count.get(p, 0) + 1
print ("number of person is %d" %i)


#Urbana corell table

x = PrettyTable()
x.field_names = ["Employer of people who live at Urbana", "Count"]
for n in count:
    x.add_row([n,count[n]])
x.sortby = "Count"
x.reversesort=True
print(x)


#first compute the best partition
spring_pos = nx.spring_layout(G)
partition = community.best_partition(G)

#print(partition)
#dendo = community.generate_dendrogram(G)
#print(dendo)
values = [partition.get(node) for node in G.nodes()]
#print(values)
list_nodes=[]
plt.axis("off")
nx.draw_networkx(G, pos = spring_pos, cmap = plt.get_cmap("jet"), node_color = values, node_size = 25, with_labels = False)
plt.show()


college={}
location={}
employer={}
# The dictionaries are loaded as dictionaries from the disk (see pickle in Python doc)
with open('mediumCollege_60percent_of_empty_profile.pickle', 'rb') as handle:
    college = pickle.load(handle)
with open('mediumLocation_60percent_of_empty_profile.pickle', 'rb') as handle:
    location = pickle.load(handle)
with open('mediumEmployer_60percent_of_empty_profile.pickle', 'rb') as handle:
    employer = pickle.load(handle)

def new_method(graph, empty, attr):
    
    nbrs_attr_values=[] 
    predicted_values={}
    for n in empty:
        for nbr in graph.neighbors(n):
            if nbr in attr :
                for val in attr[nbr]:
                    nbrs_attr_values.append(val)
        predicted_values[n]=[]
        if nbrs_attr_values: # non empty list
            # count the number of occurrence each value and returns a dict
            cpt=Counter(nbrs_attr_values)
            # take the most represented attribute value among neighbors
            a,nb_occurrence=max(cpt.items(), key=lambda t: t[1])
            predicted_values[n].append(a)
    return predicted_values

empty_nodes=[]
with open('mediumRemovedNodes_60percent_of_empty_profile.pickle', 'rb') as handle:
    empty_nodes = pickle.load(handle)

employer_predictions=new_method(G, empty_nodes, employer)
groundtruth_employer={}
with open('mediumEmployer.pickle', 'rb') as handle:
    groundtruth_employer = pickle.load(handle)
result=evaluation_accuracy(groundtruth_employer,employer_predictions)
print("%f%% of the predictions are true" % (result*100))
print("Very poor result!!! Try to better!!!!")
i