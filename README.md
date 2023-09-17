# Transport system modeling and emission estimation
>This software allows performing simulations of various delivery scenarios within the specified study area,
and using obtained traffic parameters for the transport emissions estimation.

## Technology
The software was implemented using __Python__ programming language and its basic libraries

## Description
### The developed framework consists of two main models:
* <b>transportation system model</b>, which is implemented in the scripts <i>net.py</i>, <i>node.py</i>, <i>link.py</i>, 
    <i>region.py</i>, <i>request.py</i>, <i>route.py</i> and <i>stochastic.py</i>
* <b>emissions calculation model</b> based on the __EMEP/EEA__ methodology, which is implemented in the scripts <i>co2.py</i>,
    <i>exhaust.py</i>, <i>evaporative.py</i> and <i>wear.py</i>

### The traffic simulation module has the following classes:
* <i>Net</i> – the main class, that represents a transport network model
* <i>Node</i> – implements the vertex of a graph, a transport network element representing road intersections, customers or depot points
* <i>Link</i> – implements the edge of a graph, a transport network element representing roads inside the study area
* <i>Region</i> – characterizes the traffic analysis zone within the study area
* <i>Request</i> – describes a customer request for the cargo delivery
* <i>Route</i> – describes a path, a vehicle takes while servicing requests
* <i>Stochastic</i> – implements the random variable generator

## Usage
### To run the simulations, first obtain the input data (the transport network characteristics and the amount of incoming traffic) and then, perform the following steps:
* provide the files with parameters of customers located inside the study area: 
    define the polygon along the perimeter of the study area and the geographic coordinates of its vertices
* provide the road network parameters (use the Net class method <i>load_from_file</i> to read the network information from the text files) 
    and incoming traffic counts (as the dictionary with the entry nodes as keys and traffic counts as values)
* define the probability of delivery request occurrence
    (provided as a dictionary with the types of clients as keys and the probabilities as values)
* instantiate the object of the <i>Net</i> class, read the network data
* generate the set of requests providing the input flows and the probabilities of the request appearance 
    as the arguments of the <i>gen_demand</i> method for the created net object
* for the created _net_ object run the method <i>simulate</i>, providing as the arguments the set of generated requests, the outlets, and the loading points.

## License
> You can check out the full license [here](../master/LICENSE)

This project is licensed under the terms of the MIT license.
