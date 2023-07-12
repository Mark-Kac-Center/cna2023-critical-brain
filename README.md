# cna2023-critical-brain

Repository accompanying the workshops on [Brain's Criticality](https://cna2023.ift.uj.edu.pl/documents/152426817/153646757/GRELA_JANAREK_Abstract/224aa57e-aeee-438b-b836-591c6ac939ee)
at [CNA2023](https://cna2023.ift.uj.edu.pl/start).


## Hands on brain's criticality
In this workshop, we will delve into the fascinating phenomenon of brain criticality. 
Our objective is to provide students with a comprehensive understanding of both theoretical 
and numerical aspects concerning human brain activity and criticality.

After a concise introductory lecture, we will engage in simulating cortical activity utilizing 
the Haimovici et al. [1] model, which employs a healthy human brain [2]. This model, based on 
a straightforward cellular automaton, accurately reproduces the activity of cortical regions 
of interest (ROI) through a network of weighted connections, known as the connectome. It exhibits 
a critical transition, where the dynamics of the model closely mirror empirical human brain 
activity, including correlations and resting-state networks. The purpose of this workshop is to 
familiarize students with brain activity modeling, criticality, and analysis of simulations.

Throughout the workshop, we will conduct simulations of the model using real human connectome
data and endeavor to identify the critical transition using various methodologies. 
These will encompass activity analysis, examination of clusters of active ROIs, and more.
Additionally, we will compare different measures of criticality and explore intriguing scenarios, 
such as artificial connectomes or injured brains, (e.g. lobotomy just for fun!) 8-)

References:
1. A. Haimovici et al., *Phys. Rev. Lett.* **110**, 178101 (2013)
2. P. Hagmann et al., *PLoS Biology* **6 (7)** (2008)

### Requirements (for standalone usage):
* python3
* numpy
* numba
* matplotlib
* networkx
* scikit-image
