# Algorithms-for-collaborative-logistics-services
A Python Implementation of a heuristic-based Solution to Pickup and Delivery Problem with Time Windows and WDP 

## Table of contents
* [Project Summary](#Project-Summary)
* [Requirements](#Requirements)
* [Authors](#Authors)

## Project Summary
This repo is part of the final project in Software Engineering degree at Sami Shamoon College of Engineering.
The aim of the project is to design a digital platform for logistic services based on crowd-sourced and collaborative shipping. The platform allows shippers to enter their shipping details, while potential carriers can choose sub-routes to fulfill those shipping requests. The main focus of the project was to develop an algorithm that finds the optimal order of couriers, minimizing the total cost while satisfying all constraints.

The problem was decomposed into two sub-problems: the winner determination problem and the simultaneous pickup and delivery problem with time windows. Since the problem involves considering different combinations of couriers and time windows, it is classified as an NP-complete problem. Therefore, efficient heuristics are necessary to solve it effectively.

Various methods were reviewed during the development phase, with the *IDA algorithm and a genetic algorithm identified as the most suitable approaches. A bidding graph was defined to represent a valid sequence of carriers, and simulations were conducted using different algorithms. The project also explored different crossover operators, heuristics, and parameters.

The results of the simulations were presented and analyzed. The *IDA-based algorithm demonstrated better time efficiency and lower space efficiency compared to Dijkstra's algorithm. On the other hand, the genetic algorithm yielded near-optimal solutions with superior time efficiency, making it suitable for large-scale applications.

## Requirements
Make sure you have installed all of the following prerequisites on your development machine:

* Git - [Download & Install Git](https://git-scm.com/downloads). OSX and Linux machines typically have this already installed.

The project is created with:
*  Python 3.9
* Pip
* memory_profiler
  
## Authors

* Lihi Sabag
* Bar Sela	

