# Data Camp Chapter 1
# addition subtraction
print(5 + 5)
print(5 - 5)
# multiplication
print(5 * 3)
print(5 ** 5)
print(100 * 1.1 ** 7)

# create variables
savings = 100
interest = 1.1
years = 7
result = savings * interest ** years
print(result)

# types
profitable = True
intereststr = "interest"
interestrate = 1.10349394839929
print(type(profitable))
print(type(intereststr))
print(type(interestrate))


def add(num1, num2):
    return num1 + num2


def addition():
    print(add(5, 5))
    power = (add(15, 5))


addition()

areas = ["living Room", 20, "Bathroom", 2.3]
print(areas[2])

# changing and deleting functions
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]
del (areas[-4:-2])
print(areas)

# create a copy of a list without it affecting the original list
areas_copy = (areas[:])

first = [11.25, 18.0, 20.0]
second = [10.75, 9.50]
both = first + second
both_sort = sorted(both, reverse=True)

# print(areas.upper())

# string to experiment with: room
room = "poolhouse"

# Use upper() on room: room_up
room_up = (room.upper())

# Print out room and room_up
print(room, room_up)

# Print out the number of o's in room
room.count("o")
print(room.count("o"))

# Create list areas
areas2 = [11.25, 18.0, 20.0, 10.75, 9.50]

# Print out the index of the element 20.0
print(areas2.index(20.0))

# Print out how often 14.5 appears in areas
print(areas2.count(14.5))

# Use append twice to add poolhouse and garage size
areas2.append(24.5)
areas2.append(15.45)

# Print out areas
print(areas2)

# Reverse the orders of the elements in areas
areas2.reverse()

# Print out areas
print(areas2)

# Definition of radius
r = 0.43

# Import the math package
import math

# Calculate C
C = 2 * math.pi * r

# Calculate A
A = math.pi * (r ** 2)

# Build printout
print("Circumference: " + str(C))
print("Area: " + str(A))

# height is available as a regular list
height = [74, 75, 77, 71, 78]

import numpy

# Create a numpy array from height: np_height
numpy_height = numpy.array(height)

# Print out np_height
print(numpy_height)

# Convert np_height to m: np_height_m
numpy_height_m = numpy_height * 0.0254

# Print np_height_m
print(numpy_height_m)

# Create baseball, a list of lists
baseball = [[180, 78.4],
            [215, 102.7],
            [210, 98.5],
            [188, 75.2]]

# Create a 2D numpy array from baseball: np_baseball
np_baseball = numpy.array(baseball)

# Print out the type of np_baseball
print(type(np_baseball))

# Print out the shape of np_baseball
print(np_baseball.shape)

# Print out the 50th row of np_baseball
print(np_baseball[49, :])

# Select the entire second column of np_baseball: np_weight
np_weight = np_baseball[:, 1]

# Print out height of 124th player
print(np_baseball[123, 0])

# Create np_baseball (3 cols)
np_baseball = numpy.array(baseball)

# Print out addition of np_baseball and updated
print(np_baseball)

# Create numpy array: conversion
conversion = numpy.array([0.0254, 0.453592, 1])

# Print out product of np_baseball and conversion
print(np_baseball * conversion)

height = numpy.round(numpy.random.normal(1.75, 0.20, 5000), 2)
weight = numpy.round(numpy.random.normal(60.32, 15, 5000), 2)
np_city = numpy.column_stack((height, weight))

print(numpy.median(np_city[:, 0]))
print(numpy.mean(np_city[:, 0]))

# Print mean height (first column)
avg = numpy.mean(np_city[:, 0])
print("Average: " + str(avg))

# Print median height.
med = numpy.median(np_city[:, 0])
print("Median: " + str(med))

# Print out the standard deviation on height.
stddev = numpy.std(np_city[:, 0])
print("Standard Deviation: " + str(stddev))

corr = numpy.corrcoef(np_city[:, 1], np_city[:, 0])
print("Correlation: " + str(corr))

# new_height = numpy.array(np_city[np_city == "quote"])

