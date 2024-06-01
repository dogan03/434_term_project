# Autonomous Navigation in a Virtual Factory Environment

## Introduction

This project aims to design an autonomously navigating car within a randomly generated factory environment. The primary focus is on efficient path planning and robust obstacle avoidance to ensure the robot reaches its destination swiftly and safely.

## Architecture of the Project

### Path Planning

The path planning component of the project utilizes the A-star algorithm, a popular choice for its efficiency in finding the shortest path in a grid-based map. Given the constrained environment of the factory environment, A-star is particularly effective. The algorithm iteratively selects a goal position that is one meter away from the robot towards the final destination.

### Following the Path

To follow the planned path, the project employs the Dynamic Window Approach (DWA) algorithm. DWA is adept at handling both static and dynamic obstacles by dynamically adjusting the robot's steering angle. This ensures smooth navigation even in the presence of moving obstacles.

### Decision Tree

A decision tree is integrated to manage the robot's behavior under various edge cases, optimizing its performance and ensuring timely arrival at the final destination. The decision tree helps in making critical decisions such as speed adjustments and sharp turns, enhancing the overall robustness of the navigation system.

## Problems Faced and Their Solutions

### Path Planning

#### Problems

- Time complexity being high.

#### Solutions

- To solve the time complexity problem, I chose the grid size to be as high as it could be to fit the corridor around 1.

### Following the Path

#### Problems

- Computational Burden of DWA
- Slow Velocity
- Steering edge cases

#### Solutions

- To make the algorithm faster I have removed the velocity part from the algorithm. Thus, it only calculates to goal and obstacle costs. (O(n^2) -> O(n))
- Designed a decision tree to control the velocity of the robot. This way I was able to make robot move faster when theres no obstacles around.
- I also handled the steering edge cases with the decision tree to make it robust to the environments where it has to turn sharply.

# Illustrations

## Robot dodging the dynamic object

![Alt text](/videos/dodge_gif.gif)

## Robot being fast in a wide room

![Alt text](/videos/fast_gif.gif)

## Decision Tree

![Alt text](/images/decision_tree.png)

## Robot at the Final Destination

![Alt text](/images/final_destination.png)
