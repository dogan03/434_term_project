# Autonomous Navigation in a Virtual Factory Environment

## Introduction

This project aims to design an autonomously navigating car within a randomly generated factory environment. The primary focus is on efficient path planning and robust obstacle avoidance to ensure the robot reaches its destination swiftly and safely.

## Architecture of the Project

### Path Planning

The path planner is A-star algorithm in the project. Since the dungeon is not so wide it works really well. Then, algorithm selects the goal posiiton to be the one 1 meter away from the robot towards final destination iteratively.

### Following the Path

To follow the path, project relies on the DWA algorithm which handles both static and dynamic obstacles problem.

### Decision Tree

While the robot follows the path, decision tree helps the robot to handle edge cases and reach final destination faster.

## Problems Faced and Their Solutions

### Path Planning

#### Problems

- Time complexity being high.

#### Solutions

- To solve the time complexity problem, I chose the grid size to be as high as it could be to fit the corridor around 1.

### Following the Path

#### Problems

- DWA being computational burden
- Velocity being so slow
- Edge cases steering problem

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
