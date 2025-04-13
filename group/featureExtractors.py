# featureExtractors.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
#
# This code has been modified and extended for CS 1820 at Harvard University, 
# with adjustments tailored to align with the course curriculum and objectives.

"Feature extractors for Pacman game states"

from helpers.game import Directions, Actions
import util

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        actions = [Directions.NORTH, Directions.EAST, Directions.SOUTH, Directions.WEST]
        for a in actions:
            dx, dy = Actions.directionToVector(a)
            nbr_x, nbr_y = int(pos_x + dx), int(pos_y + dy)
            # skip if neighbour is a wall
            if walls[nbr_x][nbr_y]:
                continue
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None



class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter() # Extension of the dictionary class, where all keys are defaulted to have value 0

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features
  
class CustomExtractor(FeatureExtractor):
    """
    Write your own Custom Feature Extractor
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = [] # positions of ghosts that can eat Pacman
        scaredGhosts = [] # positions of scared ghosts that Pacman can eat
        scaredTimes = [] # number of moves each scared ghost is scared for
        for ghost in state.getGhostStates(): 
            if ghost.scaredTimer > 0: 
                scaredGhosts.append(ghost.getPosition())
                scaredTimes.append(ghost.scaredTimer)
            else:
                ghosts.append(ghost.getPosition())

        features = util.Counter()
        features["bias"] = 1.0

        # get pacman's current position and next position after taking action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        
        # 1: basic food 
        # encourage eating food
        if food[next_x][next_y]:
            features["eats-food"] = 1.0
            
        # 2: distance to closest food
        dist_to_food = closestFood((next_x, next_y), food, walls)
        if dist_to_food is not None:
            # normalize distance
            features["closest-food"] = float(dist_to_food) / (walls.width * walls.height)
        
        # 3: dangerous ghosts features
        # count the number of dangerous ghosts 1-step away (avoid them)
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
        
        # 4: distance to closest dangerous ghost (avoid them)
        ghost_distances = [util.manhattanDistance((next_x, next_y), g) for g in ghosts]
        if ghost_distances and min(ghost_distances) < 3: 
            features["closest-ghost"] = 1.0 / (min(ghost_distances) + 0.1)  # remember inverse distance (higher = closer = worse)
            
        # 5: scared ghosts features - chase them to get reward
        # distance to closest scared ghost ----------- prioritize chasing scared ghosts
        scared_distances = []
        for i, sg in enumerate(scaredGhosts):
            dist = util.manhattanDistance((next_x, next_y), sg)
            # weight by remaining scared time - prioritize ghosts with less time left
            if dist < scaredTimes[i]: 
                scared_distances.append((dist, scaredTimes[i]))
                
        if scared_distances:
            # sort by distance (ascending)
            scared_distances.sort(key=lambda x: x[0])
            min_dist, time_left = scared_distances[0]
            
            # stronger incentive to chase scared ghosts with less time left
            features["chase-scared-ghost"] = 2.0 / (min_dist + 0.1) * (1.0 / (time_left + 0.1))
            
            # if next move eats a scared ghost, give high reward
            if any((next_x, next_y) == sg for sg in scaredGhosts):
                features["eats-scared-ghost"] = 5.0
                
        # 6: avoid going to dead ends when ghosts nearby
        if ghost_distances and min(ghost_distances) < 5: 
            num_escape_routes = sum(not walls[next_x + dx][next_y + dy] 
                                   for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)])
            if num_escape_routes <= 1:  
                features["dead-end"] = 1.0
        
        # normalize to prevent crazily large values --- Thank you ChatGPT!
        features.divideAll(10.0)
        return features