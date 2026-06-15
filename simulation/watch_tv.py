from utils import *
from action_skills import *

controller.reset('FloorPlan229')
step(dict(action='Initialize', gridSize=0.25, agentCount=2))



def human_phase():

    occupied_robot = step(dict(action="Pass", agentId=1)).metadata['agent']['position']
    event = step(dict(action="Pass", agentId=0))

    start = event.metadata['agent']['position']

    event = step(dict(action="Pass", agentId=0))

    tv = next((o for o in event.metadata['objects']
               if o['objectType'] == 'Television'), None)

    if tv:
        move(start, tv['position'], occupied_robot, 0)
        step(dict(action="ToggleObjectOn", objectId=tv['objectId'], agentId=0))

    sofa = next((o for o in event.metadata['objects']
               if o['objectType'] == 'Sofa'), None)
    
    start = event.metadata['agent']['position']
    event = step(dict(action="Pass", agentId=0))
    
    if sofa:
        move(start, sofa['position'], occupied_robot, 0)



def robot_phase():

    occupied_human = step(dict(action="Pass", agentId=0)).metadata['agent']['position']
    event = step(dict(action="Pass", agentId=1))

    start = event.metadata['agent']['position']

    tissue = next((o for o in event.metadata['objects']
                   if o['objectType'] == 'TissueBox'), None)

    table = next((o for o in event.metadata['objects']
                  if o['objectType'] == 'CoffeeTable'), None)

    if tissue:
        move(start, tissue['position'], occupied_human, 1)
        pickup_object(tissue['objectId'], 1)
    
    event = step(dict(action="Pass", agentId=1))
    start = event.metadata['agent']['position']

    if tissue and table:
        move(start, table['position'], occupied_human, 1)
        place_object(tissue['objectId'], table['objectId'], 1)



if __name__ == "__main__":

    human_phase()
    robot_phase()

    save_videos()
    print_stats()