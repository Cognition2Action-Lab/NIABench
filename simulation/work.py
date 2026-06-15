from utils import *
from action_skills import *

controller.reset('FloorPlan229')
step(dict(action='Initialize', gridSize=0.25, agentCount=2))



def human_phase():

    occupied_robot = step(dict(action="Pass", agentId=1)).metadata['agent']['position']
    event = step(dict(action="Pass", agentId=0))

    start = event.metadata['agent']['position']

    laptop = next((o for o in event.metadata['objects']
                   if o['objectType'] == 'Laptop'), None)
    
    if laptop:
        move(start, laptop['position'], occupied_robot, 0)
        pickup_object(laptop['objectId'], 0)

    desk = next((o for o in event.metadata['objects']
                 if o['objectType'] == 'Desk'), None)
    
    event = step(dict(action="Pass", agentId=0))
    start = event.metadata['agent']['position']

    if laptop and desk:
        move(start, desk['position'], occupied_robot, 0)
        place_object(laptop['objectId'], desk['objectId'], 0)



def robot_phase():

    occupied_human = step(dict(action="Pass", agentId=0)).metadata['agent']['position']
    event = step(dict(action="Pass", agentId=1))

    start = event.metadata['agent']['position']

    phone = next((o for o in event.metadata['objects']
                  if o['objectType'] == 'CellPhone'), None)

    if phone:
        move(start, phone['position'], occupied_human, 1)
        pickup_object(phone['objectId'], 1)

    desk = next((o for o in event.metadata['objects']
                 if o['objectType'] == 'Desk'), None)
    
    event = step(dict(action="Pass", agentId=1))
    start = event.metadata['agent']['position']

    if phone and desk:
        move(start, desk['position'], occupied_human, 1)
        place_object(phone['objectId'], desk['objectId'], 1)



if __name__ == "__main__":

    human_phase()
    robot_phase()

    save_videos()
    print_stats()