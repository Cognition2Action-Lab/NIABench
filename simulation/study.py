from utils import *
from action_skills import *

controller.reset('FloorPlan323')
step(dict(action='Initialize', gridSize=0.25, agentCount=2))



def human_phase():

    occupied_robot = step(dict(action="Pass", agentId=1)).metadata['agent']['position']
    event = step(dict(action="Pass", agentId=0))

    start = event.metadata['agent']['position']

    desk = next((o for o in event.metadata['objects'] if o['objectType'] == 'Desk'), None)
    sofa = next((o for o in event.metadata['objects'] if o['objectType'] == 'Sofa'), None)

    if desk:
        recept = desk.get('receptacleObjectIds', [])

        for obj in event.metadata['objects']:
            if obj['objectId'] in recept and obj['objectType'] in ['Book', 'Bowl']:
                move(start, obj['position'], occupied_robot, 0)
                event = pickup_object(obj['objectId'], 0)
                move(event.metadata['agent']['position'], sofa['position'], occupied_robot, 0)
                place_object(obj['objectId'], sofa['objectId'], 0)

    laptop = next((o for o in event.metadata['objects'] if o['objectType'] == 'Laptop'), None)

    event = step(dict(action="Pass", agentId=0))
    start = event.metadata['agent']['position']
    
    if laptop:
        move(start, laptop['position'], occupied_robot, 0)
        event = pickup_object(laptop['objectId'], 0)
        move(event.metadata['agent']['position'], desk['position'], occupied_robot, 0)
        place_object(laptop['objectId'], desk['objectId'], 0)

    return


def robot_phase():

    occupied_human = step(dict(action="Pass", agentId=0)).metadata['agent']['position']
    event = step(dict(action="Pass", agentId=1))

    start = event.metadata['agent']['position']

    phone = next((o for o in event.metadata['objects'] if o['objectType'] == 'CellPhone'), None)
    desk = next((o for o in event.metadata['objects'] if o['objectType'] == 'Desk'), None)

    if phone:
        move(start, phone['position'], occupied_human, 1)
        event = pickup_object(phone['objectId'], 1)
        
        move(event.metadata['agent']['position'], desk['position'], occupied_human, 1)
        place_object(phone['objectId'], desk['objectId'], 1)

    mug = next((o for o in event.metadata['objects'] if o['objectType'] == 'Mug'), None)

    event = step(dict(action="Pass", agentId=1))
    start = event.metadata['agent']['position']

    if mug:
        move(start, mug['position'], occupied_human, 1)
        event = pickup_object(mug['objectId'], 1)

        move(event.metadata['agent']['position'], desk['position'], occupied_human, 1)
        pour_water(mug['objectId'], 1)
        place_object(mug['objectId'], desk['objectId'], 1)


if __name__ == "__main__":
    human_phase()
    robot_phase()

    save_videos()
    print_stats()