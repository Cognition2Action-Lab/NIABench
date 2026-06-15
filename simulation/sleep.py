from utils import *
from action_skills import *

controller.reset('FloorPlan323')
step(dict(action='Initialize', gridSize=0.25, agentCount=2))



def human_phase():

    occupied_robot = step(dict(action="Pass", agentId=1)).metadata['agent']['position']
    event = step(dict(action="Pass", agentId=0))

    start = event.metadata['agent']['position']

    bed = next((o for o in event.metadata['objects'] if o['objectType'] == 'Bed'), None)
    sofa = next((o for o in event.metadata['objects'] if o['objectType'] == 'Sofa'), None)

    if bed:
        recept = bed.get('receptacleObjectIds', [])

        for obj in event.metadata['objects']:
            if obj['objectId'] in recept and obj['objectType'] != 'Pillow':
                move(start, obj['position'], occupied_robot, 0)
                event = pickup_object(obj['objectId'], 0)

                move(event.metadata['agent']['position'], sofa['position'], occupied_robot, 0)
                place_object(obj['objectId'], sofa['objectId'], 0)

        pillow = next((o for o in event.metadata['objects'] if o['objectType'] == 'Pillow'), None)
        
        event = step(dict(action="Pass", agentId=0))
        start = event.metadata['agent']['position']

        if pillow:
            move(start, pillow['position'], occupied_robot, 0)


def robot_phase():

    occupied_human = step(dict(action="Pass", agentId=0)).metadata['agent']['position']
    event = step(dict(action="Pass", agentId=1))

    start = event.metadata['agent']['position']

    blinds = next((o for o in event.metadata['objects'] if o['objectType'] == 'Blinds'), None)

    if blinds:
        move(start, blinds['position'], occupied_human, 1)
        open_object(blinds['objectId'], 1)

    mug = next((o for o in event.metadata['objects'] if o['objectType'] == 'Mug'), None)

    event = step(dict(action="Pass", agentId=1))
    start = event.metadata['agent']['position']

    if mug:
        move(start, mug['position'], occupied_human, 1)
        pickup_object(mug['objectId'], 1)

        pour_water(mug['objectId'], 1)
        execute_and_record(dict(action="DropHandObject", agentId=1))


if __name__ == "__main__":
    human_phase()
    robot_phase()

    save_videos()
    print_stats()