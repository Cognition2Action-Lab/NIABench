from utils import *
from action_skills import *

controller.reset('FloorPlan407')
step(dict(action='Initialize', gridSize=0.25, agentCount=2))



def human_phase():

    occupied_robot = step(dict(action="Pass", agentId=1)).metadata['agent']['position']
    event = step(dict(action="Pass", agentId=0))

    start = event.metadata['agent']['position']

    shower = next((o for o in event.metadata['objects']
                   if o['objectType'] == 'ShowerDoor'), None)

    if shower:
        target_pos = shower['axisAlignedBoundingBox']['center']

        move(start, target_pos, occupied_robot, 0)
        open_object(shower['objectId'], 0)

        for _ in range(4):
            execute_and_record(dict(action="MoveAhead", agentId=0))
        for _ in range(2):
            execute_and_record(dict(action="RotateLeft", agentId=0))

        close_object(shower['objectId'], 0)

        return target_pos


def robot_phase(shower_pos):

    occupied_human = step(dict(action="Pass", agentId=0)).metadata['agent']['position']
    event = step(dict(action="Pass", agentId=1))

    start = event.metadata['agent']['position']

    towel = next((o for o in event.metadata['objects']
                  if o['objectType'] == 'HandTowel'), None)

    if towel:
        move(start, towel['position'], occupied_human, 1)
        event = pickup_object(towel['objectId'], 1)

        move(event.metadata['agent']['position'], shower_pos, occupied_human, 1)
        execute_and_record(dict(action="DropHandObject", agentId=1))


    soap = next((o for o in event.metadata['objects']
                 if o['objectType'] == 'SoapBar'), None)
    
    event = step(dict(action="Pass", agentId=1))
    start = event.metadata['agent']['position']

    if soap:
        move(start, soap['position'], occupied_human, 1)
        event = pickup_object(soap['objectId'], 1)

        move(event.metadata['agent']['position'], shower_pos, occupied_human, 1)
        execute_and_record(dict(action="DropHandObject", agentId=1))



if __name__ == "__main__":
    shower_pos = human_phase()
    robot_phase(shower_pos)

    save_videos()
    print_stats()