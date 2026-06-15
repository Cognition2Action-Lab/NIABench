from utils import *
from action_skills import *

controller.reset('FloorPlan28')
step(dict(action='Initialize', gridSize=0.25, agentCount=2))


def human_phase_1():
    occupied_robot = step(dict(action="Pass", agentId=1)).metadata['agent']['position']
    event = step(dict(action="Pass", agentId=0))

    start = event.metadata['agent']['position']

    tomato = None
    for o in event.metadata['objects']:
        if o['objectType'] == 'Tomato':
            tomato = o
            break

    if not tomato:
        print("No tomato")
        return None

    move(start, tomato['position'], occupied_robot, 0)

    # if hidden
    event = step(dict(action="Pass", agentId=0))
    for o in event.metadata['objects']:
        if o['objectType'] == 'Tomato' and not o['visible']:
            execute_and_record(dict(action="LookDown", agentId=0))

    pickup_object(tomato['objectId'], 0)

    return tomato['objectId']


def robot_phase():
    occupied_human = step(dict(action="Pass", agentId=0)).metadata['agent']['position']
    event = step(dict(action="Pass", agentId=1))

    start = event.metadata['agent']['position']

    knife = None
    for o in event.metadata['objects']:
        if o['objectType'] == 'Knife':
            knife = o
            break

    if not knife:
        print("No knife")
        return None, None

    move(start, knife['position'], occupied_human, 1)
    pickup_object(knife['objectId'], 1)

    # move to counter
    event = step(dict(action="Pass", agentId=1))
    start = event.metadata['agent']['position']
    table = None

    for o in event.metadata['objects']:
        if o['objectType'] == 'CounterTop' and o['position']['x'] < -1:
            table = o
            break

    if table:
        target = dict(table['position'])
        target['x'] += 0.25

        move(start, target, occupied_human, 1)
        place_object(knife['objectId'], table['objectId'], 1)

    return knife['objectId'], table['objectId']


def human_phase_2(tomato_id):
    occupied_robot = step(dict(action="Pass", agentId=1)).metadata['agent']['position']
    event = step(dict(action="Pass", agentId=0))

    start = event.metadata['agent']['position']

    move(start, event.metadata['agent']['position'], occupied_robot, 0)

    wash_object_start(occupied_robot, 0)

    # counter
    event = step(dict(action="Pass", agentId=0))
    start = event.metadata['agent']['position']
    table = None

    for o in event.metadata['objects']:
        if o['objectType'] == 'CounterTop' and o['position']['x'] < -1:
            table = o
            break

    if table:
        move(start, table['position'], occupied_robot, 0)

        place_object(tomato_id, table['objectId'], 0)

        cut_object(tomato_id, 0)

        execute_and_record(dict(action="MoveHeldObjectAhead", moveMagnitude=0.3, agentId=0))
        execute_and_record(dict(action="DropHandObject", agentId=0))



if __name__ == "__main__":

    tomato_id = human_phase_1()
    knife_id, table_id = robot_phase()
    human_phase_2(tomato_id)

    save_videos()
    print_stats()