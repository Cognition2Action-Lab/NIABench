from utils import *
from action_skills import *

controller.reset('FloorPlan28')
step(dict(action='Initialize', gridSize=0.25, agentCount=2))



def human_phase_1():

    occupied_robot = step(dict(action="Pass", agentId=1)).metadata['agent']['position']
    event = step(dict(action="Pass", agentId=0))

    start = event.metadata['agent']['position']

    knife = next((o for o in event.metadata['objects'] if o['objectType'] == 'Knife'), None)

    if knife:
        move(start, knife['position'], occupied_robot, 0)
        pickup_object(knife['objectId'], 0)

    bread = next((o for o in event.metadata['objects'] if o['objectType'] == 'Bread'), None)

    event = step(dict(action="Pass", agentId=0))
    start = event.metadata['agent']['position']

    if bread:
        move(event.metadata['agent']['position'], bread['position'], occupied_robot, 0)
        cut_object(bread['objectId'], 0)

        execute_and_record(dict(action="MoveHeldObjectAhead", moveMagnitude=0.3, agentId=0))
        execute_and_record(dict(action="DropHandObject", agentId=0))

        sliced = next((o for o in step(dict(action="Pass", agentId=0)).metadata['objects']
                      if o['objectType'] == 'BreadSliced'), None)

        if sliced:
            pickup_object(sliced['objectId'], 0)

    toaster = next((o for o in step(dict(action="Pass", agentId=0)).metadata['objects']
                   if o['objectType'] == 'Toaster'), None)
    
    event = step(dict(action="Pass", agentId=0))
    start = event.metadata['agent']['position']

    if toaster:
        move(event.metadata['agent']['position'], toaster['position'], occupied_robot, 0)

        step(dict(action="PutObject", objectId=toaster['objectId'], agentId=0))
        step(dict(action="ToggleObjectOn", objectId=toaster['objectId'], agentId=0))

    return sliced['objectId'] if sliced else None, toaster['objectId'] if toaster else None




def robot_phase():

    occupied_human = step(dict(action="Pass", agentId=0)).metadata['agent']['position']
    event = step(dict(action="Pass", agentId=1))

    start = event.metadata['agent']['position']

    plate = next((o for o in event.metadata['objects'] if o['objectType'] == 'Plate'), None)

    if plate:
        move(start, plate['position'], occupied_human, 1)
        pickup_object(plate['objectId'], 1)

    counter = next((o for o in event.metadata['objects']
                   if o['objectType'] == 'CounterTop' and o['position']['x'] < -1), None)

    event = step(dict(action="Pass", agentId=1))
    start = event.metadata['agent']['position']

    if counter:
        move(event.metadata['agent']['position'], counter['position'], occupied_human, 1)
        place_object(plate['objectId'], counter['objectId'], 1)

    butter = next((o for o in step(dict(action="Pass", agentId=1)).metadata['objects']
                  if o['objectType'] == 'ButterKnife'), None)
    
    event = step(dict(action="Pass", agentId=1))
    start = event.metadata['agent']['position']

    if butter:
        move(event.metadata['agent']['position'], butter['position'], occupied_human, 1)
        pickup_object(butter['objectId'], 1)

        move(event.metadata['agent']['position'], counter['position'], occupied_human, 1)
        place_object(butter['objectId'], counter['objectId'], 1)



def human_phase_2(sliced_id, toaster_id):

    occupied_robot = step(dict(action="Pass", agentId=1)).metadata['agent']['position']

    if toaster_id:
        execute_and_record(dict(action="ToggleObjectOff", objectId=toaster_id, agentId=0))

    pickup_object(sliced_id, 0)

    counter = next((o for o in event.metadata['objects']
                   if o['objectType'] == 'CounterTop' and o['position']['x'] < -1), None)

    event = step(dict(action="Pass", agentId=0))
    start = event.metadata['agent']['position']

    if counter:
        move(start, counter['position'], occupied_robot, 0)
        place_object(sliced_id, counter['objectId'], 0)




if __name__ == "__main__":

    sliced_id, toaster_id = human_phase_1()
    robot_phase()
    human_phase_2(sliced_id, toaster_id)

    save_videos()
    print_stats()