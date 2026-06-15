from utils import *


def wash_object_start(occupied_position, agent_id=0):
    event = step(dict(action="Pass", agentId=agent_id))

    wash_id = None
    target_position = None

    for obj in event.metadata['objects']:
        if obj['objectType'] in ["Faucet", "Sink"]:
            wash_id = obj['objectId']
            target_position = obj['position']
            break

    if not wash_id:
        print("No sink/faucet")
        return event

    start = event.metadata['agent']['position']
    path = find_path(start, target_position, occupied_position, agent_id)

    if path:
        move_along_path(path, agent_id)

        execute_and_record(dict(action="ToggleObjectOn", objectId=wash_id, agentId=agent_id))
        for _ in range(2):
            execute_and_record(dict(action="Stand", agentId=agent_id))
        execute_and_record(dict(action="ToggleObjectOff", objectId=wash_id, agentId=agent_id))

    return event


def cut_object(target_id, agent_id=0):
    event = step(dict(action="Pass", agentId=agent_id))

    knife_id = None

    for o in event.metadata['objects']:
        if o['objectType'] == "Knife" and o['pickupable'] and o['visible']:
            knife_id = o['objectId']
            execute_and_record(dict(action='PickupObject', objectId=knife_id, agentId=agent_id))
            break

    if knife_id and target_id:
        execute_and_record(dict(action="SliceObject", objectId=target_id, agentId=agent_id))
    else:
        print("Cut failed")

    return event


def open_object(object_id, agent_id=0):
    event = step(dict(action="Pass", agentId=agent_id))

    for o in event.metadata['objects']:
        if o['objectId'] == object_id:
            if o.get('openable', False):
                execute_and_record(dict(action="OpenObject", objectId=object_id, agentId=agent_id))
                print(f"Open {o['objectType']}")
            else:
                print("Object is not openable")
            break

    return event


def close_object(object_id, agent_id=0):
    event = step(dict(action="Pass", agentId=agent_id))

    for o in event.metadata['objects']:
        if o['objectId'] == object_id:
            if o.get('openable', False):
                execute_and_record(dict(action="CloseObject", objectId=object_id, agentId=agent_id))
                print(f"Close {o['objectType']}")
            else:
                print("Object is not closeable")
            break

    return event


def pour_water(source_id, target_id, agent_id=0):
    event = step(dict(action="Pass", agentId=agent_id))

    source_ok = False
    target_ok = False

    for o in event.metadata['objects']:
        if o['objectId'] == source_id and o.get("isPickedUp", False):
            source_ok = True
        if o['objectId'] == target_id:
            target_ok = True

    if source_ok and target_ok:
        execute_and_record(dict(
            action="PourObject",
            objectId=source_id,
            receptacleObjectId=target_id,
            agentId=agent_id
        ))
        print("Pouring water done.")
    else:
        print("Pour failed: missing source or target")

    return event