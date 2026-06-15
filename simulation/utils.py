import cv2
from collections import deque
import ai2thor.controller


controller = ai2thor.controller.Controller(width=512, height=512)


human_step_count = 0
robot_step_count = 0
total_step_count = 0


def step(action_dict):
    global human_step_count, robot_step_count, total_step_count

    event = controller.step(action_dict)
    agent_id = action_dict.get("agentId", None)

    total_step_count += 1
    if agent_id == 0:
        human_step_count += 1
    elif agent_id == 1:
        robot_step_count += 1

    return event


third_party_frames = []
cv2_frames = []


def execute_and_record(action_dict):
    event = step(action_dict)

    cv2_frames.append(event.cv2img)

    agent_id = action_dict['agentId']
    third_party_frames.append(
        event.events[agent_id].third_party_camera_frames[0]
    )

    return event



# Geometry / Path Planning
def get_surrounding_positions(center, grid_size=0.25):
    offsets = [-grid_size, 0, grid_size]
    return [
        {'x': center['x'] + dx, 'y': center['y'], 'z': center['z'] + dz}
        for dx in offsets for dz in offsets
    ]


def get_reachable_positions(occupied, agent_id=0):
    event = step(dict(action='GetReachablePositions', agentId=agent_id))
    reachable = event.metadata['actionReturn']

    blocked = get_surrounding_positions(occupied)

    return [p for p in reachable if p not in blocked]


def round_to_grid(pos, grid=0.25):
    return {
        'x': round(pos['x'] / grid) * grid,
        'y': pos['y'],
        'z': round(pos['z'] / grid) * grid
    }


def round_to_nearest(pos, reachable):
    best = None
    best_d = 1e9

    for p in reachable:
        d = (pos['x'] - p['x'])**2 + (pos['z'] - p['z'])**2
        if d < best_d:
            best_d = d
            best = p
    return best


def find_path(start, target, occupied, agent_id=0):
    reachable = get_reachable_positions(occupied, agent_id)
    reachable_set = {(p['x'], p['z']) for p in reachable}

    start = round_to_grid(start)
    target = round_to_nearest(target, reachable)

    q = deque([[start]])
    visited = {(start['x'], start['z'])}

    while q:
        path = q.popleft()
        cur = path[-1]

        if cur['x'] == target['x'] and cur['z'] == target['z']:
            return path

        neighbors = [
            {'x': cur['x'] + 0.25, 'y': cur['y'], 'z': cur['z']},
            {'x': cur['x'] - 0.25, 'y': cur['y'], 'z': cur['z']},
            {'x': cur['x'], 'y': cur['y'], 'z': cur['z'] + 0.25},
            {'x': cur['x'], 'y': cur['y'], 'z': cur['z'] - 0.25},
        ]

        for n in neighbors:
            key = (round(n['x'], 2), round(n['z'], 2))
            if key not in visited and key in reachable_set:
                visited.add(key)
                q.append(path + [n])

    return None



# Motion
def get_direction(agent, target):
    dx = target['x'] - agent['x']
    dz = target['z'] - agent['z']

    if abs(dx) > abs(dz):
        return 90 if dx > 0 else 270
    return 0 if dz > 0 else 180


def rotate_to(target_yaw, agent_id=0):
    event = step(dict(action='Pass', agentId=agent_id))
    yaw = event.metadata['agent']['rotation']['y']
    diff = (target_yaw - yaw + 360) % 360

    if diff == 90:
        execute_and_record(dict(action='RotateRight', agentId=agent_id))
    elif diff == 270:
        execute_and_record(dict(action='RotateLeft', agentId=agent_id))
    elif diff == 180:
        execute_and_record(dict(action='RotateRight', agentId=agent_id))
        execute_and_record(dict(action='RotateRight', agentId=agent_id))


def move_along_path(path, agent_id=0):
    for target in path[1:]:
        while True:
            event = step(dict(action='Pass', agentId=agent_id))
            pos = event.metadata['agent']['position']

            yaw = get_direction(pos, target)
            rotate_to(yaw, agent_id)

            event = execute_and_record(dict(action='MoveAhead', agentId=agent_id))
            new_pos = event.metadata['agent']['position']

            if new_pos == target:
                break


def move(start, target, occupied, agent_id=0):
    path = find_path(start, target, occupied, agent_id)
    if path:
        move_along_path(path, agent_id)



# Object Ops
def pickup_object(obj_id, agent_id=0):
    event = step(dict(action='Pass', agentId=agent_id))

    for o in event.metadata['objects']:
        if o['objectId'] == obj_id and o['visible'] and o['pickupable']:
            return execute_and_record(
                dict(action='PickupObject', objectId=obj_id, agentId=agent_id)
            )
    return event


def place_object(pickup_id, place_id, agent_id=0):
    event = step(dict(action='Pass', agentId=agent_id))

    if any(o['objectId'] == pickup_id and o['isPickedUp'] for o in event.metadata['objects']):
        execute_and_record(dict(action="MoveHeldObjectAhead", moveMagnitude=0.3, agentId=agent_id))
        execute_and_record(dict(action="DropHandObject", agentId=agent_id))

    return event



# Video
def save_videos():
    video1 = "task.mp4"
    h, w, _ = cv2_frames[0].shape
    out = cv2.VideoWriter(video1, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))

    for f in cv2_frames:
        out.write(f)
        out.write(f)
    out.release()

    video2 = "global.mp4"
    h, w, _ = third_party_frames[0].shape
    out = cv2.VideoWriter(video2, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))

    for f in third_party_frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

    out.release()


def print_stats():
    print("========== STATS ==========")
    print("Human:", total_step_count)
    print("Human Saved:", human_step_count)