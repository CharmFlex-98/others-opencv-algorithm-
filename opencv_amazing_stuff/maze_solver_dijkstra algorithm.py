import cv2
import matplotlib.pyplot as plt
import numpy as np
import sorting_method


class Vertex:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.distance = float('inf')  # initialize to infinity
        self.queue_index = None
        self.popped = False

    def find_distance(self, image, adjacent_vertex):
        distance = 0.1 + int(image[adjacent_vertex.y][adjacent_vertex.x][0] - image[self.y][self.x][0]) ** 2 + \
                   int(image[adjacent_vertex.y][adjacent_vertex.x][1] - image[self.y][self.x][1]) ** 2 + \
                   int(image[adjacent_vertex.y][adjacent_vertex.x][2] - image[self.y][self.x][2]) ** 2

        return distance

    def find_neighbour_vertices(self, tensor):
        neighbour_vertices = []
        if self.x in range(0 + 1, tensor.shape[1] - 1):
            left_vertex = tensor[self.y][self.x - 1]
            right_vertex = tensor[self.y][self.x + 1]
            neighbour_vertices.append(left_vertex)
            neighbour_vertices.append(right_vertex)
        elif self.x == 0:
            right_vertex = tensor[self.y][self.x + 1]
            neighbour_vertices.append(right_vertex)
        else:
            left_vertex = tensor[self.y][self.x - 1]
            neighbour_vertices.append(left_vertex)

        if self.y in range(0 + 1, tensor.shape[0] - 1):
            up_vertex = tensor[self.y - 1][self.x]
            down_vertex = tensor[self.y + 1][self.x]
            neighbour_vertices.append(up_vertex)
            neighbour_vertices.append(down_vertex)
        elif self.y == 0:
            down_vertex = tensor[self.y + 1][self.x]
            neighbour_vertices.append(down_vertex)
        else:
            up_vertex = tensor[self.y - 1][self.x]
            neighbour_vertices.append(up_vertex)

        return neighbour_vertices


def vertex_min_distance(_vertex_queue):
    min_distance = None
    min_distance_index = None
    for _vertex in _vertex_queue:
        if min_distance is None or min_distance > _vertex.distance:
            min_distance = _vertex.distance
            min_distance_index = _vertex.queue_index

    return min_distance_index, _vertex_queue[min_distance_index]


def draw_path(_vertex):
    cv2.circle(maze, (_vertex.x, _vertex.y), 1, [255, 0, 0], -1)
    if _vertex.parent is None:
        return
    draw_path(_vertex.parent)


def find_insert_position(initial_index, distance, _vertex_queue, lower_index, higher_index):
    index = lower_index + (higher_index - lower_index) // 2
    if index != lower_index:
        if distance < _vertex_queue[index].distance:
            find_insert_position(initial_index, distance, _vertex_queue, lower_index, index - 1)
        elif distance > _vertex_queue[index].distance:
            find_insert_position(initial_index, distance, _vertex_queue, index + 1, higher_index)
        else:
            insert_position(_vertex_queue, initial_index, index)
    else:
        insert_position(_vertex_queue, initial_index, index)


def insert_position(_vertex_queue, initial_index, insert_index):
    _vertex_queue.insert(insert_index, _vertex_queue.pop(initial_index))
    _vertex_queue[insert_index].queue_index = insert_index
    for _vertex in _vertex_queue[insert_index + 1:initial_index + 1]:
        _vertex.queue_index += 1


maze_image_path = '/home/jiaming/Pictures/maze.jpg'
starting_point = 39, 78
ending_point = 84, 21

maze = cv2.imread(maze_image_path)
maze = cv2.resize(maze, (100, 100), interpolation=cv2.INTER_AREA)

# create a numpy array for storing each vertex
vertex_tensor = np.full((maze.shape[0], maze.shape[1]), fill_value=None)

# put all vertices in queue
vertex_queue = []
for row_index, row in enumerate(maze):
    for column_index, column in enumerate(row):
        vertex = Vertex(column_index, row_index)
        vertex.queue_index = len(vertex_queue)
        vertex_tensor[row_index][column_index] = vertex
        vertex_queue.append(vertex)
starting_vertex = vertex_tensor[starting_point[1]][starting_point[0]]
starting_vertex.distance = 0
sorting_method.heap_up(vertex_queue, starting_vertex.queue_index)

# loop
while vertex_queue:
    current_vertex = vertex_queue[0]
    vertex_queue[0] = vertex_queue[-1]
    vertex_queue.pop()
    sorting_method.heap_down(vertex_queue, 0)
    if current_vertex.x == ending_point[0] and current_vertex.y == ending_point[1]:
        break
    for neighbour_vertex in current_vertex.find_neighbour_vertices(vertex_tensor):
        if not neighbour_vertex.popped:
            # perform dijkstra algorithm
            edge = current_vertex.find_distance(maze, neighbour_vertex)
            if neighbour_vertex.distance > current_vertex.distance + edge:
                neighbour_vertex.distance = current_vertex.distance + edge
                sorting_method.heap_up(vertex_queue, neighbour_vertex.queue_index)
                neighbour_vertex.parent = current_vertex

    current_vertex.popped = True
    cv2.circle(maze, (current_vertex.x, current_vertex.y), 1, [0, 0, 255], -1)
    cv2.imshow('calculating path...', maze)
    cv2.waitKey(1)
    print(len(vertex_queue))

draw_path(vertex_tensor[ending_point[1]][ending_point[0]])
cv2.imshow('maze', maze)
cv2.waitKey(0)
