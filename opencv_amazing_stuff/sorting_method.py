# ----------------heap sort algorithm-----------------#
def heap_up(queue, index):  # all the nodes are initialize to infinity. Any value change in node will compare
    if index <= 0:  # its value with parent to shuffle if needed.
        return
    parent_index = (index - 1) // 2
    if queue[parent_index].distance > queue[index].distance:
        queue[parent_index], queue[index] = queue[index], queue[parent_index]
        queue[parent_index].queue_index, queue[index].queue_index = parent_index, index
        heap_up(queue, parent_index)


def heap_down(queue, index):  # when exchange the root with leaf, the leaf at root need to shuffle back downward
    l_child_index, r_child_index = index * 2 + 1, index * 2 + 2
    if l_child_index >= len(queue):  # if no children, means the node is at the lowest level. No need to shift down
        return
    elif l_child_index < len(queue) <= r_child_index:  # if there is only left leaf
        if queue[l_child_index].distance < queue[index].distance:
            queue[l_child_index], queue[index] = queue[index], queue[l_child_index]
            queue[l_child_index].queue_index = l_child_index
            queue[index].queue_index = index
            heap_down(queue, l_child_index)
    else:  # if there are two children nodes
        smaller_child_index = l_child_index
        if queue[l_child_index].distance > queue[r_child_index].distance:
            smaller_child_index = r_child_index

        if queue[smaller_child_index].distance < queue[index].distance:
            queue[smaller_child_index], queue[index] = queue[index], queue[smaller_child_index]
            queue[smaller_child_index].queue_index = smaller_child_index
            queue[index].queue_index = index
            heap_down(queue, smaller_child_index)

    return

#  ----------------------------------------------------------------------------------
