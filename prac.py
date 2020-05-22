# lst = [1, (2,1,1), (3,1,1)]
# a = lst[-1]
# print(lst[-1])
# print(a[0])

# print(lst)
# lst2 = lst
# print(lst2)
# lst.pop()
# print(lst2)


import heapq

heap = []
entry = (3, 'a')
heapq.heappush(heap, entry)
entry1 = (1, [(4,5), 'N', 1])
heapq.heappush(heap, entry1)
(_, item)= heapq.heappop(heap)
path = item
print (item)
print(item[0])


path = {1 : [2, 3]}
print(path)
path[1] = [2, 3, 4]
print(path)

ls = path[1]
ls.append(5)
print(path)

for a in range(-2,3):
    print(a)
print(range(-3,3))
action = "south"
lst = list(action)
print(lst)
print([action])

p1 = []
p1.append("1")
print(p1)
p1.append("2")
print(p1)


a = (1,0)
b = (0,1)
c = a + b
print(c)