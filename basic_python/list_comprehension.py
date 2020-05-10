even_list = []
for i in range(1, 11):
    if i%2 == 0:
        even_list.append(i)

print(even_list)

lc_pos_list = [i for i in range(1,11)]

print(lc_pos_list)

lc_even_list = [i for i in range(1,11) if i%2]
print(lc_even_list)

pos_list_gt_5 = []
for i in pos_list:
    if i >= 5:
      pos_list_gt_5.append(i)
      
print(pos_list_gt_5)

lc_pos_list_gt_5 = [i for i in pos_list if i >= 5]
print(lc_pos_list_gt_5)

pos_list2 = []
for i in range(20, 30):
    pos_list2.append(i)

print(pos_list)    
print(pos_list2)
multiply_list = []
for i, j in zip(pos_list, pos_list2):
        multiply_list.append(i*j)
        
print(multiply_list)

lc_multiply_list = [i*j for i,j in zip(pos_list, pos_list2)]
print(lc_multiply_list)


lc_multiply_list_filter = [i*j for i,j in zip(pos_list, pos_list2) if i*j < 200]
print(lc_multiply_list_filter)
