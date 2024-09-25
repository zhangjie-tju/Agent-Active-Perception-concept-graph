# from os import path
# root = "/home/user/Documents/ZJ/concept-graphs/edit_graphs"
# sys_f="system_message.txt"
# with open(path.join(root,sys_f)) as f:
#     sys_=f.readlines()
# print(0)
import re
str = "input abcdjdsywuf output sgjasgfhsyrwebcgyuc"
split_content = re.split(r'input|output| ', str)
filtered_list = [item for item in split_content if item]

print(filtered_list)