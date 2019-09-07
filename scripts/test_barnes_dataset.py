import numpy as np
from barnes_collect_dataset import *
from bite_selection_package.config import spanet_config as config

EXCLUDED_FOOD = config.excluded_item
FOOD_NAME_TO_INDEX = get_food_name()
food_items = EXCLUDED_FOOD.split("_")
food_items = list(map(lambda x:"_"+x+"_",food_items))
food_item_indices = list(map(lambda x:FOOD_NAME_TO_INDEX[x],food_items))
from conban_spanet.spanet_driver import SPANetDriver

driver = SPANetDriver(config.excluded_item, None, 1, False, True)

print(list(set(FOOD_NAME_TO_INDEX.values()) - set(food_item_indices)))

# food_data = np.genfromtxt("barnes_partial_dataset_train_all.csv",delimiter=',')
# food_data=food_data[food_data[:,-1]==1] # select  wall
# print(food_data.shape)
# banana_data = food_data[food_data[:,-2]==0]
# print(banana_data.shape)
# print(np.unique(food_data[:, -1]))
#print(food_data[:,-2][:6])
#print((food_data[:, -3][food_data[:,-2] in [12,13]]).shape)

# target = list(map(lambda x: food_data[food_data[:,-2]==x,:],[12,13]))
# print(np.unique(banana_data[:,-4])
