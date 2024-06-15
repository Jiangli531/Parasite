import csv
import copy
train_fileHeader = ["local_model", "round", "epoch", "internal_epoch", "average_loss", "accuracy", "correct_data",
                    "total_data"]
test_fileHeader = ["epoch", "average_loss", "accuracy"]
train_result = [] 
test_result = []  
posiontest_result = []  

triggertest_fileHeader = ["model", "trigger_name", "trigger_value", "epoch", "average_loss", "accuracy", "correct_data",
                          "total_data"]
poisontriggertest_result = []  

posion_test_result = []  
posion_posiontest_result = []  
weight_result=[]
scale_result=[]
scale_temp_one_row=[]

def save_result_csv(epoch, is_posion,folder_path):
    train_csvFile = open(f'{folder_path}/train_result.csv', "w")
    train_writer = csv.writer(train_csvFile)
    train_writer.writerow(train_fileHeader)
    train_writer.writerows(train_result)
    train_csvFile.close()

    test_csvFile = open(f'{folder_path}/test_result.csv', "w")
    test_writer = csv.writer(test_csvFile)
    test_writer.writerow(test_fileHeader)
    test_writer.writerows(test_result)
    test_csvFile.close()


    if is_posion:
        test_csvFile = open(f'{folder_path}/poisontest_result.csv', "w")
        test_writer = csv.writer(test_csvFile)
        test_writer.writerow(test_fileHeader)
        test_writer.writerows(posiontest_result)
        test_csvFile.close()


def add_weight_result(name,weight,alpha):
    weight_result.append(name)
    weight_result.append(weight)
    weight_result.append(alpha)


