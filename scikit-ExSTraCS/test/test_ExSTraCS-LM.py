import sys
import os
import random
import time

if __name__ == '__main__':
    print(sys.path[0])
    sys.path.append(os.path.dirname(sys.path[0]))

from skExSTraCS import ExSTraCS,StringEnumerator

start_time = time.time()
print("Timestamp:", start_time)

ftrs = "lbp_fold1"
data_path = r"datasets/cats_dogs/"

data_path += ''
file_name = ftrs + ".csv"
train_file = data_path + "/train/" + file_name
test_file = data_path + "/test/" + file_name
log_dir = sys.path[0]+"/"
log_dir += "../MetaData/cats_dogs/"
log_popfile_name = "log_pop_dist-test_" + ftrs
log_trainingfile_name = os.path.basename(__file__)+"log_" + ftrs + "_" + str(random.random())[:5] + ".txt"

train_converter = StringEnumerator(train_file,'Class')
train_headers, train_classLabel, train_dataFeatures, train_dataPhenotypes = train_converter.get_params()

learning_iterations = 100000
N = 2000
level = 1
p_spec = 0.66

model = ExSTraCS(learning_iterations=learning_iterations, N=N,level=level,p_spec=p_spec,log_dir=log_dir, log_trainingfile_name=log_trainingfile_name)

test_converter = StringEnumerator(test_file,'Class')
test_headers, test_classLabel, test_dataFeatures, test_dataPhenotypes = test_converter.get_params()

print("Model training in progress ...")
model.fit(train_dataFeatures, train_dataPhenotypes,test_dataFeatures,test_dataPhenotypes)
print("Model training Ends")

accuracy = model.score(test_dataFeatures,test_dataPhenotypes)
#model.log_trainingfile.close()

print(accuracy)
#filename = ftrs +  "_export.csv"
#model.export_final_rule_population(filename=filename)

elapsed = time.time() - start_time
hours, rem = divmod(elapsed, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Elapsed time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")