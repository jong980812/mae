


#creates id_plit folder
python ./code/data_split.py

# This is the training script for the ResNet model on the person dataset.
# The model is trained on the person dataset, and the model is saved in the
# person_resnet directory.

python ./train_multiple_model.py ./dataset/id_split -m resnet -d person -mp ./person_resnet -e 20
