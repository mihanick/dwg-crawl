'''
main runner from console
'''
from main import run
# keep lr low, or lstm output will have NaNs
# https://stackoverflow.com/questions/33962226/common-causes-of-nans-during-training

run(
    pickle_file="test_dataset_groups.pickle",
    train_verbose=False,
    batch_size=4,
    limit_seq_len=2500,
    epochs=5,
    lr=0.0006)
