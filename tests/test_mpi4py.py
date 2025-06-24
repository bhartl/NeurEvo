from mpi4py import MPI
import time
import datetime

# load MPI communicator
comm = MPI.COMM_WORLD

# get rank (process id) and size (#processes)
size = comm.Get_size()
rank = comm.Get_rank()


def test(num_iterations=4,  sleep_time=2):
    start = datetime.datetime.now()
    for i in range(num_iterations):
        if i % size == rank:
            ct = datetime.datetime.now()
            print(f"rank {rank} started batch {i+1}/{num_iterations} @ {str(ct)}")
            time.sleep(sleep_time)
    print("duration: ", datetime.datetime.now() - start)


if __name__ == "__main__":
    test()
