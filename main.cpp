#include <mpi.h>
#include <cmath>
#include <deque>
#include <cstdlib>
#include <pthread.h>

#define TASKS_PER_COMP (100)   // Each machine (or just process)
                               // have to solve such amount of tasks
                               // We need to balance tasks between them,
                               // because tasks have different weights

int iterCounter = 0;           // Only for main

std::deque<int> tasks;         // For worker & manager (shared)
bool no_more_tasks;            // For worker & manager (shared)
pthread_mutex_t mutex_tl;      // For synchronization

double global_res = 0;         // Only for worker
int tasks_count = 0;           // For worker & main -- already synchronised


int AssignWeight(int size, int rank, int idx, int L) {
    return abs(50 - (idx % TASKS_PER_COMP)) * abs(rank - (iterCounter % size)) * L;
}

void GetStartTasks(int rank, int* left, int* right) {
    *left = TASKS_PER_COMP * rank;
    *right = *left + TASKS_PER_COMP;
}

void OneTask(int repeatCount) {
    double res = 0;
    for (int i = 0; i < repeatCount; ++i) {
        res += sin(i);
    }
    global_res += res;
}

void* WorkerTask(void* attrs) {

    int task_weight;
    bool running = true;
    while (running) {

        // Receive task
        pthread_mutex_lock(&mutex_tl);
        if (tasks.empty()) {
            task_weight = -1;
        } else {
            task_weight = tasks.front();
            tasks.pop_front();
        }
        pthread_mutex_unlock(&mutex_tl);

        // Do job
        if (task_weight != -1) {
            OneTask(task_weight);
            tasks_count++;
        }

        // Check if all tasks completed
        pthread_mutex_lock(&mutex_tl);
        if (no_more_tasks) {
            running = false;
        }
        pthread_mutex_unlock(&mutex_tl);
    }

    return nullptr;
}

int DecideProcessRoot(int rank, bool want_be_root) {
    int root;
    int challenger = -1;   // Don't want to be root -- send -1 to say it
    if (want_be_root) {
        challenger = rank; // Want to be root       -- send his rank
    }
    MPI_Allreduce(&challenger, &root, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    return root;
}

int DecideRootPretendCount(bool want_be_root) {
    int root_pretend_count;
    int want_be_root_num = (int)want_be_root;
    MPI_Allreduce(&want_be_root_num, &root_pretend_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return root_pretend_count;
}

void DelegateTask(int size, int rank, int worker_rank, int weight) {
    int delegated_tasks[size];

    MPI_Gather(&weight, 1, MPI_INT, delegated_tasks, 1, MPI_INT, worker_rank, MPI_COMM_WORLD);

    if (rank == worker_rank) {
        pthread_mutex_lock(&mutex_tl);
        for (int i = 0; i < size; ++i) {
            if (delegated_tasks[i] != -1) tasks.push_back(delegated_tasks[i]);
        }
        pthread_mutex_unlock(&mutex_tl);
    }
}

void* ManagerTask(void* attrs) {
    int size = ((int*)attrs)[0];
    int rank = ((int*)attrs)[1];

    int tasks_in_deq;
    int processes_root_count = 0;
    while (processes_root_count != size) {
        pthread_mutex_lock(&mutex_tl);
        tasks_in_deq = (int)tasks.size();
        pthread_mutex_unlock(&mutex_tl);

        bool want_be_root = (bool)(tasks_in_deq == 0);
        int actual_root = DecideProcessRoot(rank, want_be_root);
        if (actual_root != -1) { // Some process want to get more tasks
            processes_root_count = DecideRootPretendCount(want_be_root);

            int delegate_task_weight = -1; // don't want to delegate
            pthread_mutex_lock(&mutex_tl);
            if (!want_be_root && !tasks.empty()) {
                delegate_task_weight = tasks.back();
                tasks.pop_back();
            }
            pthread_mutex_unlock(&mutex_tl);

            DelegateTask(size, rank, actual_root, delegate_task_weight);
        }
    }

    // Say to worker that all tasks completed
    pthread_mutex_lock(&mutex_tl);
    no_more_tasks = true;
    pthread_mutex_unlock(&mutex_tl);

    return nullptr;
}


int main(int argc, char* argv[]) {
    int rank, size;

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided != MPI_THREAD_SERIALIZED) {
        if (rank == 0) fprintf(stderr, "MPI can't provide MPI_THREAD_SERIALIZED level");
        MPI_Finalize();
        return 1;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Read params and handle errors
    if (argc < 3) {
        if (rank == 0) fprintf(stderr, "Specify exact 2 args: L, itersMax");
        MPI_Finalize();
        return 1;
    }
    int L = std::atoi(argv[1]);
    if (L <= 0) {
        fprintf(stderr, "Expect positive integer L, got: %d\n", L);
        MPI_Finalize();
        return 1;
    }
    int itersMax = std::atoi(argv[2]);
    if (itersMax <= 0) {
        fprintf(stderr, "Expect positive integer itersMax, got: %d\n", itersMax);
        MPI_Finalize();
        return 1;
    }

    pthread_mutex_init(&mutex_tl, NULL);

    // Get tasks range
    int left_s, right_s;
    GetStartTasks(rank, &left_s, &right_s);

    // Prepare manager function attributes
    int size_rank[2] = {size, rank};

    pthread_attr_t worker_attr, manager_attr;

    // Attributes init
    pthread_attr_init(&worker_attr);
    pthread_attr_init(&manager_attr);
    pthread_attr_setdetachstate(&worker_attr, PTHREAD_CREATE_JOINABLE);
    pthread_attr_setdetachstate(&manager_attr, PTHREAD_CREATE_JOINABLE);

    double max_disbalance_prop = 0;
    double full_start_time = MPI_Wtime();
    for (int i = 0; i < itersMax; ++i) {

        // Load start tasks
        for (int j = left_s; j < right_s; ++j) {
            tasks.push_back(AssignWeight(size, rank, j, L));
        }
        no_more_tasks = false;

        pthread_t worker_tid, manager_tid;

        // Start Worker and Manager
        pthread_create(&worker_tid, &worker_attr, WorkerTask, NULL);
        pthread_create(&manager_tid, &manager_attr, ManagerTask, (void*)size_rank);

        double start_time = MPI_Wtime();

        // Finally join
        pthread_join(worker_tid, NULL);
        pthread_join(manager_tid, NULL);

        double end_time = MPI_Wtime();
        double time_required = end_time - start_time;
        double max_time_required, min_time_required;
        MPI_Reduce(&time_required, &max_time_required, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&time_required, &min_time_required, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

        // Print iteration results
        for (int j = 0; j < size; ++j) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (j == rank) {
                printf("| %2d | %2d | Tasks computed: %10d |\n", iterCounter, rank, tasks_count);
                printf("| %2d | %2d | Global result: %11lf |\n", iterCounter, rank, global_res);
                printf("| %2d | %2d | Time required: %11lf |\n", iterCounter, rank, time_required);
                printf("+----+----+----------------------------+\n");
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            double time_disbalance = max_time_required - min_time_required;
            double disbalance_prop = time_disbalance / max_time_required * 100.0;
            if (disbalance_prop > max_disbalance_prop) max_disbalance_prop = disbalance_prop;

            printf("| %2d | Time disbalance: %14lf |\n", iterCounter, time_disbalance);
            printf("| %2d | Disbalance prop: %12lf % |\n", iterCounter, disbalance_prop);
            printf("+----+----+----------------------------+\n");
        }

        tasks_count = 0;
        global_res = 0;

        iterCounter++;
    }

    // Destroy attributes
    pthread_attr_destroy(&worker_attr);
    pthread_attr_destroy(&manager_attr);

    pthread_mutex_destroy(&mutex_tl);

    double full_end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Full time required: %lf\n", full_end_time - full_start_time);
        printf("Max disbalance prop: %lf\n", max_disbalance_prop);
    }

    MPI_Finalize();
    return 0;
}
