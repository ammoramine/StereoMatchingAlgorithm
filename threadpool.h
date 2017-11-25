#ifndef THREADPOOL_INCLUDED
#define THREADPOOL_INCLUDED
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <deque>
#include <future>
#include <iostream>
#include <chrono>
// #include "main.cpp"
class Worker;
// global std::timed_mutex verrou;


  // the actual thread pool
class ThreadPool {
public:
    ThreadPool(size_t nb_threads);
    template<class F,typename ...Args> void enqueue(F&& f,Args&& ...args);
    ~ThreadPool();
    void addCompletedTask();
    void drainCounter();
    int getNumberTasks();
    void temporaryWait();
    void drainCounterAndSetLimitTasks(int limitTasks);

private:
    friend class Worker;
 
    // need to keep track of threads so we can join them
    std::vector< std::thread > workers;
 
    // the task queue
    std::deque< std::function<void()> > tasks;
    // synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::condition_variable conditionNbTasks;
    bool stop;
    bool m_waitFlag;
    int m_counterCompletedTask;
    int m_limitCompletedTask;
};

// add new work item to the pool
template<class F,typename ...Args> void ThreadPool::enqueue(F&& f,Args&& ... args)
{
    { // acquire lock
        std::unique_lock<std::mutex> lock(queue_mutex);
         
        // add the task
        tasks.push_back(std::function<void()>(std::bind(f,args...)));
    } // release lock
     
    // wake up one thread
    condition.notify_one();
}


// our worker thread objects
class Worker {
public:
    Worker(ThreadPool &s) : pool(s) { }
    void operator()();
private:
    ThreadPool &pool;
};
#endif

