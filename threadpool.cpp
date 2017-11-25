#include "threadpool.h"
std::timed_mutex verrou;

// the constructor just launches some amount of workers
ThreadPool::ThreadPool(size_t nb_threads) :   stop(false),m_counterCompletedTask(0),m_waitFlag(false),m_limitCompletedTask(0)
{
    for(size_t i = 0;i<nb_threads;++i)
        workers.push_back(std::thread(Worker(*this))); //it calls Worker which havec the operator ()
}
   
// the destructor joins all threads
ThreadPool::~ThreadPool()
{
    // stop all threads
    stop = true;
    condition.notify_all();
     
    // join them
    for(size_t i = 0;i<workers.size();++i)
        workers[i].join();
    // std::cout<<"allWorkers sleeping"<<std::endl;
}


void Worker::operator()()
{
    std::function<void()> task;
    while(true)
    {
        {   // acquire lock
            std::unique_lock<std::mutex> lock(pool.queue_mutex);
            // look for a work item
            while(!pool.stop && pool.tasks.empty())
            { // if there are none wait for notification
                pool.condition.wait(lock); // we need "while" and not "if" because if the pool's task is empty we need to wait
            }
 
            if(pool.stop && pool.tasks.empty()) // exit if the pool is stopped
                return;
 
            // get the task from the queue
            task = pool.tasks.front();
            pool.tasks.pop_front();
 
        }   // release lock
 
        // execute the task
        // std::lock_guard<std::timed_mutex> lck(verrou);
        task();
        pool.addCompletedTask();
  }
}

void ThreadPool::addCompletedTask()
{
    // if (m_counterCompletedTask < m_limitCompletedTask-1)
    // {
        // m_counterCompletedTask+=1;
    // }
    if (m_waitFlag==true and m_counterCompletedTask>=m_limitCompletedTask-1 )
    {
        // we completed all the tasks, we free the wait lock
        conditionNbTasks.notify_one();
    }
    m_counterCompletedTask+=1;
    // else if(m_limitCompletedTask=!0)
    // {
    //     throw std::invalid_argument( "problem with thread management" );
    // }

}
void ThreadPool::drainCounterAndSetLimitTasks(int limitTasks)
{
    m_counterCompletedTask=0;
    m_limitCompletedTask=limitTasks;
}
void ThreadPool::temporaryWait()
// here we stop the thread pool until m_limitCompletedTask is reached
{
    std::unique_lock<std::mutex> lock(queue_mutex);
    m_waitFlag=true;
    conditionNbTasks.wait(lock);
}
int ThreadPool::getNumberTasks()
{
    return m_counterCompletedTask;
}