/**
 * 测量调度器各部分开销
 */

#include <iostream>
#include <chrono>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

using namespace std::chrono;

const int NUM_OPS = 100000;
const int NUM_RUNS = 5;

// 模拟调度器的数据结构
struct Operation {
    int id;
    std::atomic<bool> completed{false};
    std::mutex mtx;
    std::condition_variable cv;
};

std::queue<Operation*> op_queue;
std::mutex queue_mutex;
std::condition_variable queue_cv;
std::atomic<bool> running{true};

void scheduler_thread() {
    while (running.load()) {
        Operation* op = nullptr;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (!queue_cv.wait_for(lock, microseconds(10), [] { 
                return !op_queue.empty() || !running.load(); 
            })) {
                continue;
            }
            if (!op_queue.empty()) {
                op = op_queue.front();
                op_queue.pop();
            }
        }
        if (op) {
            // 模拟执行
            op->completed.store(true);
            op->cv.notify_one();
        }
    }
}

double measure_mutex_lock() {
    std::mutex mtx;
    auto start = high_resolution_clock::now();
    for (int i = 0; i < NUM_OPS; i++) {
        std::lock_guard<std::mutex> lock(mtx);
    }
    auto end = high_resolution_clock::now();
    return duration_cast<nanoseconds>(end - start).count() / (double)NUM_OPS;
}

double measure_atomic_ops() {
    std::atomic<int> counter{0};
    auto start = high_resolution_clock::now();
    for (int i = 0; i < NUM_OPS; i++) {
        counter.fetch_add(1);
        counter.load();
    }
    auto end = high_resolution_clock::now();
    return duration_cast<nanoseconds>(end - start).count() / (double)NUM_OPS;
}

double measure_queue_push_pop() {
    std::queue<int> q;
    std::mutex mtx;
    
    auto start = high_resolution_clock::now();
    for (int i = 0; i < NUM_OPS; i++) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            q.push(i);
        }
        {
            std::lock_guard<std::mutex> lock(mtx);
            q.pop();
        }
    }
    auto end = high_resolution_clock::now();
    return duration_cast<nanoseconds>(end - start).count() / (double)NUM_OPS;
}

double measure_cv_notify_wait() {
    std::mutex mtx;
    std::condition_variable cv;
    bool ready = false;
    std::atomic<bool> done{false};
    
    std::thread worker([&]() {
        while (!done.load()) {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&] { return ready || done.load(); });
            ready = false;
            cv.notify_one();
        }
    });
    
    auto start = high_resolution_clock::now();
    for (int i = 0; i < NUM_OPS; i++) {
        {
            std::unique_lock<std::mutex> lock(mtx);
            ready = true;
        }
        cv.notify_one();
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&] { return !ready; });
        }
    }
    auto end = high_resolution_clock::now();
    
    done.store(true);
    cv.notify_one();
    worker.join();
    
    return duration_cast<nanoseconds>(end - start).count() / (double)NUM_OPS;
}

double measure_full_scheduler_roundtrip() {
    running.store(true);
    std::thread sched(scheduler_thread);
    
    // Warmup
    for (int i = 0; i < 1000; i++) {
        Operation op;
        op.id = i;
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            op_queue.push(&op);
        }
        queue_cv.notify_one();
        {
            std::unique_lock<std::mutex> lock(op.mtx);
            op.cv.wait(lock, [&] { return op.completed.load(); });
        }
    }
    
    auto start = high_resolution_clock::now();
    for (int i = 0; i < NUM_OPS; i++) {
        Operation op;
        op.id = i;
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            op_queue.push(&op);
        }
        queue_cv.notify_one();
        {
            std::unique_lock<std::mutex> lock(op.mtx);
            op.cv.wait(lock, [&] { return op.completed.load(); });
        }
    }
    auto end = high_resolution_clock::now();
    
    running.store(false);
    queue_cv.notify_one();
    sched.join();
    
    return duration_cast<nanoseconds>(end - start).count() / (double)NUM_OPS;
}

double measure_memory_allocation() {
    auto start = high_resolution_clock::now();
    for (int i = 0; i < NUM_OPS; i++) {
        auto* op = new Operation();
        delete op;
    }
    auto end = high_resolution_clock::now();
    return duration_cast<nanoseconds>(end - start).count() / (double)NUM_OPS;
}

int main() {
    std::cout << "============================================================\n";
    std::cout << "SCHEDULER OVERHEAD BREAKDOWN (C++)\n";
    std::cout << "============================================================\n";
    std::cout << "Operations: " << NUM_OPS << "\n\n";
    
    double mutex_ns = measure_mutex_lock();
    std::cout << "1. Mutex lock/unlock:       " << mutex_ns << " ns/op\n";
    
    double atomic_ns = measure_atomic_ops();
    std::cout << "2. Atomic fetch_add+load:   " << atomic_ns << " ns/op\n";
    
    double alloc_ns = measure_memory_allocation();
    std::cout << "3. new/delete Operation:    " << alloc_ns << " ns/op\n";
    
    double queue_ns = measure_queue_push_pop();
    std::cout << "4. Queue push+pop (locked): " << queue_ns << " ns/op\n";
    
    double cv_ns = measure_cv_notify_wait();
    std::cout << "5. CV notify+wait:          " << cv_ns << " ns/op\n";
    
    double full_ns = measure_full_scheduler_roundtrip();
    std::cout << "6. Full roundtrip:          " << full_ns << " ns/op\n";
    
    std::cout << "\n============================================================\n";
    std::cout << "SUMMARY\n";
    std::cout << "============================================================\n";
    std::cout << "Total per-operation overhead: " << full_ns / 1000.0 << " us\n";
    std::cout << "\nBreakdown estimate:\n";
    std::cout << "  - Memory allocation:  " << alloc_ns << " ns\n";
    std::cout << "  - Queue operations:   " << queue_ns << " ns\n";
    std::cout << "  - CV sync:            " << cv_ns << " ns\n";
    std::cout << "  - Other:              " << (full_ns - alloc_ns - queue_ns - cv_ns) << " ns\n";
    
    return 0;
}
