#include <iostream>
#include <chrono>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

using namespace std::chrono;

const int NUM_OPS = 10000;

double measure_mutex_lock() {
    std::mutex mtx;
    auto start = high_resolution_clock::now();
    for (int i = 0; i < NUM_OPS; i++) {
        std::lock_guard<std::mutex> lock(mtx);
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

double measure_memory_allocation() {
    struct Op { int id; std::atomic<bool> done{false}; };
    auto start = high_resolution_clock::now();
    for (int i = 0; i < NUM_OPS; i++) {
        auto* op = new Op();
        delete op;
    }
    auto end = high_resolution_clock::now();
    return duration_cast<nanoseconds>(end - start).count() / (double)NUM_OPS;
}

double measure_cv_roundtrip() {
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> ready{false};
    std::atomic<bool> done{false};
    std::atomic<bool> running{true};
    
    std::thread worker([&]() {
        while (running.load()) {
            std::unique_lock<std::mutex> lock(mtx);
            if (cv.wait_for(lock, std::chrono::microseconds(100), [&] { return ready.load(); })) {
                ready.store(false);
                done.store(true);
                cv.notify_one();
            }
        }
    });
    
    // Warmup
    for (int i = 0; i < 100; i++) {
        ready.store(true);
        cv.notify_one();
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] { return done.load(); });
        done.store(false);
    }
    
    auto start = high_resolution_clock::now();
    for (int i = 0; i < NUM_OPS; i++) {
        ready.store(true);
        cv.notify_one();
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] { return done.load(); });
        done.store(false);
    }
    auto end = high_resolution_clock::now();
    
    running.store(false);
    cv.notify_one();
    worker.join();
    
    return duration_cast<nanoseconds>(end - start).count() / (double)NUM_OPS;
}

int main() {
    std::cout << "============================================================\n";
    std::cout << "SCHEDULER OVERHEAD BREAKDOWN\n";
    std::cout << "============================================================\n";
    
    double mutex_ns = measure_mutex_lock();
    std::cout << "1. Mutex lock/unlock:       " << mutex_ns << " ns\n";
    
    double alloc_ns = measure_memory_allocation();
    std::cout << "2. new/delete Operation:    " << alloc_ns << " ns\n";
    
    double queue_ns = measure_queue_push_pop();
    std::cout << "3. Queue push+pop (locked): " << queue_ns << " ns\n";
    
    double cv_ns = measure_cv_roundtrip();
    std::cout << "4. CV roundtrip:            " << cv_ns << " ns\n";
    
    double total = alloc_ns + queue_ns + cv_ns;
    std::cout << "\nEstimated total overhead:   " << total << " ns = " << total/1000 << " us\n";
    
    return 0;
}
