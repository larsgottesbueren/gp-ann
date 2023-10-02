#pragma once

#include <atomic>

class SpinLock {
public:
    // boilerplate to make it 'copyable'. but we just clear the spinlock. there is never a use case to copy a locked spinlock
    SpinLock() { }
    SpinLock(const SpinLock&) { }
    SpinLock& operator=(const SpinLock&) { spinner.clear(std::memory_order_relaxed); return *this; }

    bool tryLock() {
        return !spinner.test_and_set(std::memory_order_acquire);
    }

    void lock() {
        while (spinner.test_and_set(std::memory_order_acquire)) {
            // spin
            // stack overflow says adding 'cpu_relax' instruction may improve performance
        }
    }

    void unlock() {
        spinner.clear(std::memory_order_release);
    }

private:
    std::atomic_flag spinner = ATOMIC_FLAG_INIT;
};
