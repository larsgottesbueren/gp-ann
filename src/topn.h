#pragma once

#include <queue>
#include <cstdint>

struct TopN {
    TopN(size_t n) : n(n) { }
    size_t n;
    using value_type = std::pair<float, uint32_t>;
    std::priority_queue<value_type> pq;

    void Add(value_type e) {
        if (pq.size() < n) {
            pq.push(e);
        } else if (pq.top() > e) {
            pq.pop();
            pq.push(e);
        }
    }

    std::vector<value_type> Take() {
        std::vector<value_type> res(pq.size());
        // heap-sort because we can't access the container without hacks via inheritance
        while (!pq.empty()) {
            res[pq.size() - 1] = pq.top();
            pq.pop();
        }
        return res;
    }

    const value_type& Top() const { return pq.top(); }
};
