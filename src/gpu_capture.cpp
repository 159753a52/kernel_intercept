#include "gpu_capture.h"
#include <cstdlib>
#include <cstring>
#include <thread>
#include <chrono>

namespace orion {

// ============================================================================
// 全局变量定义
// ============================================================================

CaptureLayerState g_capture_state;
LogLevel g_log_level = LogLevel::INFO;

// Thread-local client index
static thread_local int tl_client_idx = -1;

// ============================================================================
// 日志初始化
// ============================================================================

void init_log_level() {
    const char* level_str = std::getenv("ORION_LOG_LEVEL");
    if (level_str) {
        if (strcmp(level_str, "NONE") == 0 || strcmp(level_str, "0") == 0) {
            g_log_level = LogLevel::NONE;
        } else if (strcmp(level_str, "ERROR") == 0 || strcmp(level_str, "1") == 0) {
            g_log_level = LogLevel::ERROR;
        } else if (strcmp(level_str, "WARN") == 0 || strcmp(level_str, "2") == 0) {
            g_log_level = LogLevel::WARN;
        } else if (strcmp(level_str, "INFO") == 0 || strcmp(level_str, "3") == 0) {
            g_log_level = LogLevel::INFO;
        } else if (strcmp(level_str, "DEBUG") == 0 || strcmp(level_str, "4") == 0) {
            g_log_level = LogLevel::DEBUG;
        } else if (strcmp(level_str, "TRACE") == 0 || strcmp(level_str, "5") == 0) {
            g_log_level = LogLevel::TRACE;
        }
    }
}

// ============================================================================
// 拦截层初始化和关闭
// ============================================================================

int init_capture_layer(int num_clients) {
    if (g_capture_state.initialized.load()) {
        LOG_WARN("Capture layer already initialized");
        return 0;
    }
    
    if (num_clients <= 0 || num_clients > MAX_CLIENTS) {
        LOG_ERROR("Invalid num_clients: %d (must be 1-%d)", num_clients, MAX_CLIENTS);
        return -1;
    }
    
    init_log_level();
    
    g_capture_state.num_clients = num_clients;
    
    // 初始化 per-client 队列
    g_capture_state.client_queues.resize(num_clients);
    for (int i = 0; i < num_clients; i++) {
        g_capture_state.client_queues[i] = std::make_unique<ClientQueue>();
    }
    
    // 初始化同步原语 (使用 new 分配数组)
    g_capture_state.client_blocked = new std::atomic<bool>[num_clients];
    g_capture_state.client_mutexes = new std::mutex[num_clients];
    g_capture_state.client_cvs = new std::condition_variable[num_clients];
    
    for (int i = 0; i < num_clients; i++) {
        g_capture_state.client_blocked[i].store(false);
    }
    
    g_capture_state.shutdown.store(false);
    g_capture_state.initialized.store(true);
    g_capture_state.enabled.store(true);
    
    LOG_INFO("Capture layer initialized with %d clients", num_clients);
    return 0;
}

void shutdown_capture_layer() {
    if (!g_capture_state.initialized.load()) {
        return;
    }
    
    LOG_INFO("Shutting down capture layer...");
    
    g_capture_state.shutdown.store(true);
    g_capture_state.enabled.store(false);
    
    // 唤醒所有等待的 client
    for (int i = 0; i < g_capture_state.num_clients; i++) {
        g_capture_state.client_blocked[i].store(false);
        g_capture_state.client_cvs[i].notify_all();
        g_capture_state.client_queues[i]->shutdown();
    }
    
    // 唤醒调度器
    g_capture_state.scheduler_cv.notify_all();
    
    g_capture_state.initialized.store(false);
    LOG_INFO("Capture layer shutdown complete");
}

// ============================================================================
// Client 管理
// ============================================================================

int get_current_client_idx() {
    return tl_client_idx;
}

void set_current_client_idx(int idx) {
    if (idx < -1 || idx >= g_capture_state.num_clients) {
        LOG_ERROR("Invalid client index: %d", idx);
        return;
    }
    tl_client_idx = idx;
    LOG_DEBUG("Thread %lu set to client %d", 
              (unsigned long)std::hash<std::thread::id>{}(std::this_thread::get_id()), idx);
}

bool is_managed_thread() {
    return tl_client_idx >= 0 && g_capture_state.enabled.load();
}

bool is_capture_enabled() {
    return g_capture_state.initialized.load() && g_capture_state.enabled.load();
}

void set_capture_enabled(bool enabled) {
    g_capture_state.enabled.store(enabled);
    LOG_INFO("Capture %s", enabled ? "enabled" : "disabled");
}

// ============================================================================
// 操作提交
// ============================================================================

OperationPtr create_operation(int client_idx, OperationType type) {
    if (client_idx < 0 || client_idx >= g_capture_state.num_clients) {
        LOG_ERROR("Invalid client index for create: %d", client_idx);
        return nullptr;
    }
    
    auto op = std::make_shared<OperationRecord>();
    op->type = type;
    op->client_idx = client_idx;
    op->op_id = g_capture_state.next_op_id.fetch_add(1);
    
    LOG_DEBUG("Created op %lu type %s for client %d", op->op_id, op_type_name(type), client_idx);
    
    return op;
}

void enqueue_operation(OperationPtr op) {
    if (!op) return;
    
    LOG_TRACE("Client %d enqueuing op %lu type %s",
              op->client_idx, op->op_id, op_type_name(op->type));
    
    g_capture_state.client_queues[op->client_idx]->push(op);
    notify_scheduler();
}

// 兼容旧接口
OperationPtr submit_operation(int client_idx, OperationType type) {
    auto op = create_operation(client_idx, type);
    if (op) {
        enqueue_operation(op);
    }
    return op;
}

void wait_operation(OperationPtr op) {
    if (!op) return;
    op->wait_completion();
}

// ============================================================================
// Block/Unblock 机制
// ============================================================================

extern "C" void block(int phase) {
    int client_idx = get_current_client_idx();
    if (client_idx < 0) {
        LOG_DEBUG("block(%d) called from unmanaged thread, ignoring", phase);
        return;
    }
    
    if (g_capture_state.shutdown.load()) {
        return;
    }
    
    LOG_TRACE("Client %d blocking at phase %d", client_idx, phase);
    
    // 设置 blocked 标志
    g_capture_state.client_blocked[client_idx].store(true);
    
    // 等待被 unblock
    std::unique_lock<std::mutex> lock(g_capture_state.client_mutexes[client_idx]);
    g_capture_state.client_cvs[client_idx].wait(lock, [client_idx] {
        return !g_capture_state.client_blocked[client_idx].load() ||
               g_capture_state.shutdown.load();
    });
    
    LOG_TRACE("Client %d unblocked from phase %d", client_idx, phase);
}

void unblock_client(int client_idx) {
    if (client_idx < 0 || client_idx >= g_capture_state.num_clients) {
        return;
    }
    
    g_capture_state.client_blocked[client_idx].store(false);
    g_capture_state.client_cvs[client_idx].notify_all();
    
    LOG_TRACE("Unblocked client %d", client_idx);
}

void notify_scheduler() {
    g_capture_state.scheduler_cv.notify_one();
}

// ============================================================================
// 等待操作完成的辅助函数
// ============================================================================

// block_until_allowed: 在 wrapper 中调用，等待调度器允许执行
// 这个函数实现了 Orion 论文中的 block() 语义
void block_until_allowed(int client_idx, int phase) {
    if (client_idx < 0 || !g_capture_state.enabled.load()) {
        return;
    }
    
    if (g_capture_state.shutdown.load()) {
        return;
    }
    
    LOG_TRACE("Client %d waiting for phase %d", client_idx, phase);
    
    // 设置等待标志
    g_capture_state.client_blocked[client_idx].store(true);
    
    // 自旋等待 + 条件变量混合策略
    // 先自旋一小段时间，如果还没被 unblock 则进入等待
    constexpr int SPIN_COUNT = 1000;
    for (int i = 0; i < SPIN_COUNT; i++) {
        if (!g_capture_state.client_blocked[client_idx].load() ||
            g_capture_state.shutdown.load()) {
            return;
        }
        std::this_thread::yield();
    }
    
    // 进入条件变量等待
    std::unique_lock<std::mutex> lock(g_capture_state.client_mutexes[client_idx]);
    g_capture_state.client_cvs[client_idx].wait(lock, [client_idx] {
        return !g_capture_state.client_blocked[client_idx].load() ||
               g_capture_state.shutdown.load();
    });
}

} // namespace orion

// ============================================================================
// C 接口 (用于 LD_PRELOAD 和 Python ctypes)
// ============================================================================

extern "C" {

int orion_init(int num_clients) {
    return orion::init_capture_layer(num_clients);
}

void orion_shutdown() {
    orion::shutdown_capture_layer();
}

void orion_set_client_idx(int idx) {
    orion::set_current_client_idx(idx);
}

int orion_get_client_idx() {
    return orion::get_current_client_idx();
}

void orion_set_enabled(int enabled) {
    orion::set_capture_enabled(enabled != 0);
}

int orion_is_enabled() {
    return orion::is_capture_enabled() ? 1 : 0;
}

} // extern "C"
