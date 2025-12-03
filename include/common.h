#ifndef ORION_COMMON_H
#define ORION_COMMON_H

#include <cstdint>
#include <atomic>
#include <string>

namespace orion {

// 最大支持的 client 数量
constexpr int MAX_CLIENTS = 16;

// 操作类型枚举
enum class OperationType : uint8_t {
    KERNEL_LAUNCH = 0,
    MALLOC,
    FREE,
    MEMCPY,
    MEMCPY_ASYNC,
    MEMSET,
    MEMSET_ASYNC,
    DEVICE_SYNC,
    STREAM_SYNC,
    EVENT_SYNC,
    CUDNN_CONV_FWD,
    CUDNN_CONV_BWD_DATA,
    CUDNN_CONV_BWD_FILTER,
    CUDNN_BATCHNORM_FWD,
    CUDNN_BATCHNORM_BWD,
    CUBLAS_SGEMM,
    CUBLAS_SGEMM_BATCHED,
    CUBLAS_SGEMM_STRIDED_BATCHED,
    UNKNOWN
};

// Profile 类型
enum class ProfileType : uint8_t {
    COMPUTE_BOUND = 0,
    MEMORY_BOUND,
    UNKNOWN
};

// 获取操作类型名称
inline const char* op_type_name(OperationType type) {
    switch (type) {
        case OperationType::KERNEL_LAUNCH: return "KERNEL_LAUNCH";
        case OperationType::MALLOC: return "MALLOC";
        case OperationType::FREE: return "FREE";
        case OperationType::MEMCPY: return "MEMCPY";
        case OperationType::MEMCPY_ASYNC: return "MEMCPY_ASYNC";
        case OperationType::MEMSET: return "MEMSET";
        case OperationType::MEMSET_ASYNC: return "MEMSET_ASYNC";
        case OperationType::DEVICE_SYNC: return "DEVICE_SYNC";
        case OperationType::STREAM_SYNC: return "STREAM_SYNC";
        case OperationType::EVENT_SYNC: return "EVENT_SYNC";
        case OperationType::CUDNN_CONV_FWD: return "CUDNN_CONV_FWD";
        case OperationType::CUDNN_CONV_BWD_DATA: return "CUDNN_CONV_BWD_DATA";
        case OperationType::CUDNN_CONV_BWD_FILTER: return "CUDNN_CONV_BWD_FILTER";
        case OperationType::CUDNN_BATCHNORM_FWD: return "CUDNN_BATCHNORM_FWD";
        case OperationType::CUDNN_BATCHNORM_BWD: return "CUDNN_BATCHNORM_BWD";
        case OperationType::CUBLAS_SGEMM: return "CUBLAS_SGEMM";
        case OperationType::CUBLAS_SGEMM_BATCHED: return "CUBLAS_SGEMM_BATCHED";
        case OperationType::CUBLAS_SGEMM_STRIDED_BATCHED: return "CUBLAS_SGEMM_STRIDED_BATCHED";
        default: return "UNKNOWN";
    }
}

// 日志级别
enum class LogLevel : uint8_t {
    NONE = 0,
    ERROR,
    WARN,
    INFO,
    DEBUG,
    TRACE
};

// 全局日志级别 (可通过环境变量 ORION_LOG_LEVEL 设置)
extern LogLevel g_log_level;

// 初始化日志级别
void init_log_level();

// 日志宏
#define ORION_LOG(level, fmt, ...) \
    do { \
        if (static_cast<uint8_t>(level) <= static_cast<uint8_t>(orion::g_log_level)) { \
            fprintf(stderr, "[ORION][%s] " fmt "\n", \
                    #level, ##__VA_ARGS__); \
            fflush(stderr); \
        } \
    } while(0)

#define LOG_ERROR(fmt, ...) ORION_LOG(orion::LogLevel::ERROR, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  ORION_LOG(orion::LogLevel::WARN, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)  ORION_LOG(orion::LogLevel::INFO, fmt, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) ORION_LOG(orion::LogLevel::DEBUG, fmt, ##__VA_ARGS__)
#define LOG_TRACE(fmt, ...) ORION_LOG(orion::LogLevel::TRACE, fmt, ##__VA_ARGS__)

} // namespace orion

#endif // ORION_COMMON_H
