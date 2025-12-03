#include "kernel_profile.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cstring>

namespace orion {

// 全局 profiler 实例
KernelProfiler g_profiler;

// ============================================================================
// 简单 JSON 解析器 (避免外部依赖)
// ============================================================================

namespace json {

// 去除字符串两端空白
static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    size_t end = s.find_last_not_of(" \t\n\r");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

// 提取引号内的字符串
static std::string extract_string(const std::string& s, size_t& pos) {
    if (pos >= s.length() || s[pos] != '"') return "";
    size_t start = ++pos;
    while (pos < s.length() && s[pos] != '"') {
        if (s[pos] == '\\' && pos + 1 < s.length()) pos++;
        pos++;
    }
    std::string result = s.substr(start, pos - start);
    if (pos < s.length()) pos++;  // skip closing quote
    return result;
}

// 提取数值
static double extract_number(const std::string& s, size_t& pos) {
    size_t start = pos;
    while (pos < s.length() && (isdigit(s[pos]) || s[pos] == '.' || s[pos] == '-' || s[pos] == 'e' || s[pos] == 'E' || s[pos] == '+')) {
        pos++;
    }
    return std::stod(s.substr(start, pos - start));
}

// 跳过空白
static void skip_whitespace(const std::string& s, size_t& pos) {
    while (pos < s.length() && isspace(s[pos])) pos++;
}

} // namespace json

// ============================================================================
// KernelProfileTable 实现
// ============================================================================

KernelProfileTable::KernelProfileTable() {}

bool KernelProfileTable::load_from_json(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open profile file: %s", filepath.c_str());
        return false;
    }
    
    // 读取整个文件
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    file.close();
    
    // 简单解析 (不是完整的 JSON 解析器，但足够处理我们的格式)
    size_t pos = 0;
    
    // 查找 "kernels" 数组
    size_t kernels_pos = content.find("\"kernels\"");
    if (kernels_pos == std::string::npos) {
        LOG_ERROR("No 'kernels' array found in profile file");
        return false;
    }
    
    // 找到数组开始的 '['
    pos = content.find('[', kernels_pos);
    if (pos == std::string::npos) {
        LOG_ERROR("Invalid kernels array format");
        return false;
    }
    pos++;  // skip '['
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    while (pos < content.length()) {
        json::skip_whitespace(content, pos);
        
        if (content[pos] == ']') break;  // 数组结束
        if (content[pos] == ',') { pos++; continue; }
        if (content[pos] != '{') { pos++; continue; }
        
        // 解析一个 kernel 对象
        size_t obj_end = content.find('}', pos);
        if (obj_end == std::string::npos) break;
        
        std::string obj_str = content.substr(pos, obj_end - pos + 1);
        pos = obj_end + 1;
        
        KernelProfile profile;
        
        // 提取字段
        size_t field_pos;
        
        // kernel_id
        field_pos = obj_str.find("\"kernel_id\"");
        if (field_pos != std::string::npos) {
            field_pos = obj_str.find(':', field_pos);
            field_pos = obj_str.find('"', field_pos);
            if (field_pos != std::string::npos) {
                profile.kernel_id = json::extract_string(obj_str, field_pos);
            }
        }
        
        // duration_ms
        field_pos = obj_str.find("\"duration_ms\"");
        if (field_pos != std::string::npos) {
            field_pos = obj_str.find(':', field_pos);
            json::skip_whitespace(obj_str, ++field_pos);
            profile.duration_ms = json::extract_number(obj_str, field_pos);
        }
        
        // sm_needed
        field_pos = obj_str.find("\"sm_needed\"");
        if (field_pos != std::string::npos) {
            field_pos = obj_str.find(':', field_pos);
            json::skip_whitespace(obj_str, ++field_pos);
            profile.sm_needed = static_cast<int>(json::extract_number(obj_str, field_pos));
        }
        
        // profile_type
        field_pos = obj_str.find("\"profile_type\"");
        if (field_pos != std::string::npos) {
            field_pos = obj_str.find(':', field_pos);
            field_pos = obj_str.find('"', field_pos);
            if (field_pos != std::string::npos) {
                std::string type_str = json::extract_string(obj_str, field_pos);
                if (type_str == "compute") {
                    profile.profile_type = ProfileType::COMPUTE_BOUND;
                } else if (type_str == "memory") {
                    profile.profile_type = ProfileType::MEMORY_BOUND;
                } else {
                    profile.profile_type = ProfileType::UNKNOWN;
                }
            }
        }
        
        // grid_size (可选)
        field_pos = obj_str.find("\"grid_size\"");
        if (field_pos != std::string::npos) {
            field_pos = obj_str.find(':', field_pos);
            json::skip_whitespace(obj_str, ++field_pos);
            profile.grid_size = static_cast<int>(json::extract_number(obj_str, field_pos));
        }
        
        // block_size (可选)
        field_pos = obj_str.find("\"block_size\"");
        if (field_pos != std::string::npos) {
            field_pos = obj_str.find(':', field_pos);
            json::skip_whitespace(obj_str, ++field_pos);
            profile.block_size = static_cast<int>(json::extract_number(obj_str, field_pos));
        }
        
        if (!profile.kernel_id.empty()) {
            profiles_[profile.kernel_id] = profile;
            LOG_DEBUG("Loaded profile: %s (%.3f ms, %d SMs, type=%d)",
                      profile.kernel_id.c_str(), profile.duration_ms,
                      profile.sm_needed, (int)profile.profile_type);
        }
    }
    
    LOG_INFO("Loaded %zu kernel profiles from %s", profiles_.size(), filepath.c_str());
    return true;
}

bool KernelProfileTable::load_from_yaml(const std::string& filepath) {
    // YAML 解析类似，这里简化实现
    // 实际应用中可以使用 yaml-cpp 库
    LOG_WARN("YAML parsing not fully implemented, using simplified parser");
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open profile file: %s", filepath.c_str());
        return false;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    KernelProfile current;
    bool in_kernels = false;
    std::string line;
    
    while (std::getline(file, line)) {
        line = json::trim(line);
        if (line.empty() || line[0] == '#') continue;
        
        if (line.find("kernels:") != std::string::npos) {
            in_kernels = true;
            continue;
        }
        
        if (in_kernels && line[0] == '-') {
            // 新的 kernel 条目
            if (!current.kernel_id.empty()) {
                profiles_[current.kernel_id] = current;
            }
            current = KernelProfile();
            line = json::trim(line.substr(1));
        }
        
        if (in_kernels) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                std::string key = json::trim(line.substr(0, colon));
                std::string value = json::trim(line.substr(colon + 1));
                
                if (key == "kernel_id") {
                    current.kernel_id = value;
                } else if (key == "duration_ms") {
                    current.duration_ms = std::stof(value);
                } else if (key == "sm_needed") {
                    current.sm_needed = std::stoi(value);
                } else if (key == "profile_type") {
                    if (value == "compute") {
                        current.profile_type = ProfileType::COMPUTE_BOUND;
                    } else if (value == "memory") {
                        current.profile_type = ProfileType::MEMORY_BOUND;
                    }
                }
            }
        }
    }
    
    if (!current.kernel_id.empty()) {
        profiles_[current.kernel_id] = current;
    }
    
    file.close();
    LOG_INFO("Loaded %zu kernel profiles from %s", profiles_.size(), filepath.c_str());
    return true;
}

const KernelProfile* KernelProfileTable::find(const std::string& kernel_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = profiles_.find(kernel_id);
    return (it != profiles_.end()) ? &it->second : nullptr;
}

void KernelProfileTable::add(const KernelProfile& profile) {
    std::lock_guard<std::mutex> lock(mutex_);
    profiles_[profile.kernel_id] = profile;
}

void KernelProfileTable::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    profiles_.clear();
}

float KernelProfileTable::compute_average_duration() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (profiles_.empty()) return 0.0f;
    
    float total = 0.0f;
    for (const auto& pair : profiles_) {
        total += pair.second.duration_ms;
    }
    return total / profiles_.size();
}

float KernelProfileTable::compute_recommended_dur_threshold(float target_ratio) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (profiles_.empty()) return 0.0f;
    
    // 计算总延迟作为一次推理请求的延迟估计
    float total_duration = 0.0f;
    for (const auto& pair : profiles_) {
        total_duration += pair.second.duration_ms;
    }
    
    return total_duration * target_ratio;
}

// ============================================================================
// KernelProfiler 实现
// ============================================================================

KernelProfiler::KernelProfiler() : session_active_(false) {}

KernelProfiler::~KernelProfiler() {
    if (session_active_) {
        end_session("");
    }
}

void KernelProfiler::start_session(const std::string& model_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    model_name_ = model_name;
    recorded_profiles_.clear();
    session_active_ = true;
    LOG_INFO("Started profiling session for model: %s", model_name.c_str());
}

void KernelProfiler::end_session(const std::string& output_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    session_active_ = false;
    
    if (output_path.empty()) {
        LOG_INFO("Profiling session ended, no output file specified");
        return;
    }
    
    // 输出为 JSON
    std::ofstream file(output_path);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open output file: %s", output_path.c_str());
        return;
    }
    
    file << "{\n";
    file << "    \"model_name\": \"" << model_name_ << "\",\n";
    file << "    \"kernels\": [\n";
    
    for (size_t i = 0; i < recorded_profiles_.size(); i++) {
        const auto& p = recorded_profiles_[i];
        file << "        {\n";
        file << "            \"kernel_id\": \"" << p.kernel_id << "\",\n";
        file << "            \"duration_ms\": " << p.duration_ms << ",\n";
        file << "            \"sm_needed\": " << p.sm_needed << ",\n";
        file << "            \"profile_type\": \"";
        switch (p.profile_type) {
            case ProfileType::COMPUTE_BOUND: file << "compute"; break;
            case ProfileType::MEMORY_BOUND: file << "memory"; break;
            default: file << "unknown"; break;
        }
        file << "\"\n";
        file << "        }";
        if (i < recorded_profiles_.size() - 1) file << ",";
        file << "\n";
    }
    
    file << "    ]\n";
    file << "}\n";
    
    file.close();
    LOG_INFO("Profiling session ended, %zu profiles written to %s", 
             recorded_profiles_.size(), output_path.c_str());
}

void KernelProfiler::record_kernel(const std::string& kernel_id, 
                                    float duration_ms,
                                    int sm_needed,
                                    ProfileType profile_type) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!session_active_) return;
    
    KernelProfile profile;
    profile.kernel_id = kernel_id;
    profile.duration_ms = duration_ms;
    profile.sm_needed = sm_needed;
    profile.profile_type = profile_type;
    
    recorded_profiles_.push_back(profile);
}

void KernelProfiler::get_profile_table(KernelProfileTable& table) const {
    std::lock_guard<std::mutex> lock(mutex_);
    table.clear();
    for (const auto& p : recorded_profiles_) {
        table.add(p);
    }
}

} // namespace orion

// ============================================================================
// C 接口
// ============================================================================

extern "C" {

int orion_load_profile(const char* filepath) {
    static orion::KernelProfileTable table;
    std::string path(filepath);
    
    if (path.find(".yaml") != std::string::npos || 
        path.find(".yml") != std::string::npos) {
        return table.load_from_yaml(path) ? 0 : -1;
    }
    return table.load_from_json(path) ? 0 : -1;
}

void orion_start_profiling(const char* model_name) {
    orion::g_profiler.start_session(model_name ? model_name : "unnamed");
}

void orion_end_profiling(const char* output_path) {
    orion::g_profiler.end_session(output_path ? output_path : "");
}

} // extern "C"
