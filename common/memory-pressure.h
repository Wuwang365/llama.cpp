#pragma once

#include "llama.h"

#include <climits>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

static constexpr int COMMON_MEMORY_PRESSURE_LAYER_OUTPUT = INT_MAX;

enum common_memory_pressure_policy_type {
    COMMON_MEMORY_PRESSURE_POLICY_PRESSURE_ONLY,
    COMMON_MEMORY_PRESSURE_POLICY_EXTERNAL_HINT,
};

enum common_memory_pressure_state {
    COMMON_MEMORY_PRESSURE_STATE_DISABLED,
    COMMON_MEMORY_PRESSURE_STATE_OBSERVING,
    COMMON_MEMORY_PRESSURE_STATE_RELIEVING,
    COMMON_MEMORY_PRESSURE_STATE_COOLDOWN,
    COMMON_MEMORY_PRESSURE_STATE_SATURATED,
    COMMON_MEMORY_PRESSURE_STATE_RESTORE_PENDING,
};

enum common_memory_pressure_level {
    COMMON_MEMORY_PRESSURE_LEVEL_LOW,
    COMMON_MEMORY_PRESSURE_LEVEL_MEDIUM,
    COMMON_MEMORY_PRESSURE_LEVEL_HIGH,
    COMMON_MEMORY_PRESSURE_LEVEL_CRITICAL,
};

enum common_memory_pressure_action {
    COMMON_MEMORY_PRESSURE_ACTION_DISABLED,
    COMMON_MEMORY_PRESSURE_ACTION_NO_ACTION,
    COMMON_MEMORY_PRESSURE_ACTION_OBSERVE_ONLY,
    COMMON_MEMORY_PRESSURE_ACTION_UNLOAD,
    COMMON_MEMORY_PRESSURE_ACTION_DEFER_BUSY,
    COMMON_MEMORY_PRESSURE_ACTION_SATURATED,
};

struct common_memory_pressure_psi_line {
    double avg10  = 0.0;
    double avg60  = 0.0;
    double avg300 = 0.0;
    uint64_t total = 0;
};

struct common_memory_pressure_sample {
    bool has_some = false;
    bool has_full = false;
    common_memory_pressure_psi_line some;
    common_memory_pressure_psi_line full;
};

struct common_memory_pressure_bytes {
    size_t resident = 0;
    size_t unloaded = 0;

    size_t total() const {
        return resident + unloaded;
    }
};

struct common_memory_pressure_config {
    bool enabled = false;
    std::string path = "/proc/pressure/memory";
    int32_t interval_ms = 500;
    int32_t cooldown_ms = 1000;
    size_t step_bytes = 128ull * 1024ull * 1024ull;
    double max_fraction = 0.80;
    double some_avg10_thold = 1.0;
    double some_avg60_thold = 2.0;
    double full_avg10_thold = 0.0;
    uint64_t some_total_delta_us_thold = 10000;
    common_memory_pressure_policy_type policy = COMMON_MEMORY_PRESSURE_POLICY_PRESSURE_ONLY;
    bool protected_app_active = true;
    bool dry_run = false;
    std::string log_path;
    std::string keep_regex;
};

struct common_memory_pressure_decision {
    common_memory_pressure_action action = COMMON_MEMORY_PRESSURE_ACTION_NO_ACTION;
    common_memory_pressure_state state = COMMON_MEMORY_PRESSURE_STATE_OBSERVING;
    common_memory_pressure_level level = COMMON_MEMORY_PRESSURE_LEVEL_LOW;
    size_t target_bytes = 0;
    std::string reason;
};

struct common_memory_pressure_weight {
    std::string name;
    size_t nbytes = 0;
    int layer = -1;
    bool loaded = false;
};

struct common_memory_pressure_selection {
    std::vector<common_memory_pressure_weight> selected;
    size_t target_bytes = 0;
    size_t selected_bytes = 0;
    size_t resident_bytes = 0;
    size_t unloaded_bytes = 0;
    size_t max_unload_bytes = 0;
    size_t skipped_keep = 0;
    size_t skipped_unloaded = 0;
    bool saturated = false;
};

std::string common_memory_pressure_default_keep_regex();

bool common_memory_pressure_parse_psi(
        const std::string & text,
        common_memory_pressure_sample & sample,
        std::string & err);

bool common_memory_pressure_read_psi(
        const std::string & path,
        common_memory_pressure_sample & sample,
        std::string & err);

bool common_memory_pressure_select_weights(
        const std::vector<common_memory_pressure_weight> & weights,
        const common_memory_pressure_config & config,
        size_t target_bytes,
        common_memory_pressure_selection & selection,
        std::string & err);

bool common_memory_pressure_collect_weights(
        const llama_model * model,
        std::vector<common_memory_pressure_weight> & weights,
        common_memory_pressure_bytes & bytes,
        std::string & err);

class common_memory_pressure_policy {
public:
    common_memory_pressure_decision decide(
            const common_memory_pressure_config & config,
            const common_memory_pressure_sample & sample,
            const common_memory_pressure_bytes & bytes,
            bool busy,
            int64_t now_ms);

    void record_unload(int64_t now_ms);
    void reset();

private:
    bool has_last_sample = false;
    common_memory_pressure_sample last_sample;
    int64_t last_unload_ms = -1;
};

class common_memory_pressure_manager {
public:
    common_memory_pressure_manager(
            const common_memory_pressure_config & config,
            llama_model * model);
    ~common_memory_pressure_manager();

    common_memory_pressure_manager(const common_memory_pressure_manager &) = delete;
    common_memory_pressure_manager & operator=(const common_memory_pressure_manager &) = delete;

    bool enabled() const;
    bool tick(bool busy, std::string * err = nullptr);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
