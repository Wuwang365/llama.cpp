#include "memory-pressure.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>

static int64_t common_memory_pressure_now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

static const char * common_memory_pressure_action_name(common_memory_pressure_action action) {
    switch (action) {
        case COMMON_MEMORY_PRESSURE_ACTION_DISABLED:     return "disabled";
        case COMMON_MEMORY_PRESSURE_ACTION_NO_ACTION:    return "no_action";
        case COMMON_MEMORY_PRESSURE_ACTION_OBSERVE_ONLY: return "observe_only";
        case COMMON_MEMORY_PRESSURE_ACTION_UNLOAD:       return "unload";
        case COMMON_MEMORY_PRESSURE_ACTION_DEFER_BUSY:   return "defer_busy";
        case COMMON_MEMORY_PRESSURE_ACTION_SATURATED:    return "saturated";
    }
    return "unknown";
}

static const char * common_memory_pressure_state_name(common_memory_pressure_state state) {
    switch (state) {
        case COMMON_MEMORY_PRESSURE_STATE_DISABLED:        return "disabled";
        case COMMON_MEMORY_PRESSURE_STATE_OBSERVING:       return "observing";
        case COMMON_MEMORY_PRESSURE_STATE_RELIEVING:       return "relieving";
        case COMMON_MEMORY_PRESSURE_STATE_COOLDOWN:        return "cooldown";
        case COMMON_MEMORY_PRESSURE_STATE_SATURATED:       return "saturated";
        case COMMON_MEMORY_PRESSURE_STATE_RESTORE_PENDING: return "restore_pending";
    }
    return "unknown";
}

static const char * common_memory_pressure_level_name(common_memory_pressure_level level) {
    switch (level) {
        case COMMON_MEMORY_PRESSURE_LEVEL_LOW:      return "low";
        case COMMON_MEMORY_PRESSURE_LEVEL_MEDIUM:   return "medium";
        case COMMON_MEMORY_PRESSURE_LEVEL_HIGH:     return "high";
        case COMMON_MEMORY_PRESSURE_LEVEL_CRITICAL: return "critical";
    }
    return "unknown";
}

static std::string common_memory_pressure_json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

static bool common_memory_pressure_parse_double(
        const std::string & value,
        const char * field,
        double & out,
        std::string & err) {
    try {
        size_t pos = 0;
        out = std::stod(value, &pos);
        if (pos != value.size()) {
            err = std::string("invalid ") + field + ": " + value;
            return false;
        }
        return true;
    } catch (const std::exception &) {
        err = std::string("invalid ") + field + ": " + value;
        return false;
    }
}

static bool common_memory_pressure_parse_u64(
        const std::string & value,
        const char * field,
        uint64_t & out,
        std::string & err) {
    try {
        size_t pos = 0;
        out = std::stoull(value, &pos);
        if (pos != value.size()) {
            err = std::string("invalid ") + field + ": " + value;
            return false;
        }
        return true;
    } catch (const std::exception &) {
        err = std::string("invalid ") + field + ": " + value;
        return false;
    }
}

static bool common_memory_pressure_parse_line(
        const std::string & line,
        const char * expected_kind,
        common_memory_pressure_psi_line & out,
        std::string & err) {
    std::istringstream iss(line);
    std::string kind;
    iss >> kind;
    if (kind != expected_kind) {
        err = std::string("expected ") + expected_kind + " line";
        return false;
    }

    bool has_avg10 = false;
    bool has_avg60 = false;
    bool has_avg300 = false;
    bool has_total = false;

    std::string tok;
    while (iss >> tok) {
        const size_t eq = tok.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        const std::string key = tok.substr(0, eq);
        const std::string value = tok.substr(eq + 1);

        if (key == "avg10") {
            has_avg10 = common_memory_pressure_parse_double(value, "avg10", out.avg10, err);
            if (!has_avg10) {
                return false;
            }
        } else if (key == "avg60") {
            has_avg60 = common_memory_pressure_parse_double(value, "avg60", out.avg60, err);
            if (!has_avg60) {
                return false;
            }
        } else if (key == "avg300") {
            has_avg300 = common_memory_pressure_parse_double(value, "avg300", out.avg300, err);
            if (!has_avg300) {
                return false;
            }
        } else if (key == "total") {
            has_total = common_memory_pressure_parse_u64(value, "total", out.total, err);
            if (!has_total) {
                return false;
            }
        }
    }

    if (!has_avg10 || !has_avg60 || !has_avg300 || !has_total) {
        err = std::string("missing field in ") + expected_kind + " line";
        return false;
    }
    return true;
}

std::string common_memory_pressure_default_keep_regex() {
    return "(^token_embd\\.weight$|^blk\\.[0-3]\\.)";
}

bool common_memory_pressure_parse_psi(
        const std::string & text,
        common_memory_pressure_sample & sample,
        std::string & err) {
    sample = {};
    err.clear();

    std::istringstream input(text);
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }

        if (line.rfind("some ", 0) == 0) {
            if (!common_memory_pressure_parse_line(line, "some", sample.some, err)) {
                return false;
            }
            sample.has_some = true;
        } else if (line.rfind("full ", 0) == 0) {
            if (!common_memory_pressure_parse_line(line, "full", sample.full, err)) {
                return false;
            }
            sample.has_full = true;
        }
    }

    if (!sample.has_some) {
        err = "missing some line";
        return false;
    }
    return true;
}

bool common_memory_pressure_read_psi(
        const std::string & path,
        common_memory_pressure_sample & sample,
        std::string & err) {
    std::ifstream file(path);
    if (!file) {
        err = "failed to open PSI file: " + path;
        return false;
    }

    std::ostringstream ss;
    ss << file.rdbuf();
    return common_memory_pressure_parse_psi(ss.str(), sample, err);
}

common_memory_pressure_decision common_memory_pressure_policy::decide(
        const common_memory_pressure_config & config,
        const common_memory_pressure_sample & sample,
        const common_memory_pressure_bytes & bytes,
        bool busy,
        int64_t now_ms) {
    common_memory_pressure_decision decision;

    if (!config.enabled) {
        decision.action = COMMON_MEMORY_PRESSURE_ACTION_DISABLED;
        decision.state = COMMON_MEMORY_PRESSURE_STATE_DISABLED;
        decision.reason = "disabled";
        return decision;
    }

    if (config.policy == COMMON_MEMORY_PRESSURE_POLICY_EXTERNAL_HINT && !config.protected_app_active) {
        decision.action = COMMON_MEMORY_PRESSURE_ACTION_OBSERVE_ONLY;
        decision.state = COMMON_MEMORY_PRESSURE_STATE_OBSERVING;
        decision.reason = "external_hint_inactive";
        last_sample = sample;
        has_last_sample = true;
        return decision;
    }

    bool total_delta_trigger = false;
    if (has_last_sample && sample.has_some && last_sample.has_some &&
            sample.some.total > last_sample.some.total) {
        total_delta_trigger = (sample.some.total - last_sample.some.total) >= config.some_total_delta_us_thold;
    }

    common_memory_pressure_level level = COMMON_MEMORY_PRESSURE_LEVEL_LOW;
    std::string reason = "low_pressure";

    if (sample.has_full && sample.full.avg10 > config.full_avg10_thold) {
        level = COMMON_MEMORY_PRESSURE_LEVEL_CRITICAL;
        reason = "full_avg10";
    } else if (sample.has_some && sample.some.avg10 >= std::max(5.0, config.some_avg10_thold * 5.0)) {
        level = COMMON_MEMORY_PRESSURE_LEVEL_HIGH;
        reason = "some_avg10_high";
    } else if (sample.has_some && sample.some.avg60 >= config.some_avg60_thold) {
        level = COMMON_MEMORY_PRESSURE_LEVEL_HIGH;
        reason = "some_avg60";
    } else if (sample.has_some && sample.some.avg10 >= config.some_avg10_thold) {
        level = COMMON_MEMORY_PRESSURE_LEVEL_MEDIUM;
        reason = "some_avg10";
    } else if (total_delta_trigger) {
        level = COMMON_MEMORY_PRESSURE_LEVEL_MEDIUM;
        reason = "some_total_delta";
    }

    decision.level = level;
    decision.reason = reason;
    last_sample = sample;
    has_last_sample = true;

    if (level == COMMON_MEMORY_PRESSURE_LEVEL_LOW) {
        decision.action = COMMON_MEMORY_PRESSURE_ACTION_OBSERVE_ONLY;
        decision.state = COMMON_MEMORY_PRESSURE_STATE_OBSERVING;
        return decision;
    }

    if (bytes.total() == 0 || config.max_fraction <= 0.0) {
        decision.action = COMMON_MEMORY_PRESSURE_ACTION_NO_ACTION;
        decision.state = COMMON_MEMORY_PRESSURE_STATE_OBSERVING;
        decision.reason = "no_unloadable_weights";
        return decision;
    }

    const size_t max_unload = (size_t) ((long double) bytes.total() * config.max_fraction);
    if (bytes.unloaded >= max_unload) {
        decision.action = COMMON_MEMORY_PRESSURE_ACTION_SATURATED;
        decision.state = COMMON_MEMORY_PRESSURE_STATE_SATURATED;
        decision.reason = "max_fraction";
        return decision;
    }

    if (last_unload_ms >= 0 && now_ms - last_unload_ms < config.cooldown_ms) {
        decision.action = COMMON_MEMORY_PRESSURE_ACTION_OBSERVE_ONLY;
        decision.state = COMMON_MEMORY_PRESSURE_STATE_COOLDOWN;
        decision.reason = "cooldown";
        return decision;
    }

    size_t target = config.step_bytes;
    if (level == COMMON_MEMORY_PRESSURE_LEVEL_CRITICAL) {
        target *= 2;
    }
    target = std::min(target, max_unload - bytes.unloaded);

    decision.target_bytes = target;
    if (busy) {
        decision.action = COMMON_MEMORY_PRESSURE_ACTION_DEFER_BUSY;
        decision.state = COMMON_MEMORY_PRESSURE_STATE_RESTORE_PENDING;
        return decision;
    }

    decision.action = COMMON_MEMORY_PRESSURE_ACTION_UNLOAD;
    decision.state = COMMON_MEMORY_PRESSURE_STATE_RELIEVING;
    return decision;
}

void common_memory_pressure_policy::record_unload(int64_t now_ms) {
    last_unload_ms = now_ms;
}

void common_memory_pressure_policy::reset() {
    has_last_sample = false;
    last_sample = {};
    last_unload_ms = -1;
}

bool common_memory_pressure_select_weights(
        const std::vector<common_memory_pressure_weight> & weights,
        const common_memory_pressure_config & config,
        size_t target_bytes,
        common_memory_pressure_selection & selection,
        std::string & err) {
    selection = {};
    selection.target_bytes = target_bytes;
    err.clear();

    std::regex keep;
    const bool has_keep = !config.keep_regex.empty();
    if (has_keep) {
        try {
            keep = std::regex(config.keep_regex);
        } catch (const std::regex_error & e) {
            err = std::string("invalid keep regex: ") + e.what();
            return false;
        }
    }

    for (const auto & weight : weights) {
        if (weight.loaded) {
            selection.resident_bytes += weight.nbytes;
        } else {
            selection.unloaded_bytes += weight.nbytes;
        }
    }

    const size_t total = selection.resident_bytes + selection.unloaded_bytes;
    selection.max_unload_bytes = config.max_fraction <= 0.0 ? 0 :
        (size_t) ((long double) total * config.max_fraction);

    if (selection.unloaded_bytes >= selection.max_unload_bytes || target_bytes == 0) {
        selection.saturated = true;
        return true;
    }

    const size_t remaining_fraction_bytes = selection.max_unload_bytes - selection.unloaded_bytes;
    const size_t capped_target = std::min(target_bytes, remaining_fraction_bytes);

    std::vector<common_memory_pressure_weight> candidates;
    for (const auto & weight : weights) {
        if (!weight.loaded || weight.nbytes == 0) {
            selection.skipped_unloaded++;
            continue;
        }
        if (has_keep && std::regex_search(weight.name, keep)) {
            selection.skipped_keep++;
            continue;
        }
        candidates.push_back(weight);
    }

    auto priority = [](const common_memory_pressure_weight & weight) -> long long {
        if (weight.layer == COMMON_MEMORY_PRESSURE_LAYER_OUTPUT) {
            return (long long) INT_MAX + 1ll;
        }
        if (weight.layer >= 0) {
            return weight.layer;
        }
        return -1;
    };

    std::sort(candidates.begin(), candidates.end(), [&](const auto & a, const auto & b) {
        const long long pa = priority(a);
        const long long pb = priority(b);
        if (pa != pb) {
            return pa > pb;
        }
        if (a.nbytes != b.nbytes) {
            return a.nbytes > b.nbytes;
        }
        return a.name < b.name;
    });

    for (const auto & weight : candidates) {
        if (selection.selected_bytes >= capped_target) {
            break;
        }
        selection.selected.push_back(weight);
        selection.selected_bytes += weight.nbytes;
    }

    selection.saturated = selection.selected.empty() && selection.unloaded_bytes >= selection.max_unload_bytes;
    return true;
}

bool common_memory_pressure_collect_weights(
        const llama_model * model,
        std::vector<common_memory_pressure_weight> & weights,
        common_memory_pressure_bytes & bytes,
        std::string & err) {
    weights.clear();
    bytes = {};
    err.clear();

    if (!model) {
        err = "model is null";
        return false;
    }

    const int32_t n_weights = llama_model_weight_count(model);
    if (n_weights < 0) {
        err = "failed to count model weights";
        return false;
    }

    weights.reserve((size_t) n_weights);
    for (int32_t i = 0; i < n_weights; ++i) {
        const int32_t len = llama_model_weight_name_by_index(model, i, nullptr, 0);
        if (len < 0) {
            continue;
        }
        std::vector<char> name((size_t) len + 1);
        llama_model_weight_name_by_index(model, i, name.data(), name.size());

        size_t nbytes = 0;
        int32_t layer = -1;
        bool loaded = false;
        if (!llama_model_weight_info_by_index(model, i, &nbytes, &layer, &loaded)) {
            continue;
        }

        if (loaded) {
            bytes.resident += nbytes;
        } else {
            bytes.unloaded += nbytes;
        }
        weights.push_back({name.data(), nbytes, layer, loaded});
    }

    return true;
}

static bool common_memory_pressure_read_mem_available_kib(uint64_t & out) {
    std::ifstream file("/proc/meminfo");
    if (!file) {
        return false;
    }
    std::string key;
    uint64_t value = 0;
    std::string unit;
    while (file >> key >> value >> unit) {
        if (key == "MemAvailable:") {
            out = value;
            return true;
        }
    }
    return false;
}

struct common_memory_pressure_manager::impl {
    common_memory_pressure_config config;
    llama_model * model = nullptr;
    common_memory_pressure_policy policy;
    int64_t last_tick_ms = -1;
    bool disabled_by_error = false;
    size_t pending_target_bytes = 0;
    common_memory_pressure_level pending_level = COMMON_MEMORY_PRESSURE_LEVEL_LOW;
    std::string pending_reason;
    std::ofstream log;

    impl(const common_memory_pressure_config & config, llama_model * model) :
        config(config), model(model) {
        if (this->config.keep_regex.empty()) {
            this->config.keep_regex = common_memory_pressure_default_keep_regex();
        }
        const char * env_keep = std::getenv("LLAMA_WEIGHT_UNLOAD_KEEP_REGEX");
        if (env_keep && env_keep[0] != '\0') {
            this->config.keep_regex = "(" + this->config.keep_regex + ")|(" + std::string(env_keep) + ")";
        }
        if (!this->config.log_path.empty()) {
            log.open(this->config.log_path, std::ios::out | std::ios::trunc);
        }
    }

    void write_json(const std::string & body) {
        if (!log) {
            return;
        }
        log << "{" << "\"ts_ms\":" << common_memory_pressure_now_ms();
        if (!body.empty()) {
            log << "," << body;
        }
        log << "}\n";
        log.flush();
    }
};

common_memory_pressure_manager::common_memory_pressure_manager(
        const common_memory_pressure_config & config,
        llama_model * model) :
    pimpl(new impl(config, model)) {
}

common_memory_pressure_manager::~common_memory_pressure_manager() = default;

bool common_memory_pressure_manager::enabled() const {
    return pimpl && pimpl->config.enabled && !pimpl->disabled_by_error;
}

bool common_memory_pressure_manager::tick(bool busy, std::string * err) {
    if (!pimpl || !pimpl->config.enabled || pimpl->disabled_by_error || !pimpl->model) {
        return true;
    }

    const int64_t now = common_memory_pressure_now_ms();
    if (pimpl->last_tick_ms >= 0 && now - pimpl->last_tick_ms < pimpl->config.interval_ms) {
        return true;
    }
    pimpl->last_tick_ms = now;

    common_memory_pressure_sample sample;
    std::string local_err;
    if (!common_memory_pressure_read_psi(pimpl->config.path, sample, local_err)) {
        pimpl->write_json("\"event\":\"psi_error\",\"path\":\"" +
                common_memory_pressure_json_escape(pimpl->config.path) +
                "\",\"error\":\"" + common_memory_pressure_json_escape(local_err) + "\"");
        pimpl->disabled_by_error = true;
        if (err) {
            *err = local_err;
        }
        return false;
    }

    std::vector<common_memory_pressure_weight> weights;
    common_memory_pressure_bytes bytes;
    if (!common_memory_pressure_collect_weights(pimpl->model, weights, bytes, local_err)) {
        if (err) {
            *err = local_err;
        }
        return false;
    }

    uint64_t mem_available_kib = 0;
    const bool has_mem_available = common_memory_pressure_read_mem_available_kib(mem_available_kib);

    {
        std::ostringstream body;
        body << "\"event\":\"psi_sample\""
             << ",\"some_avg10\":" << sample.some.avg10
             << ",\"some_avg60\":" << sample.some.avg60
             << ",\"full_avg10\":" << sample.full.avg10
             << ",\"some_total\":" << sample.some.total
             << ",\"resident_mib\":" << std::fixed << std::setprecision(2) << bytes.resident / 1024.0 / 1024.0
             << ",\"unloaded_mib\":" << std::fixed << std::setprecision(2) << bytes.unloaded / 1024.0 / 1024.0;
        if (has_mem_available) {
            body << ",\"mem_available_kib\":" << mem_available_kib;
        }
        pimpl->write_json(body.str());
    }

    auto decision = pimpl->policy.decide(pimpl->config, sample, bytes, busy, now);
    if (decision.action != COMMON_MEMORY_PRESSURE_ACTION_OBSERVE_ONLY &&
            decision.action != COMMON_MEMORY_PRESSURE_ACTION_NO_ACTION) {
        std::ostringstream body;
        body << "\"event\":\"unload_decision\""
             << ",\"action\":\"" << common_memory_pressure_action_name(decision.action) << "\""
             << ",\"state\":\"" << common_memory_pressure_state_name(decision.state) << "\""
             << ",\"level\":\"" << common_memory_pressure_level_name(decision.level) << "\""
             << ",\"target_bytes\":" << decision.target_bytes
             << ",\"reason\":\"" << common_memory_pressure_json_escape(decision.reason) << "\""
             << ",\"busy\":" << (busy ? "true" : "false")
             << ",\"dry_run\":" << (pimpl->config.dry_run ? "true" : "false");
        pimpl->write_json(body.str());
    }

    if (decision.action == COMMON_MEMORY_PRESSURE_ACTION_DEFER_BUSY) {
        pimpl->pending_target_bytes = std::max(pimpl->pending_target_bytes, decision.target_bytes);
        pimpl->pending_level = decision.level;
        pimpl->pending_reason = decision.reason;
        return true;
    }

    if (!busy && pimpl->pending_target_bytes > 0 &&
            decision.action != COMMON_MEMORY_PRESSURE_ACTION_UNLOAD &&
            decision.action != COMMON_MEMORY_PRESSURE_ACTION_SATURATED &&
            decision.action != COMMON_MEMORY_PRESSURE_ACTION_DISABLED) {
        decision.action = COMMON_MEMORY_PRESSURE_ACTION_UNLOAD;
        decision.state = COMMON_MEMORY_PRESSURE_STATE_RELIEVING;
        decision.level = pimpl->pending_level;
        decision.target_bytes = pimpl->pending_target_bytes;
        decision.reason = "pending_busy:" + pimpl->pending_reason;

        std::ostringstream body;
        body << "\"event\":\"unload_decision\""
             << ",\"action\":\"" << common_memory_pressure_action_name(decision.action) << "\""
             << ",\"state\":\"" << common_memory_pressure_state_name(decision.state) << "\""
             << ",\"level\":\"" << common_memory_pressure_level_name(decision.level) << "\""
             << ",\"target_bytes\":" << decision.target_bytes
             << ",\"reason\":\"" << common_memory_pressure_json_escape(decision.reason) << "\""
             << ",\"busy\":false"
             << ",\"dry_run\":" << (pimpl->config.dry_run ? "true" : "false");
        pimpl->write_json(body.str());
    }

    if (decision.action != COMMON_MEMORY_PRESSURE_ACTION_UNLOAD) {
        return true;
    }

    common_memory_pressure_selection selection;
    if (!common_memory_pressure_select_weights(weights, pimpl->config, decision.target_bytes, selection, local_err)) {
        pimpl->write_json("\"event\":\"unload_error\",\"error\":\"" +
                common_memory_pressure_json_escape(local_err) + "\"");
        if (err) {
            *err = local_err;
        }
        return false;
    }

    size_t bytes_freed = 0;
    size_t tensors_unloaded = 0;
    for (const auto & weight : selection.selected) {
        size_t freed = 0;
        bool ok = true;
        if (!pimpl->config.dry_run) {
            ok = llama_model_unload_tensor(pimpl->model, weight.name.c_str(), &freed);
        } else {
            freed = weight.nbytes;
        }

        if (ok) {
            bytes_freed += freed;
            tensors_unloaded++;
            std::ostringstream body;
            body << "\"event\":\"tensor_unloaded\""
                 << ",\"name\":\"" << common_memory_pressure_json_escape(weight.name) << "\""
                 << ",\"layer\":" << weight.layer
                 << ",\"bytes\":" << freed
                 << ",\"dry_run\":" << (pimpl->config.dry_run ? "true" : "false");
            pimpl->write_json(body.str());
        } else {
            pimpl->write_json("\"event\":\"unload_error\",\"name\":\"" +
                    common_memory_pressure_json_escape(weight.name) + "\"");
        }
    }

    if (!selection.selected.empty()) {
        pimpl->policy.record_unload(now);
        pimpl->pending_target_bytes = 0;
        pimpl->pending_reason.clear();
    }

    pimpl->write_json("\"event\":\"unload_summary\",\"bytes_freed\":" +
            std::to_string(bytes_freed) +
            ",\"tensors\":" + std::to_string(tensors_unloaded) +
            ",\"resident_mib\":" + std::to_string(selection.resident_bytes / 1024.0 / 1024.0) +
            ",\"unloaded_mib\":" + std::to_string((selection.unloaded_bytes + bytes_freed) / 1024.0 / 1024.0) +
            ",\"dry_run\":" + (pimpl->config.dry_run ? std::string("true") : std::string("false")));

    return true;
}
