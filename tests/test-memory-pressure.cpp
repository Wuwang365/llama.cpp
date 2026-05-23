#include "memory-pressure.h"

#include <cassert>
#include <cmath>
#include <string>
#include <vector>

static bool near(double a, double b) {
    return std::fabs(a - b) < 0.0001;
}

static void test_parse_normal_psi() {
    const std::string text =
        "some avg10=2.30 avg60=0.80 avg300=0.10 total=123456\n"
        "full avg10=0.01 avg60=0.00 avg300=0.00 total=123\n";

    common_memory_pressure_sample sample;
    std::string err;
    assert(common_memory_pressure_parse_psi(text, sample, err));
    assert(sample.has_some);
    assert(sample.has_full);
    assert(near(sample.some.avg10, 2.30));
    assert(near(sample.some.avg60, 0.80));
    assert(near(sample.some.avg300, 0.10));
    assert(sample.some.total == 123456);
    assert(near(sample.full.avg10, 0.01));
    assert(sample.full.total == 123);
}

static void test_parse_missing_full_is_allowed() {
    const std::string text = "some total=42 avg300=0.00 avg60=0.25 avg10=1.50\n";

    common_memory_pressure_sample sample;
    std::string err;
    assert(common_memory_pressure_parse_psi(text, sample, err));
    assert(sample.has_some);
    assert(!sample.has_full);
    assert(near(sample.some.avg10, 1.50));
    assert(near(sample.some.avg60, 0.25));
    assert(sample.some.total == 42);
}

static void test_parse_invalid_number_fails() {
    const std::string text = "some avg10=oops avg60=0.00 avg300=0.00 total=1\n";

    common_memory_pressure_sample sample;
    std::string err;
    assert(!common_memory_pressure_parse_psi(text, sample, err));
    assert(err.find("avg10") != std::string::npos);
}

static void test_policy_thresholds_cooldown_and_saturation() {
    common_memory_pressure_config cfg;
    cfg.enabled = true;
    cfg.step_bytes = 128ull * 1024ull * 1024ull;
    cfg.cooldown_ms = 1000;
    cfg.some_avg10_thold = 1.0;
    cfg.some_avg60_thold = 2.0;
    cfg.full_avg10_thold = 0.0;
    cfg.max_fraction = 0.80;

    common_memory_pressure_policy policy;

    common_memory_pressure_sample sample;
    sample.has_some = true;
    sample.some.avg10 = 1.25;
    sample.some.avg60 = 0.0;
    sample.some.total = 100;

    common_memory_pressure_bytes bytes;
    bytes.resident = 900ull * 1024ull * 1024ull;
    bytes.unloaded = 0;

    auto decision = policy.decide(cfg, sample, bytes, false, 1000);
    assert(decision.action == COMMON_MEMORY_PRESSURE_ACTION_UNLOAD);
    assert(decision.level == COMMON_MEMORY_PRESSURE_LEVEL_MEDIUM);
    assert(decision.target_bytes == cfg.step_bytes);

    policy.record_unload(1000);
    sample.some.avg10 = 3.0;
    sample.some.total = 200;
    decision = policy.decide(cfg, sample, bytes, false, 1500);
    assert(decision.action == COMMON_MEMORY_PRESSURE_ACTION_OBSERVE_ONLY);
    assert(decision.state == COMMON_MEMORY_PRESSURE_STATE_COOLDOWN);

    decision = policy.decide(cfg, sample, bytes, true, 2500);
    assert(decision.action == COMMON_MEMORY_PRESSURE_ACTION_DEFER_BUSY);

    bytes.resident = 100ull * 1024ull * 1024ull;
    bytes.unloaded = 800ull * 1024ull * 1024ull;
    decision = policy.decide(cfg, sample, bytes, false, 3000);
    assert(decision.action == COMMON_MEMORY_PRESSURE_ACTION_SATURATED);
    assert(decision.state == COMMON_MEMORY_PRESSURE_STATE_SATURATED);
}

static void test_policy_external_hint_observes_when_inactive() {
    common_memory_pressure_config cfg;
    cfg.enabled = true;
    cfg.policy = COMMON_MEMORY_PRESSURE_POLICY_EXTERNAL_HINT;
    cfg.protected_app_active = false;
    cfg.some_avg10_thold = 1.0;

    common_memory_pressure_sample sample;
    sample.has_some = true;
    sample.some.avg10 = 10.0;

    common_memory_pressure_bytes bytes;
    bytes.resident = 1024;

    common_memory_pressure_policy policy;
    const auto decision = policy.decide(cfg, sample, bytes, false, 1000);
    assert(decision.action == COMMON_MEMORY_PRESSURE_ACTION_OBSERVE_ONLY);
    assert(decision.reason == "external_hint_inactive");
}

static void test_select_weights_default_keep_and_order() {
    common_memory_pressure_config cfg;
    cfg.step_bytes = 220;
    cfg.max_fraction = 1.0;
    cfg.keep_regex = common_memory_pressure_default_keep_regex();

    std::vector<common_memory_pressure_weight> weights = {
        {"token_embd.weight", 100, -1, true},
        {"blk.0.attn_q.weight", 100, 0, true},
        {"blk.4.attn_q.weight", 100, 4, true},
        {"blk.7.attn_q.weight", 100, 7, true},
        {"output.weight", 80, COMMON_MEMORY_PRESSURE_LAYER_OUTPUT, true},
        {"blk.6.ffn_down.weight", 100, 6, false},
    };

    common_memory_pressure_selection selection;
    std::string err;
    assert(common_memory_pressure_select_weights(weights, cfg, cfg.step_bytes, selection, err));
    assert(selection.selected.size() == 3);
    assert(selection.selected[0].name == "output.weight");
    assert(selection.selected[1].name == "blk.7.attn_q.weight");
    assert(selection.selected[2].name == "blk.4.attn_q.weight");
    assert(selection.selected_bytes == 280);
    assert(selection.skipped_keep == 2);
    assert(selection.skipped_unloaded == 1);
}

static void test_select_weights_respects_max_fraction() {
    common_memory_pressure_config cfg;
    cfg.step_bytes = 500;
    cfg.max_fraction = 0.50;

    std::vector<common_memory_pressure_weight> weights = {
        {"blk.4.a", 100, 4, false},
        {"blk.5.a", 100, 5, true},
        {"blk.6.a", 100, 6, true},
        {"blk.7.a", 100, 7, true},
    };

    common_memory_pressure_selection selection;
    std::string err;
    assert(common_memory_pressure_select_weights(weights, cfg, cfg.step_bytes, selection, err));
    assert(selection.selected.size() == 1);
    assert(selection.selected[0].name == "blk.7.a");
    assert(selection.selected_bytes == 100);
    assert(selection.saturated == false);
}

int main() {
    test_parse_normal_psi();
    test_parse_missing_full_is_allowed();
    test_parse_invalid_number_fails();
    test_policy_thresholds_cooldown_and_saturation();
    test_policy_external_hint_observes_when_inactive();
    test_select_weights_default_keep_and_order();
    test_select_weights_respects_max_fraction();
    return 0;
}
