#pragma once

#include <climits>
#include <cstring>

enum ggml_vk_managed_reclaim_component {
    GGML_VK_MANAGED_RECLAIM_INVALID = 0,
    GGML_VK_MANAGED_RECLAIM_ATTN,
    GGML_VK_MANAGED_RECLAIM_MLP,
    GGML_VK_MANAGED_RECLAIM_OUTPUT,
};

struct ggml_vk_managed_reclaim_class {
    int layer = -1;
    ggml_vk_managed_reclaim_component component = GGML_VK_MANAGED_RECLAIM_INVALID;
    bool permanent = false;
    bool reclaimable = false;
};

static inline bool ggml_vk_managed_reclaim_parse_layer(const char * name, int * layer) {
    static constexpr const char * prefix = "blk.";
    if (std::strncmp(name, prefix, 4) != 0) {
        return false;
    }

    int value = 0;
    const char * p = name + 4;
    if (*p < '0' || *p > '9') {
        return false;
    }
    while (*p >= '0' && *p <= '9') {
        value = value * 10 + (*p - '0');
        ++p;
    }
    if (*p != '.') {
        return false;
    }

    *layer = value;
    return true;
}

static inline bool ggml_vk_managed_reclaim_classify(
        const char * name,
        ggml_vk_managed_reclaim_class * out) {
    if (name == nullptr || out == nullptr) {
        return false;
    }

    *out = ggml_vk_managed_reclaim_class{};

    if (std::strstr(name, "token_embd") != nullptr) {
        out->permanent = true;
        return true;
    }

    if (std::strncmp(name, "output", 6) == 0) {
        out->layer = INT_MAX;
        out->component = GGML_VK_MANAGED_RECLAIM_OUTPUT;
        out->reclaimable = true;
        return true;
    }

    int layer = -1;
    if (!ggml_vk_managed_reclaim_parse_layer(name, &layer)) {
        return false;
    }

    out->layer = layer;
    out->reclaimable = true;
    if (std::strstr(name, ".ffn_") != nullptr || std::strstr(name, ".mlp_") != nullptr) {
        out->component = GGML_VK_MANAGED_RECLAIM_MLP;
        return true;
    }
    if (std::strstr(name, ".attn_") != nullptr || std::strstr(name, ".attn") != nullptr) {
        out->component = GGML_VK_MANAGED_RECLAIM_ATTN;
        return true;
    }

    out->reclaimable = false;
    return false;
}

static inline int ggml_vk_managed_reclaim_component_rank(ggml_vk_managed_reclaim_component component) {
    switch (component) {
        case GGML_VK_MANAGED_RECLAIM_OUTPUT:
            return 3;
        case GGML_VK_MANAGED_RECLAIM_MLP:
            return 2;
        case GGML_VK_MANAGED_RECLAIM_ATTN:
            return 1;
        default:
            return 0;
    }
}

static inline bool ggml_vk_managed_reclaim_precedes(
        const ggml_vk_managed_reclaim_class & lhs,
        const ggml_vk_managed_reclaim_class & rhs) {
    if (lhs.layer != rhs.layer) {
        return lhs.layer > rhs.layer;
    }
    return ggml_vk_managed_reclaim_component_rank(lhs.component) >
        ggml_vk_managed_reclaim_component_rank(rhs.component);
}
