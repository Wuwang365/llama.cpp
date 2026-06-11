#include "ggml-vulkan-managed-reclaim.h"

#include <cstdio>
#include <cstdlib>

static void require_true(bool cond, const char * msg) {
    if (!cond) {
        fprintf(stderr, "%s\n", msg);
        exit(EXIT_FAILURE);
    }
}

int main() {
    ggml_vk_managed_reclaim_class cls;

    require_true(ggml_vk_managed_reclaim_classify("token_embd.weight", &cls), "token embedding should classify");
    require_true(cls.permanent, "token embedding should be permanent");
    require_true(!cls.reclaimable, "token embedding should not be reclaimable");

    require_true(ggml_vk_managed_reclaim_classify("blk.4.ffn_down.weight", &cls), "ffn weight should classify");
    require_true(cls.reclaimable, "ffn weight should be reclaimable");
    require_true(cls.layer == 4, "ffn weight layer should be parsed");
    require_true(cls.component == GGML_VK_MANAGED_RECLAIM_MLP, "ffn weight should be MLP component");

    require_true(ggml_vk_managed_reclaim_classify("blk.4.attn_q.weight", &cls), "attention weight should classify");
    require_true(cls.reclaimable, "attention weight should be reclaimable");
    require_true(cls.layer == 4, "attention weight layer should be parsed");
    require_true(cls.component == GGML_VK_MANAGED_RECLAIM_ATTN, "attention weight should be attention component");

    ggml_vk_managed_reclaim_class output_cls;
    ggml_vk_managed_reclaim_class mlp7_cls;
    ggml_vk_managed_reclaim_class attn7_cls;
    ggml_vk_managed_reclaim_class mlp6_cls;

    require_true(ggml_vk_managed_reclaim_classify("output_norm.weight", &output_cls), "output norm should classify");
    require_true(ggml_vk_managed_reclaim_classify("blk.7.ffn_up.weight", &mlp7_cls), "layer 7 MLP should classify");
    require_true(ggml_vk_managed_reclaim_classify("blk.7.attn_output.weight", &attn7_cls), "layer 7 attention should classify");
    require_true(ggml_vk_managed_reclaim_classify("blk.6.ffn_up.weight", &mlp6_cls), "layer 6 MLP should classify");

    require_true(ggml_vk_managed_reclaim_precedes(output_cls, mlp7_cls), "output should be reclaimed before layer weights");
    require_true(ggml_vk_managed_reclaim_precedes(mlp7_cls, attn7_cls), "MLP should be reclaimed before attention within a layer");
    require_true(ggml_vk_managed_reclaim_precedes(attn7_cls, mlp6_cls), "deeper layer should be reclaimed before shallower layer");

    return EXIT_SUCCESS;
}
