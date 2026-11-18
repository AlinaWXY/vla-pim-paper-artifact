#pragma once
#include "ramulator/controller/impl/pim_model.h"

namespace PimVla {
// Streaming JSON-to-Operation conversion. This reads the same immutable
// program bytes as the YAML path without constructing/copying a generic DOM.
std::map<std::string, ProgramStage> read_program_json(const std::string& path, int* layer_stride_bytes_out=nullptr,
                                                   std::string* execution_model_out=nullptr);
}
