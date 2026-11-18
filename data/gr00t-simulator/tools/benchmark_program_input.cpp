#include "ramulator/frontend/impl/json_input.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#ifdef __linux__
#include <sys/resource.h>
#endif

using namespace PimVla;
struct SemanticDigest {
  uint64_t hash=14695981039346656037ULL,count=0;
  void bytes(const std::string& s) {for(unsigned char c:s){hash^=c;hash*=1099511628211ULL;}}
  void add(const std::string& s) {bytes(std::to_string(s.size()));bytes(":");bytes(s);}
  void add(int64_t n) {add(std::to_string(n));}
  void operations(const std::vector<Operation>& ops) {
    add(ops.size());
    for(const auto& o:ops) {
      ++count;
      add(o.name);add(o.epoch);add(o.id);add(o.slot);add(o.result_slot);add(o.rows);add(o.pairs);
      add(o.dest_slot);add(o.source_bank);add(o.tensor_fragment);add(o.output_columns);add(o.output_column);add(o.reduction_word);
      add(o.fragment);add(o.output_tile);add(o.kv_token);add(o.lanes);add(o.elements);add(o.bytes);add(o.token);
      add(o.release_result?1:0);add(o.dest);add(o.source);
      add(o.addr.size());for(int n:o.addr)add(n);
      add(o.deps.size());for(int n:o.deps)add(n);
      add(o.issue_deps.size());for(int n:o.issue_deps)add(n);
      add(o.source_regions.size());for(const auto& r:o.source_regions){add(r.bank);add(r.byte_offset);add(r.bytes);}
      operations(o.children);
    }
  }
};
int main(int argc,char** argv) {
  try {
    require(argc==2,"usage: benchmark_program_input PROGRAM.json");
    int layer_stride=0;
    auto start=std::chrono::steady_clock::now();auto program=read_program_json(argv[1],&layer_stride);
    auto parsed=std::chrono::steady_clock::now();
    SemanticDigest digest;digest.add(program.size());
    for(const auto& [name,stage]:program){digest.add(name);for(const auto& channel:stage.channels)digest.operations(channel);}
    long rss=0;
#ifdef __linux__
    struct rusage usage{};getrusage(RUSAGE_SELF,&usage);rss=usage.ru_maxrss;
#endif
    std::cout<<"{\"stages\":"<<program.size()<<",\"operations\":"<<digest.count<<",\"layer_stride_bytes\":"<<layer_stride
      <<",\"semantic_digest64\":\""<<std::hex<<std::setw(16)<<std::setfill('0')<<digest.hash<<std::dec
      <<"\",\"parse_ms\":"<<std::chrono::duration<double,std::milli>(parsed-start).count()
      <<",\"peak_rss_KiB\":"<<rss<<"}\n";
    return 0;
  } catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
