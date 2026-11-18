#include "ramulator/frontend/impl/json_input.h"
#include "../../../../../third_party/nlohmann_json/json.hpp"
#include <climits>
#include <fstream>
#include <limits>
#include <memory>
#include <utility>

namespace PimVla {
namespace {
using Json = nlohmann::json;
enum class Kind {Root, Contract, Stages, Stage, Channels, Channel, Op, Children, Address, Deps, IssueDeps, Regions, Region, SkipObject, SkipArray};
struct Frame {
  Kind kind;
  std::string key;
  std::unique_ptr<ProgramStage> stage;
  Operation operation;
  ResourceRegion region{};
  std::vector<Operation> operations;
  std::vector<int> integers;
  std::vector<ResourceRegion> regions;
  int channel_count=0;
  bool explicit_id=false;
  explicit Frame(Kind k):kind(k) {}
};

class ProgramSax final : public nlohmann::json_sax<Json> {
 public:
  std::vector<Frame> stack;
  std::map<std::string, ProgramStage> programs;
  int schema_version=0;
  int layer_stride_bytes=1048576;
  std::string execution_model;
  bool root_seen=false, stages_seen=false;

  bool null() override {return true;}
  bool boolean(bool value) override {
    if(!stack.empty() && stack.back().kind==Kind::Op && stack.back().key=="release_result")
      stack.back().operation.release_result=value;
    return true;
  }
  bool number_integer(number_integer_t value) override {return number(value);}
  bool number_unsigned(number_unsigned_t value) override {
    if(value>static_cast<uint64_t>(INT64_MAX)) {
      require(!integer_field(),"integer overflow in PIM program");return true;
    }
    return number(static_cast<int64_t>(value));
  }
  bool number_float(number_float_t, const string_t&) override {
    require(!integer_field(),"PIM integer field cannot contain a floating-point value");return true;
  }
  bool string(string_t& value) override {
    if(stack.empty())return false;
    auto& f=stack.back();
    if(f.kind==Kind::Root && f.key=="execution_model")execution_model=value;
    if(f.kind==Kind::Stage && f.key=="name")f.stage->name=std::move(value);
    if(f.kind==Kind::Op) {
      if(f.key=="op")f.operation.name=std::move(value);
      else if(f.key=="epoch")f.operation.epoch=std::move(value);
      else if(f.key=="dest")f.operation.dest=std::move(value);
      else if(f.key=="source")f.operation.source=std::move(value);
    }
    return true;
  }
  bool binary(binary_t&) override {return false;}
  bool key(string_t& value) override {
    require(!stack.empty(),"JSON key outside object");stack.back().key=std::move(value);return true;
  }
  bool start_object(std::size_t) override {
    Kind kind=Kind::SkipObject;
    if(stack.empty()) {require(!root_seen,"multiple JSON roots");kind=Kind::Root;root_seen=true;}
    else switch(stack.back().kind) {
      case Kind::Root:if(stack.back().key=="tensor_contract")kind=Kind::Contract;break;
      case Kind::Stages:kind=Kind::Stage;break;
      case Kind::Channel:case Kind::Children:kind=Kind::Op;break;
      case Kind::Regions:kind=Kind::Region;break;
      default:break;
    }
    stack.emplace_back(kind);
    if(kind==Kind::Stage)stack.back().stage=std::make_unique<ProgramStage>();
    return true;
  }
  bool end_object() override {
    require(!stack.empty(),"unbalanced program JSON object");
    Frame f=std::move(stack.back());stack.pop_back();
    if(f.kind==Kind::Stage) {
      require(!f.stage->name.empty(),"program stage name is missing");
      require(f.channel_count==16,"program stage has no 16-channel collection");
      std::string name=f.stage->name;
      require(programs.emplace(name,std::move(*f.stage)).second,"duplicate PIM stage "+name);
    } else if(f.kind==Kind::Op) {
      require(!stack.empty()&&(stack.back().kind==Kind::Channel||stack.back().kind==Kind::Children),"operation has no sequence");
      auto& list=stack.back().operations;
      if(!f.explicit_id)f.operation.id=static_cast<int>(list.size());
      validate(f.operation,list.size());list.push_back(std::move(f.operation));
    } else if(f.kind==Kind::Region) {
      require(!stack.empty()&&stack.back().kind==Kind::Regions,"region has no sequence");
      stack.back().regions.push_back(f.region);
    }
    return true;
  }
  bool start_array(std::size_t) override {
    Kind kind=Kind::SkipArray;
    if(!stack.empty()) {
      auto& parent=stack.back();
      if(parent.kind==Kind::Root && parent.key=="stages") {kind=Kind::Stages;stages_seen=true;}
      else if(parent.kind==Kind::Stage && parent.key=="channels")kind=Kind::Channels;
      else if(parent.kind==Kind::Channels)kind=Kind::Channel;
      else if(parent.kind==Kind::Op) {
        if(parent.key=="addr" || parent.key=="spill_addr")kind=Kind::Address;
        else if(parent.key=="deps")kind=Kind::Deps;
        else if(parent.key=="issue_deps")kind=Kind::IssueDeps;
        else if(parent.key=="children")kind=Kind::Children;
        else if(parent.key=="source_regions")kind=Kind::Regions;
      }
    }
    stack.emplace_back(kind);return true;
  }
  bool end_array() override {
    require(!stack.empty(),"unbalanced program JSON array");
    Frame f=std::move(stack.back());stack.pop_back();
    if(f.kind==Kind::Channel) {
      require(stack.size()>=2&&stack.back().kind==Kind::Channels,"channel outside channel collection");
      int index=stack.back().channel_count++;
      require(index<16,"program has more than 16 channels");
      stack[stack.size()-2].stage->channels[index]=std::move(f.operations);
    } else if(f.kind==Kind::Channels) {
      require(f.channel_count==16,"program stage must cover exactly 16 channels");
      stack.back().channel_count=f.channel_count;
    } else if(f.kind==Kind::Address) {
      auto& op=stack.back().operation;
      (stack.back().key=="spill_addr"?op.spill_addr:op.addr)=std::move(f.integers);
    }
    else if(f.kind==Kind::Deps)stack.back().operation.deps=std::move(f.integers);
    else if(f.kind==Kind::IssueDeps)stack.back().operation.issue_deps=std::move(f.integers);
    else if(f.kind==Kind::Children)stack.back().operation.children=std::move(f.operations);
    else if(f.kind==Kind::Regions)stack.back().operation.source_regions=std::move(f.regions);
    return true;
  }
  bool parse_error(std::size_t position,const std::string&,const nlohmann::detail::exception& e) override {
    throw std::runtime_error("PIM JSON parse error at byte "+std::to_string(position)+": "+e.what());
  }
 private:
  static int* operation_integer(Frame& f) {
    auto& o=f.operation;const auto& k=f.key;
    if(k=="id")return &o.id;
    if(k=="slot")return &o.slot;
    if(k=="result_slot")return &o.result_slot;
    if(k=="rows")return &o.rows;
    if(k=="pairs")return &o.pairs;
    if(k=="dest_slot")return &o.dest_slot;
    if(k=="source_bank")return &o.source_bank;
    if(k=="tensor_fragment")return &o.tensor_fragment;
    if(k=="output_columns")return &o.output_columns;
    if(k=="output_column")return &o.output_column;
    if(k=="fragment")return &o.fragment;
    if(k=="output_tile")return &o.output_tile;
    if(k=="kv_token")return &o.kv_token;
    if(k=="reduction_word")return &o.reduction_word;
    if(k=="lanes")return &o.lanes;
    if(k=="weight_lane_offset")return &o.weight_lane_offset;
    if(k=="elements")return &o.elements;
    if(k=="bytes")return &o.bytes;
    if(k=="token")return &o.token;
    return nullptr;
  }
  bool integer_field() {
    if(stack.empty())return false;
    auto& f=stack.back();
    return f.kind==Kind::Address||f.kind==Kind::Deps||f.kind==Kind::IssueDeps||f.kind==Kind::Region||
        (f.kind==Kind::Root&&f.key=="schema_version")||
        (f.kind==Kind::Contract&&f.key=="layer_stride_bytes_per_bank")||(f.kind==Kind::Op&&operation_integer(f));
  }
  bool number(int64_t value) {
    if(!integer_field())return true;
    require(value>=INT_MIN&&value<=INT_MAX,"integer out of range in PIM program");
    auto& f=stack.back();int v=static_cast<int>(value);
    if(f.kind==Kind::Address||f.kind==Kind::Deps||f.kind==Kind::IssueDeps)f.integers.push_back(v);
    else if(f.kind==Kind::Root)schema_version=v;
    else if(f.kind==Kind::Contract) {require(v>0,"layer stride must be positive");layer_stride_bytes=v;}
    else if(f.kind==Kind::Op) {*operation_integer(f)=v;if(f.key=="id")f.explicit_id=true;}
    else if(f.kind==Kind::Region) {
      if(f.key=="bank")f.region.bank=v;
      else if(f.key=="byte_offset")f.region.byte_offset=v;
      else if(f.key=="bytes")f.region.bytes=v;
    }
    return true;
  }
  static void validate(const Operation& o,size_t index) {
    require(!o.name.empty(),"program operation has no name");
    require(o.id==static_cast<int>(index),"program IDs must be contiguous");
    require(o.addr.size()==6,"physical address must have 6 levels");
    require(o.addr[1]==1,"PIM may only access rank1");
    require(o.spill_addr.empty() || (o.spill_addr.size()==6 && o.spill_addr[0]==o.addr[0] &&
            o.spill_addr[1]==1 && o.spill_addr[2]>=0 && o.spill_addr[2]<4 &&
            o.spill_addr[3]>=0 && o.spill_addr[3]<4 &&
            o.spill_addr[4]>=0 && o.spill_addr[5]>=0 && o.spill_addr[5]<64),
            "invalid accumulator spill address");
    require(o.slot>=0&&o.slot<32,"SRF capacity exceeded");
    require(o.result_slot>=0&&o.result_slot<64,"output pool capacity exceeded");
    require(o.reduction_word>=16&&o.reduction_word<512,"reduction outside bank output pool");
    for(int d:o.deps)require(d>=0&&d<o.id,"dependency must point to previous request");
    for(int d:o.issue_deps)require(d>=0&&d<o.id,"issue dependency must point to previous request");
  }
};
}

std::map<std::string,ProgramStage> read_program_json(const std::string& path,int* layer_stride_bytes_out,std::string* execution_model_out) {
  std::ifstream stream(path,std::ios::binary);
  require(stream.good(),"cannot open PIM JSON program: "+path);
  ProgramSax sax;
  require(Json::sax_parse(stream,&sax,Json::input_format_t::json,true),"PIM JSON parsing failed");
  require(sax.schema_version==1,"unsupported PIM program version");
  require(sax.stages_seen&&!sax.programs.empty(),"PIM program has no stages");
  if(layer_stride_bytes_out)*layer_stride_bytes_out=sax.layer_stride_bytes;
  if(execution_model_out)*execution_model_out=sax.execution_model;
  return std::move(sax.programs);
}
}
