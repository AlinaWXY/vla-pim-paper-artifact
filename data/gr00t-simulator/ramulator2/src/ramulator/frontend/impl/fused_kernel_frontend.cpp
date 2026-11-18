#include "ramulator/controller/impl/pim_model.h"
#include "ramulator/frontend/impl/json_input.h"
#include "../../../../../third_party/nlohmann_json/json.hpp"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <set>

namespace PimVla {
namespace {
using Json=nlohmann::json;
Json read_json(const std::string& path) {
  std::ifstream in(path);require(in.good(),"cannot open JSON: "+path);
  return Json::parse(in);
}
struct CoarseEvent {std::string kind,name;Clock duration=0;int step=0,layer=0;};
struct CalibratedKernel {int ordinal=0;std::string name,kind;Clock duration=0,start=0,end=0;uint64_t reads=0,writes=0;};
enum class Phase {None,Upload,Launch,Compute,Join,Continue,Download,Transfer};
const char* phase_name(Phase p) {
  switch(p) {case Phase::Upload:return "upload";case Phase::Launch:return "launch";case Phase::Compute:return "compute";
    case Phase::Join:return "join";case Phase::Continue:return "continue";case Phase::Download:return "download";
    case Phase::Transfer:return "transfer";default:return "none";}
}
int descriptor_index(const std::vector<Operation>& ops) {
  int index=-1;
  for(size_t i=0;i<ops.size();++i) {
    const auto& op=ops[i];
    if(op.name=="MAC_RUN"||op.name=="ROPE_RUN") {
      require(index<0&&!op.children.empty(),"kernel window requires exactly one nonempty descriptor per compute stage");
      index=static_cast<int>(i);
      for(const auto& child:op.children) {
        require(child.children.empty(),"nested descriptor in kernel window");
        require(child.name!="RD"&&child.name!="WR"&&child.name!="RDRES"&&child.name!="WRGB"&&child.epoch.empty(),
                "kernel window descriptor contains external I/O; spill results to DRAM before export");
      }
    }
  }
  return index;
}
}

int run_fused_kernels(const std::string& config_file,const std::string& output) {
  auto cfg=Config::parse_config_file(config_file);
  const auto model=cfg["model"].as<std::string>("pi05");
  require(model=="pi05"||model=="gr00t_n16","unknown workload model");
  const auto rope=cfg["rope"].as<std::string>("gpu");
  require(model=="pi05" ? rope=="gpu"||rope=="local" : rope=="none","RoPE location differs from workload model");
  require(cfg["mode"].as<std::string>()=="concurrent"&&cfg["dataflow"].as<std::string>()=="fused"&&
          cfg["run_kind"].as<std::string>("combined")=="combined","fused_kernel_windows requires concurrent fused combined execution");
  require(!cfg["arbitration"]&&!cfg["refresh_allbank"],"kernel windows do not accept request arbitration or refresh overrides");
  auto spec=cfg["spec"];
  if(spec.is_scalar())spec=Config::parse_config_file(spec.as<std::string>());
  const Clock clock_ps=spec["clock_ps"].as<Clock>();
  const double tick_us=static_cast<double>(clock_ps)/1000000.0;
  auto us=[&](Clock c){return c*tick_us;};
  auto cycles=[&](double t){require(std::isfinite(t)&&t>=0,"invalid AE measured duration");return static_cast<Clock>(std::ceil(t/tick_us-1e-10));};
  std::array<std::unique_ptr<PIMController>,16> channels;
  for(int c=0;c<16;++c) {channels[c]=std::make_unique<PIMController>(c,spec,spec["metadata"]);channels[c]->fused=true;}
  int stride=0;std::string program_model,workload_model;
  auto programs=read_program_json(cfg["program_path"].as<std::string>(),&stride,&program_model,&workload_model);
  require(program_model=="fused_kernel_windows","kernel windows require a generated fused_kernel_windows program");
  require(stride>0&&stride%2048==0,"invalid layer stride");
  std::map<std::string,std::array<int,16>> descriptors;
  for(const auto& [name,stage]:programs) {
    auto& indices=descriptors[name];
    for(int c=0;c<16;++c)indices[c]=descriptor_index(stage.channels[c]);
    for(int c=1;c<16;++c)require((indices[c]>=0)==(indices[0]>=0),"descriptor phase differs across channels");
  }
  auto schedule=read_json(cfg["ae_schedule_path"].as<std::string>());
  require(workload_model==model&&schedule.value("model",std::string("pi05"))==model,
          "AE program and schedule must match the workload model");
  const int layer_capacity=spec["organization"]["row"].as<int>(32768)/(stride/2048);
  require(schedule.value("layers",0)>=0&&schedule.value("layers",0)<=layer_capacity,
          "AE layers exceed rank1 bank capacity");
  std::vector<CoarseEvent> events;
  for(const auto& e:schedule.at("events")) {
    CoarseEvent event;event.kind=e.at("kind");event.name=e.at(event.kind=="pim"?"stage":"name");
    event.step=e.value("step",0);event.layer=e.value("layer",0);event.duration=cycles(e.value("us",0.0));
    require(event.kind=="gpu"||event.layer<layer_capacity,"AE event exceeds rank1 bank capacity");
    require(event.kind=="gpu"||(event.kind=="pim"&&programs.count(event.name)),"invalid AE event");events.push_back(event);
  }
  // A descriptor boundary is not a GPU handoff when both sides contain only
  // private PIM work. Check the actual operations on every channel, so payload
  // uploads/exports and AE GPU helpers can never take this shortcut.
  std::vector<bool> continues(events.size(),false);
  for(size_t i=0;i+1<events.size();++i) {
    const auto& from=events[i];const auto& to=events[i+1];
    if(from.kind!="pim"||to.kind!="pim"||from.step!=to.step||from.layer!=to.layer)continue;
    bool safe=true;
    for(int c=0;c<16&&safe;++c) {
      int a=descriptors.at(from.name)[c],b=descriptors.at(to.name)[c];
      if(a<0||b<0){safe=false;break;}
      const auto& outgoing=programs.at(from.name).channels[c];
      const auto& incoming=programs.at(to.name).channels[c];
      auto barrier=[](const Operation& op){return op.name=="BARRIER";};
      safe=std::all_of(outgoing.begin()+a+1,outgoing.end(),barrier)&&
           std::all_of(incoming.begin(),incoming.begin()+b,barrier);
    }
    continues[i]=safe;
  }
  auto vlm=read_json(cfg["vlm_path"].as<std::string>());
  auto ledger=read_json(cfg["calibrated_vlm_path"].as<std::string>());
  require(ledger.at("kind")=="calibrated_vlm_durations"&&ledger.at("clock_ps")==clock_ps,"calibrated VLM ledger kind/clock mismatch");
  require(ledger.at("units").size()==vlm.at("units").size(),"calibrated VLM ledger count mismatch");
  std::vector<CalibratedKernel> kernels;std::set<int> ordinals;Clock calibrated_total=0;
  const double alpha=cfg["calibration"]["alpha"].as<double>(1.0);
  for(size_t i=0;i<vlm.at("units").size();++i) {
    const auto& v=vlm["units"][i];const auto& d=ledger["units"][i];CalibratedKernel k;
    k.ordinal=v.at("ordinal");k.name=v.at("name");k.kind=v.at("kind");
    require(d.at("ordinal")==k.ordinal&&d.at("name")==k.name&&d.at("kind")==k.kind&&ordinals.insert(k.ordinal).second,
            "calibrated VLM ledger identity mismatch");
    require(d.at("duration_cycles").is_number_integer(),"calibrated duration must be integer cycles");
    k.duration=d.at("duration_cycles").get<Clock>();require(k.duration>0,"calibrated duration must be positive");
    require(std::abs(d.at("us").get<double>()-us(k.duration))<2e-6,"calibrated duration cycle/us mismatch");
    k.reads=v.at("read_sectors").get<uint64_t>()*32;
    uint64_t writes=v.at("write_sectors");
    k.writes=static_cast<uint64_t>(v.value("write_is_proxy",true)?std::ceil(alpha*writes):writes)*32;
    calibrated_total+=k.duration;kernels.push_back(k);
  }
  require(ledger.at("total_cycles")==calibrated_total,"calibrated ledger total mismatch");
  Json stages=Json::array(),phases=Json::array(),ae_gpu=Json::array();
  Clock now=0,ae_end=0,stage_start=0,phase_start=0,ae_gpu_start=0,ae_gpu_end=NEVER,join_total=0;
  size_t ae_index=0,vlm_index=0;int active_kernel=-1,stage_id=-1;bool ae_recorded=false;
  Phase phase=Phase::None;std::array<Clock,16> next_tick{};
  auto all=[&](auto test){for(const auto& mc:channels)if(!test(*mc))return false;return true;};
  auto wake=[&](){next_tick.fill(now);};
  auto finish_phase=[&](const std::array<Clock,16>& ready){
    phases.push_back({{"stage_id",stage_id},{"phase",phase_name(phase)},{"start_ck",phase_start},{"end_ck",now},
                      {"start_us",us(phase_start)},{"end_us",us(now)},{"channel_ready_ck",ready}});
    phase_start=now;
  };
  auto change_phase=[&](Phase target,int controller_phase){phase=target;phase_start=now;for(auto& mc:channels)mc->coarse_phase=controller_phase;wake();};
  uint64_t iterations=0;
  while(true) {
    bool changed=true;
    while(changed) {
      changed=false;
      if(active_kernel>=0&&now>=kernels[active_kernel].start+kernels[active_kernel].duration) {
        kernels[active_kernel].end=now;active_kernel=-1;changed=true;
      }
      if(ae_gpu_end!=NEVER&&now>=ae_gpu_end) {
        const auto& e=events[ae_index];ae_gpu.push_back({{"name",e.name},{"step",e.step},{"layer",e.layer},
            {"start_ck",ae_gpu_start},{"end_ck",now},{"start_us",us(ae_gpu_start)},{"end_us",us(now)}});
        ae_gpu_end=NEVER;++ae_index;changed=true;
      }
      std::array<Clock,16> ready;ready.fill(now);
      if((phase==Phase::Upload||phase==Phase::Launch)&&all([&](const auto& mc){return mc.coarse_upload_ready()<=now;})) {
        for(int c=0;c<16;++c)ready[c]=channels[c]->coarse_upload_ready();
        finish_phase(ready);change_phase(Phase::Compute,1);changed=true;
      }
      if(phase==Phase::Compute)for(auto& mc:channels)mc->settle_coarse(now);
      if(phase==Phase::Compute&&all([&](const auto& mc){return mc.coarse_result_ready()<=now;})) {
        for(int c=0;c<16;++c)ready[c]=channels[c]->coarse_result_ready();
        finish_phase(ready);change_phase(continues[ae_index]?Phase::Continue:Phase::Join,1);changed=true;
      }
      if(phase==Phase::Continue||(phase==Phase::Join&&active_kernel<0)) {
        ready.fill(now);if(phase==Phase::Join)join_total+=now-phase_start;
        finish_phase(ready);change_phase(Phase::Download,2);changed=true;
      }
      if((phase==Phase::Download||phase==Phase::Transfer)&&all([&](const auto& mc){return mc.stage_finished(now);})) {
        for(int c=0;c<16;++c)ready[c]=std::max(channels[c]->top.tail,channels[c]->pim_dq_end);
        finish_phase(ready);const auto& e=events[ae_index];
        stages.push_back({{"stage_id",stage_id},{"name",e.name},{"step",e.step},{"layer",e.layer},
                         {"start_ck",stage_start},{"end_ck",now},
                         {"start_us",us(stage_start)},{"end_us",us(now)},{"duration_us",us(now-stage_start)}});
        for(auto& mc:channels){mc->checker.set_stage(now,-1);mc->stage=-1;mc->coarse_phase=-1;}
        phase=Phase::None;++ae_index;changed=true;
      }
      if(phase==Phase::None&&ae_gpu_end==NEVER&&ae_index<events.size()) {
        bool continuation=ae_index>0&&continues[ae_index-1];
        require(active_kernel<0||continuation,"new AE handoff started before the current VLM kernel ended");
        const auto& e=events[ae_index];
        if(e.kind=="gpu") {ae_gpu_start=now;ae_gpu_end=now+e.duration;}
        else {
          stage_id=static_cast<int>(stages.size());stage_start=now;phase_start=now;
          const auto& program=programs.at(e.name);const auto& indices=descriptors.at(e.name);
          for(int c=0;c<16;++c) {
            auto& mc=*channels[c];mc.row_offset=e.layer*(stride/2048);mc.set_stage(program.channels[c],stage_id,now,e.name);
            mc.coarse_descriptor=indices[c];mc.coarse_phase=indices[0]<0?-1:0;
          }
          // Native CA/decode still times this launch. An already-running VLM
          // kernel keeps its calibrated deadline across descriptor-only CA;
          // new VLM kernels are admitted once private computation starts.
          phase=indices[0]<0?Phase::Transfer:(continuation?Phase::Launch:Phase::Upload);wake();
        }
        changed=true;
      }
      if(ae_index==events.size()&&phase==Phase::None&&ae_gpu_end==NEVER&&!ae_recorded) {ae_end=now;ae_recorded=true;changed=true;}
      // Resolve all native completion boundaries before considering a new kernel.
      if(active_kernel<0&&vlm_index<kernels.size()&&(phase==Phase::Compute||ae_recorded)) {
        active_kernel=static_cast<int>(vlm_index++);kernels[active_kernel].start=now;changed=true;
      }
    }
    if(ae_recorded&&active_kernel<0&&vlm_index==kernels.size())break;
    for(int c=0;c<16;++c)if(next_tick[c]<=now)next_tick[c]=channels[c]->step(now,false);
    Clock next=*std::min_element(next_tick.begin(),next_tick.end());next=std::min(next,ae_gpu_end);
    if(active_kernel>=0)next=std::min(next,kernels[active_kernel].start+kernels[active_kernel].duration);
    require(next>=now&&next<NEVER,"kernel-window frontend deadlock");
    if(++iterations%10000000==0)std::cerr<<"progress ck="<<now<<" kernels="<<vlm_index<<" stages="<<stages.size()<<"\n";
    now=next;require(now<cycles(5000000),"simulation exceeded five seconds of model time");
  }
  Json raw={{"schema_version",3},{"execution_model","fused_kernel_windows"},{"clock_ps",clock_ps},
            {"window_policy","continuous_pim_descriptors"},
            {"vlm_memory_model","calibrated_kernel_durations; no per-request GPU DRAM simulation"},
            {"refresh_policy",{{"rank0","REFpb"},{"rank1","REFab"}}},{"stages",stages},{"phases",phases},{"ae_gpu",ae_gpu}};
  uint64_t pr=0,pw=0,commands=0,ooo=0,prefetch=0,vr=0,vw=0;
  Clock background=now;for(const auto& s:stages)background-=s.at("end_ck").get<Clock>()-s.at("start_ck").get<Clock>();
  raw["bus"]=Json::array();raw["kernels"]=Json::array();
  for(auto& mc:channels) {
    mc->checker.advance(now);pr+=mc->pim_read_bytes;pw+=mc->pim_write_bytes;commands+=mc->checker.checked;
    ooo+=mc->out_of_order_issues;prefetch+=mc->prefetched_prerequisites;
    require(mc->gpu_read_bytes==0&&mc->gpu_write_bytes==0,"kernel-window mode issued GPU transactions");
    for(const auto& [sid,s]:mc->checker.stats) {
      Clock duration=sid<0?background:stages.at(sid).at("end_ck").get<Clock>()-stages.at(sid).at("start_ck").get<Clock>();
      raw["bus"].push_back({{"channel",mc->channel},{"stage_id",sid},{"ca_cycles",s.ca},{"dq_cycles",s.dq},
        {"read_bytes",s.read},{"write_bytes",s.write},{"gpu_read_bytes",s.gpu_read},{"gpu_write_bytes",s.gpu_write},
        {"pim_read_bytes",s.pim_read},{"pim_write_bytes",s.pim_write},{"duration_cycles",duration}});
    }
  }
  for(int rank=0;rank<2;++rank) {
    uint64_t pb=0,ab=0;for(const auto& mc:channels){pb+=mc->refresh_counts[rank][0];ab+=mc->refresh_counts[rank][1];}
    raw["refresh_commands"]["rank"+std::to_string(rank)]={{"REFpb",pb},{"REFab",ab}};
  }
  for(const auto& k:kernels) {
    vr+=k.reads;vw+=k.writes;raw["kernels"].push_back({{"ordinal",k.ordinal},{"name",k.name},{"kind",k.kind},
      {"start_ck",k.start},{"end_ck",k.end},{"duration_cycles",k.duration},{"start_us",us(k.start)},
      {"end_us",us(k.end)},{"duration_us",us(k.duration)},{"read_bytes",k.reads},{"write_bytes",k.writes}});
  }
  raw["metrics"]={{"total_us",us(now)},{"AE_elapsed_us",us(ae_end)},{"AE_GPU_wait_us",0},
    {"AE_VLM_join_us",us(join_total)},{"VLM_remaining_us",us(now-ae_end)},
    {"vlm_calibrated_sum_us",us(calibrated_total)},{"vlm_accounted_read_bytes",vr},{"vlm_accounted_write_bytes",vw},
    {"gpu_read_bytes",0},{"gpu_write_bytes",0},{"pim_read_bytes",pr},{"pim_write_bytes",pw},
    {"checked_commands",commands},{"PIM_out_of_order_issues",ooo},{"PIM_prefetched_prerequisites",prefetch},
    {"PIM_descriptor_continuations",std::count(continues.begin(),continues.end(),true)},
    {"serial_epoch_reset",false},{"frontend_iterations",iterations}};
  for(const char* key:{"GPU_act1","GPU_act2","GPU_prepb","GPU_preab","GPU_reads","GPU_writes","GPU_row_hits","GPU_row_misses"})raw["metrics"][key]=0;
  std::filesystem::create_directories(output);std::ofstream out(output+"/raw.json");out<<raw.dump(2)<<'\n';
  std::cout<<"completed fused_kernel_windows total_us="<<us(now)<<" commands="<<commands<<"\n";
  return 0;
}
}
