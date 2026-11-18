#pragma once
// Bounded data lifetimes, deliberately free of DRAM timing constants.
// The controller supplies completion timestamps obtained from its one spec.
#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

namespace PimVla {

struct ResourceRegion { int bank=0, byte_offset=0, bytes=0; };

class PimResources {
 public:
  using Time = int64_t;
  static constexpr int BANKS=16, SRF_SLOTS=32, OUTPUT_WORDS=512;

  void start_stage(const std::string& name, Time now) {
    for (const auto& bank: output_) for (const auto& w:bank)
      ensure(w.busy<=now && w.ready<=now,"stage boundary precedes output completion");
    for (const auto& bank:srf_) for (const auto& s:bank)
      ensure(s.busy<=now && s.ready<=now,"stage boundary precedes SRF completion");
    for(int b=0;b<BANKS;++b) {
      ensure(coef_[b].busy<=now && coef_[b].ready<=now,"stage boundary precedes coefficient completion");
      for(int s=0;s<2;++s) {
        ensure(rope_input_[b][s].busy<=now && rope_input_[b][s].ready<=now,
               "stage boundary precedes RoPE input completion");
        ensure(rope_output_[b][s].busy<=now && rope_output_[b][s].ready<=now,
               "stage boundary precedes RoPE output completion");
      }
      ensure(rope_fifo_[b].empty(),"stage boundary precedes RoPE FIFO consumption");
    }
    stage_=name; ++epoch_;
    for (auto& regions:last_gather_) regions.clear();
    // Q stays in each bank's SRF across export/import and QKT stages.
    // Output values retain their identity but cannot masquerade as new work.
  }

  template<class Op> Time ready(const Op& o, Time now, uint16_t bank_mask=0xffff) const {
    validate(o);
    Time at=now;
    if (o.name=="WRGB" || o.name=="BCAST") {
      if (o.source=="host_query") at=std::max(at,srf_[bank(o)][o.slot].busy);
      else for (int b=0;b<BANKS;++b) at=std::max(at,srf_[b][o.slot].busy);
      if (o.name=="BCAST") {
        ensure(o.source_bank>=0 && o.source_bank<BANKS,"BCAST source bank invalid");
        const auto& src=srf_[o.source_bank][o.slot];
        ensure(src.valid && src.epoch==epoch_ && src.fragment==o.tensor_fragment,
               "BCAST source SRF was not filled by this fragment");
        at=std::max(at,src.ready);
      }
    } else if (o.name=="LOCALRD") {
      const int b=bank(o);
      if (o.dest_slot>=0) at=std::max(at,srf_[b][o.dest_slot].busy);
      else if (o.dest=="rope_coefficient") at=std::max(at,coef_[b].busy);
      else if (o.dest=="rope_input") {
        ensure(o.slot<2,"RoPE has two input vectors");
        at=std::max(at,rope_input_[b][o.slot].busy);
      }
    } else if (o.name=="MACAB" || o.name=="MACQK") {
      for (int b=0;b<BANKS;++b) {
        if(!(bank_mask&(1u<<b)))continue;
        for (int s=o.slot;s<o.slot+o.rows;++s) {
          const auto& input=srf_[b][s];
          ensure(input.valid,"MAC reads an unfilled SRF slot");
          if (o.name=="MACQK") ensure(input.query,"MACQK input is not a resident Q vector");
          else ensure(input.epoch==epoch_ && input.fragment==o.fragment,
                      "MAC reads a stale or different SRF fragment");
          at=std::max(at,input.ready);
        }
        for (int q=0;q<o.rows;++q) {
          int index=q*o.output_columns+o.output_column;
          const auto& out=output_[b][index];
          at=std::max(at,out.busy);
          if(o.name=="MACQK" && out.generation>>32==epoch_)
            ensure(!out.valid,"MACQK overwrites an unconsumed contribution");
          if(o.name=="MACAB" && o.fragment>0) {
            ensure(out.valid && out.generation==generation(o) && out.fragment==o.fragment-1,
                   "MAC accumulation has no preceding fragment output");
            at=std::max(at,out.ready);
          }
        }
      }
    } else if(o.name=="REDUCE") {
      ensure(o.elements==16,"QK reduction must produce sixteen FP32 scores");
      for(int b=0;b<BANKS;++b) for(int q=0;q<16;++q) {
        const auto& w=output_[b][q];
        ensure(w.valid && w.generation==reduction_generation(o.kv_token),
               "REDUCE reads an unproduced QK contribution");
        at=std::max(at,w.ready);
      }
      for(int b=0;b<BANKS;++b) {
        const auto& w=output_[b][o.reduction_word];
        if(w.generation>>32==epoch_) ensure(!w.valid,"REDUCE overwrites unread scores");
        at=std::max(at,w.busy);
      }
    } else if(o.name=="GATHER") {
      ensure(!o.source_regions.empty(),"GATHER must identify output-pool source regions");
      at=read_regions(o,o.source_regions,at);
    } else if(o.name=="CAST") {
      const int b=bank(o);
      ensure(!last_gather_[b].empty(),"CAST has no gathered output values");
      at=read_regions(o,last_gather_[b],at);
    } else if(o.name=="RDRES" || (o.name=="LOCALWR" && !o.source_regions.empty())) {
      ensure(!o.source_regions.empty(),"result transfer must identify produced output words");
      at=read_regions(o,o.source_regions,at);
    } else if(o.name=="ROPE_EXEC") {
      const int b=bank(o);
      ensure(o.slot<2,"RoPE has two input vectors");
      const auto& input=rope_input_[b][o.slot];const auto& coef=coef_[b];
      ensure(input.valid && input.epoch==epoch_ && input.fragment==o.token,
             "RoPE executes without its input vector");
      ensure(coef.valid && coef.epoch==epoch_ && coef.fragment==o.token,
             "RoPE executes without matching coefficients");
      const auto& output=rope_output_[b][o.slot];
      ensure(!output.valid,"RoPE overwrites an unconsumed output vector");
      ensure(rope_fifo_[b].size()<2,"RoPE output FIFO exceeds two entries");
      // Bank-local vector initiation interval is imposed by the native spec.
      // Completion protects the source slot, not every other bank or slot.
      at=std::max({at,input.ready,coef.ready,output.busy});
    } else if(o.name=="LOCALWR" && o.source=="rope_output") {
      const int b=bank(o);
      ensure(o.slot<2,"RoPE has two output vectors");
      const auto& w=rope_output_[b][o.slot];
      ensure(w.valid && w.epoch==epoch_ && w.fragment==o.token,
             "RoPE store precedes execution or rereads a released output");
      ensure(!rope_fifo_[b].empty() && rope_fifo_[b].front()==o.slot,
             "RoPE store violates output FIFO order");
      at=std::max(at,w.ready);
    }
    return at;
  }

  template<class Op> void issue(const Op& o, Time now, Time complete, uint16_t bank_mask=0xffff) {
    ensure(complete>=now,"resource completion precedes issue");
    ensure(ready(o,now,bank_mask)<=now,"issued an operation before resource readiness");
    if(o.name=="WRGB" || o.name=="BCAST") {
      if(o.source=="host_query") fill_srf(bank(o),o.slot,o.slot,true,complete);
      else for(int b=0;b<BANKS;++b) fill_srf(b,o.slot,o.tensor_fragment,false,complete);
    } else if(o.name=="LOCALRD") {
      const int b=bank(o);
      if(o.dest_slot>=0) fill_srf(b,o.dest_slot,o.tensor_fragment,
                                 stage_.rfind("q_exchange",0)==0,complete);
      else if(o.dest=="rope_coefficient") fill_vector(coef_[b],o.token,complete);
      else if(o.dest=="rope_input") fill_vector(rope_input_[b][o.slot],o.token,complete);
    } else if(o.name=="MACAB" || o.name=="MACQK") {
      for(int b=0;b<BANKS;++b) {
        if(!(bank_mask&(1u<<b)))continue;
        for(int s=o.slot;s<o.slot+o.rows;++s) srf_[b][s].busy=std::max(srf_[b][s].busy,complete);
        for(int q=0;q<o.rows;++q) {
          auto& out=output_[b][q*o.output_columns+o.output_column];
          out.valid=true;out.ready=complete;out.generation=generation(o);out.fragment=o.fragment;
        }
      }
    } else if(o.name=="REDUCE") {
      for(int b=0;b<BANKS;++b) {
        for(int q=0;q<16;++q) {
          auto& contribution=output_[b][q];
          contribution.busy=std::max(contribution.busy,complete);
          contribution.valid=false;
        }
        // Query b resides in bank b: 64B scratch + 496 FP32 scores = 2KiB.
        auto& w=output_[b][o.reduction_word];
        w.valid=true;w.ready=complete;w.generation=reduction_generation(o.kv_token);
      }
    } else if(o.name=="GATHER") {
      auto& regions=last_gather_[bank(o)];regions.clear();
      for(const auto& r:o.source_regions) regions.push_back({r.bank,r.byte_offset,r.bytes});
      lock_regions(regions,complete,false);
    } else if(o.name=="CAST") {
      lock_regions(last_gather_[bank(o)],complete,false);
      for(const auto& r:last_gather_[bank(o)])
        for(int i=r.byte_offset/4;i<(r.byte_offset+r.bytes)/4;++i)
          output_[r.bank][i].ready=std::max(output_[r.bank][i].ready,complete);
    } else if(o.name=="RDRES" || (o.name=="LOCALWR" && !o.source_regions.empty())) {
      lock_regions(o.source_regions,complete,o.release_result);
    } else if(o.name=="ROPE_EXEC") {
      const int b=bank(o);auto& input=rope_input_[b][o.slot];
      input.busy=complete;input.valid=false;coef_[b].busy=std::max(coef_[b].busy,complete);
      fill_vector(rope_output_[b][o.slot],o.token,complete);
      rope_fifo_[b].push_back(o.slot);
    } else if(o.name=="LOCALWR" && o.source=="rope_output") {
      int b=bank(o);auto& w=rope_output_[b][o.slot];w.valid=false;w.busy=complete;
      rope_fifo_[b].pop_front();
    }
  }

 private:
  struct Slot {bool valid=false,query=false;Time ready=0,busy=0;uint64_t epoch=0;int fragment=-1;};
  struct Word {bool valid=false;Time ready=0,busy=0;uint64_t generation=0;int fragment=-1;};
  std::array<std::array<Slot,SRF_SLOTS>,BANKS> srf_{};
  std::array<std::array<Word,OUTPUT_WORDS>,BANKS> output_{};
  std::array<Slot,BANKS> coef_{};
  std::array<std::array<Slot,2>,BANKS> rope_input_{},rope_output_{};
  std::array<std::deque<int>,BANKS> rope_fifo_{};
  std::array<std::vector<ResourceRegion>,BANKS> last_gather_{};
  std::string stage_;
  uint64_t epoch_=0;

  static void ensure(bool ok,const char* message) {
    if(!ok) throw std::runtime_error(std::string("PIM resource violation: ")+message);
  }
  template<class Op> static int bank(const Op& o) {
    ensure(o.addr.size()==6 && o.addr[2]>=0 && o.addr[2]<4 && o.addr[3]>=0 && o.addr[3]<4,
           "local resource operation needs one physical bank");
    return o.addr[2]*4+o.addr[3];
  }
  template<class Op> void validate(const Op& o) const {
    ensure(o.slot>=0 && o.slot<32,"SRF slot outside 1KiB");
    ensure(o.dest_slot>=-1 && o.dest_slot<32,"SRF destination outside 1KiB");
    if(o.source=="host_query")
      ensure(o.name=="WRGB" && o.slot<16,"Host Q write must target a query SRF slot");
    if(o.name=="MACAB" || o.name=="MACQK") {
      ensure(o.rows>0 && o.slot+o.rows<=32,"MAC input exceeds 1KiB SRF");
      ensure(o.output_columns>0 && o.output_column>=0 && o.output_column<o.output_columns &&
             o.rows*o.output_columns<=OUTPUT_WORDS,"MAC output exceeds 2KiB pool");
      ensure(o.fragment>=0 && o.output_tile>=0,"invalid MAC fragment/tile");
    }
    if(o.name=="REDUCE")
      ensure(o.reduction_word>=16 && o.reduction_word<OUTPUT_WORDS,
             "reduction result overlaps scratch or exceeds 2KiB pool");
    for(const auto& r:o.source_regions) {
      ensure(r.bank>=0 && (r.bank)<BANKS && r.byte_offset>=0 && r.bytes>0 &&
             r.byte_offset%4==0 && r.bytes%4==0 && r.byte_offset+r.bytes<=2048,
             "source region outside FP32 output pool");
    }
  }
  uint64_t reduction_generation(int token) const {
    return (epoch_<<32)|0x80000000ULL|static_cast<uint32_t>(token);
  }
  template<class Op> uint64_t generation(const Op& o) const {
    return o.name=="MACQK"?reduction_generation(o.kv_token):(epoch_<<32)|static_cast<uint32_t>(o.output_tile);
  }
  template<class Op,class Regions> Time read_regions(const Op& o,const Regions& regions,Time at) const {
    const uint64_t expected=o.source=="reduction_output"?reduction_generation(o.kv_token):
                            (epoch_<<32)|static_cast<uint32_t>(o.output_tile);
    for(const auto& r:regions) for(int i=r.byte_offset/4;i<(r.byte_offset+r.bytes)/4;++i) {
      const auto& w=output_[r.bank][i];
      ensure(w.valid && w.generation==expected,"read of unproduced, stale, or released output word");
      at=std::max({at,w.ready,w.busy});
    }
    return at;
  }
  template<class Regions> void lock_regions(const Regions& regions,Time complete,bool release) {
    for(const auto& r:regions) for(int i=r.byte_offset/4;i<(r.byte_offset+r.bytes)/4;++i) {
      auto& w=output_[r.bank][i];w.busy=std::max(w.busy,complete);
      if(release) w.valid=false;
    }
  }
  void fill_srf(int b,int s,int fragment,bool query,Time complete) {
    auto& v=srf_[b][s];v.valid=true;v.query=query;v.epoch=epoch_;v.fragment=fragment;v.ready=complete;
  }
  void fill_vector(Slot& v,int token,Time complete) {
    v.valid=true;v.epoch=epoch_;v.fragment=token;v.ready=complete;
  }
};
} // namespace PimVla
