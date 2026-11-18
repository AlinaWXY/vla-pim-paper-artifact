#include "ramulator/controller/impl/pim_resources.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using namespace PimVla;

struct Op {
  std::string name,dest,source;
  std::vector<int> addr{0,1,0,0,0,0};
  int slot=0,dest_slot=-1,source_bank=0,tensor_fragment=0,rows=1;
  int output_columns=1,output_column=0,fragment=0,output_tile=0,kv_token=0;
  int elements=16,result_slot=0,token=0,reduction_word=16;
  bool release_result=true;
  std::vector<ResourceRegion> source_regions;
};
template<class F> void rejects(F f,const std::string& text) {
  try {f();assert(false && "expected resource guard");}
  catch(const std::runtime_error& e) {assert(std::string(e.what()).find(text)!=std::string::npos);}
}
Op make(const std::string& name) {Op o;o.name=name;return o;}

void set_bank(Op& op,int bank) {op.addr[2]=bank/4;op.addr[3]=bank%4;}
void resident_queries(PimResources& resource) {
  resource.start_stage("q_exchange_export",0);
  for(int bank=0;bank<16;++bank)for(int slot=0;slot<16;++slot) {
    auto load=make("LOCALRD");set_bank(load,bank);load.dest_slot=slot;
    resource.issue(load,0,1);
  }
  resource.start_stage("q_exchange_import",1);
  resource.start_stage("qkt",1);
}
Op score_packet(int token,int word,int first_bank=0) {
  auto op=make("RDRES");op.source="reduction_output";op.kv_token=token;
  for(int bank=first_bank;bank<first_bank+8;++bank)op.source_regions.push_back({bank,word*4,4});
  return op;
}

int main(int argc,char** argv) {
  if(argc==2) {
    std::ifstream input(argv[1]);assert(input.good());
    PimResources resource;PimResources::Time now=0;std::string line;int count=0;
    while(std::getline(input,line)) {
      std::istringstream row(line);std::string label;row>>label;
      if(label=="stage") {row>>label;resource.start_stage(label,now);continue;}
      Op op;op.name=label;int regions=0;
      row>>op.dest>>op.source>>op.addr[2]>>op.addr[3]>>op.slot>>op.dest_slot
         >>op.source_bank>>op.tensor_fragment>>op.rows>>op.output_columns>>op.output_column
         >>op.fragment>>op.output_tile>>op.kv_token>>op.elements>>op.result_slot>>op.token
         >>op.release_result>>op.reduction_word>>regions;
      if(row.fail()) {std::cerr<<"malformed resource fixture: "<<line<<'\n';return 1;}
      if(op.dest=="-")op.dest.clear();
      if(op.source=="-")op.source.clear();
      for(int i=0;i<regions;++i) {ResourceRegion r;row>>r.bank>>r.byte_offset>>r.bytes;op.source_regions.push_back(r);}
      try {now=resource.ready(op,now);resource.issue(op,now,now+1);++now;++count;}
      catch(const std::exception& e) {std::cerr<<"request "<<count<<" "<<line<<"\n"<<e.what()<<"\n";return 1;}
    }
    std::cout<<"PimResources accepts "<<count<<" generated requests\n";return 0;
  }
  // A legal input broadcast does not become consumable until its completion.
  PimResources r;r.start_stage("qkv",0);
  auto mac=make("MACAB");
  rejects([&]{r.ready(mac,0);},"unfilled SRF");
  auto wr=make("WRGB");r.issue(wr,0,14);
  assert(r.ready(mac,1)==14);
  rejects([&]{r.issue(mac,13,17);},"before resource readiness");
  r.issue(mac,14,18);
  // Output word production and rank DQ tail are independent.
  auto rd=make("RDRES");rd.source_regions={{0,0,4}};
  assert(r.ready(rd,15)==18);
  auto missing=rd;missing.source_regions={{0,4,4}};
  rejects([&]{r.ready(missing,18);},"unproduced");
  r.issue(rd,18,43);
  assert(r.ready(mac,19)==43);
  rejects([&]{r.issue(mac,19,23);},"before resource readiness");
  rejects([&]{r.ready(rd,43);},"released");
  r.issue(mac,43,47); // reinitialize after the old DQ tail is gone

  // Fragment accumulation needs both the current input and its previous sum.
  auto f2=mac;f2.fragment=2;
  auto input2=wr;input2.tensor_fragment=2;
  r.issue(input2,47,61);
  rejects([&]{r.ready(f2,61);},"preceding fragment");
  auto f1=mac;f1.fragment=1;
  rejects([&]{r.ready(f1,61);},"different SRF fragment");
  auto input1=wr;input1.tensor_fragment=1;r.issue(input1,61,75);
  r.issue(f1,75,79);

  // A naked LOCALRD into a scratch must never silently fill an SRF slot.
  PimResources scratch;scratch.start_stage("qkv",0);
  auto local=make("LOCALRD");local.dest="rmw_scratch";scratch.issue(local,0,4);
  rejects([&]{scratch.ready(mac,4);},"unfilled SRF");
  local.dest_slot=0;scratch.issue(local,4,8);
  auto broadcast=make("BCAST");assert(scratch.ready(broadcast,5)==8);
  scratch.issue(broadcast,8,10);scratch.issue(mac,10,14);
  auto too_big=mac;too_big.rows=16;too_big.output_columns=33;
  rejects([&]{scratch.ready(too_big,14);},"2KiB");
  auto bad_slot=wr;bad_slot.slot=32;
  rejects([&]{scratch.ready(bad_slot,14);},"1KiB");

  // Resident Q is intentionally preserved across global stage boundaries.
  PimResources q;resident_queries(q);
  auto qk=make("MACQK");qk.rows=16;q.issue(qk,1,17);
  auto reduce=make("REDUCE");assert(q.ready(reduce,2)==17);q.issue(reduce,17,19);
  auto anonymous_score=make("RDRES");anonymous_score.source="reduction_output";
  rejects([&]{q.ready(anonymous_score,19);},"identify produced output words");
  auto score=score_packet(0,16);
  auto wrong_token=score;wrong_token.kv_token=7;
  rejects([&]{q.ready(wrong_token,19);},"stale");
  q.issue(score,19,44);
  // Score word16 does not occupy the sixteen contribution scratch words.
  qk.kv_token=1;assert(q.ready(qk,20)==20);q.issue(qk,20,36);
  auto second_packet=score_packet(0,16,8);q.issue(second_packet,23,48);
  reduce.kv_token=1;
  assert(q.ready(reduce,36)==48); // Reusing word16 waits for both DQ tails.
  rejects([&]{q.issue(reduce,47,49);},"before resource readiness");
  q.issue(reduce,48,50);
  qk.kv_token=2;assert(q.ready(qk,49)==50);q.issue(qk,50,66);
  reduce.kv_token=2;
  rejects([&]{q.ready(reduce,66);},"unread scores");
  q.start_stage("pv",66);
  auto stale=score_packet(1,16);
  rejects([&]{q.ready(stale,66);},"stale");

  // All 496 score words fit in the existing 2KiB bank output pool, while
  // words0..15 remain contribution scratch. No separate rank score cache.
  PimResources scores;resident_queries(scores);
  PimResources::Time time=1;
  for(int token=0;token<496;++token) {
    auto contribution=make("MACQK");contribution.rows=16;contribution.kv_token=token;
    auto reduction=make("REDUCE");reduction.kv_token=token;reduction.reduction_word=16+token;
    scores.issue(contribution,time,time+1);
    scores.issue(reduction,time+1,time+2);time+=2;
  }
  assert(scores.ready(score_packet(0,16),time)==time);
  assert(scores.ready(score_packet(495,511,8),time)==time);
  auto bad_reduction=make("REDUCE");bad_reduction.reduction_word=15;
  rejects([&]{scores.ready(bad_reduction,time);},"overlaps scratch");
  bad_reduction.reduction_word=512;
  rejects([&]{scores.ready(bad_reduction,time);},"exceeds 2KiB");
  auto overflow=score_packet(495,511);overflow.source_regions={{0,2044,8}};
  rejects([&]{scores.ready(overflow,time);},"outside FP32");
  overflow.source_regions={{0,65,4}};
  rejects([&]{scores.ready(overflow,time);},"outside FP32");
  auto next_contribution=make("MACQK");next_contribution.rows=16;next_contribution.kv_token=496;
  scores.issue(next_contribution,time,time+1);
  auto wrap=make("REDUCE");wrap.kv_token=496;wrap.reduction_word=16;
  rejects([&]{scores.ready(wrap,time+1);},"unread scores");

  // RoPE input/output FIFO and execution resources are per bank. Independent
  // banks can execute at the same clock; native spec enforces each bank's II.
  PimResources rope;rope.start_stage("rope",0);
  auto execute=make("ROPE_EXEC");
  rejects([&]{rope.ready(execute,0);},"without its input");
  auto coefficient=make("LOCALRD");coefficient.dest="rope_coefficient";rope.issue(coefficient,0,2);
  auto input=make("LOCALRD");input.dest="rope_input";rope.issue(input,0,2);
  auto coefficient1=coefficient,input1b=input,execute1=execute;
  set_bank(coefficient1,1);set_bank(input1b,1);set_bank(execute1,1);
  rope.issue(coefficient1,0,2);rope.issue(input1b,0,2);
  rope.issue(execute,2,13);
  assert(rope.ready(execute1,2)==2);rope.issue(execute1,2,13);
  auto store=make("LOCALWR");store.source="rope_output";
  assert(rope.ready(store,3)==13);
  rope.issue(store,13,17);
  auto store1=store;set_bank(store1,1);rope.issue(store1,13,17);
  rejects([&]{rope.ready(store,17);},"released output");
  rejects([&]{rope.ready(mac,17);},"unfilled SRF");
  rejects([&]{rope.start_stage("next",16);},"RoPE output completion");
  rope.start_stage("next",17);

  PimResources fifo;fifo.start_stage("rope",0);
  coefficient.token=7;input.token=7;execute.token=7;store.token=7;
  fifo.issue(coefficient,0,2);fifo.issue(input,0,2);
  auto input_slot1=input;input_slot1.slot=1;fifo.issue(input_slot1,0,2);
  fifo.issue(execute,2,13);
  auto execute_slot1=execute;execute_slot1.slot=1;
  assert(fifo.ready(execute_slot1,10)==10);fifo.issue(execute_slot1,10,21);
  // Two occupied entries in bank0 do not consume bank1's FIFO capacity.
  coefficient1.token=input1b.token=execute1.token=7;
  fifo.issue(coefficient1,10,12);fifo.issue(input1b,10,12);
  assert(fifo.ready(execute1,12)==12);fifo.issue(execute1,12,23);
  fifo.issue(input,13,15);
  rejects([&]{fifo.ready(execute,15);},"unconsumed output");
  auto invalid_slot=execute;invalid_slot.slot=2;
  rejects([&]{fifo.ready(invalid_slot,21);},"two input vectors");
  auto store_slot1=store;store_slot1.slot=1;
  rejects([&]{fifo.ready(store_slot1,21);},"FIFO order");
  fifo.issue(store,21,25);
  assert(fifo.ready(execute,22)==25); // Dequeue does not free an in-flight store.
  rejects([&]{fifo.issue(execute,24,35);},"before resource readiness");
  fifo.issue(store_slot1,25,29);fifo.issue(execute,25,36);
  assert(fifo.ready(coefficient,26)==36); // Shared coefficient lifetime covers both slots.
  rejects([&]{fifo.start_stage("next",100);},"FIFO consumption");
  store1.token=7;fifo.issue(store1,29,33);fifo.issue(store,36,40);
  fifo.start_stage("next",40);

  PimResources mismatch;mismatch.start_stage("rope",0);
  coefficient.token=0;input.token=1;execute.token=1;
  mismatch.issue(coefficient,0,2);mismatch.issue(input,0,2);
  rejects([&]{mismatch.ready(execute,2);},"matching coefficients");
  std::cout<<"PimResources lifecycle tests passed\n";
}
